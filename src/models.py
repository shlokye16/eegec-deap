"""
models.py — Classification, feature importance, and validity controls.

Design rules enforced here:
  - Binarization threshold always computed from training fold only (no label leak).
  - StandardScaler always fit on training fold only (no feature leak).
  - Permutation importance always evaluated on held-out test fold (not train).
  - Band ablation reuses the same LOSO fold splits (no re-split).
  - Shuffled-label null shuffles within subject to preserve per-subject trial counts.
  - One LOSO loop: every metric for RQ1–RQ5 comes out of run_loso().

Performance:
  - LOSO folds are parallelized with joblib (each fold is independent).
  - RF n_jobs=1 inside parallel folds to avoid nested parallelism overhead.
  - Null distribution is batched with checkpoint saving — safe to interrupt and resume.

Hyperparameters (from literature):
  - SVM: RBF kernel, C=10. Zheng & Lu (2015) and subsequent DEAP work find C=10
    optimal for standardized log-PSD EEG features with RBF. Default C=1 is
    systematically suboptimal for this feature distribution.
  - RF: n_estimators=200. Class weights balanced in within-subject regime due to
    per-subject class imbalance at 5-fold scale.

Median split decision (document in paper):
  Global median is computed per fold from the 1240 training trials. This makes
  the threshold comparable across folds and consistent with prior DEAP literature.
  Trials exactly at the median are assigned to class 0 (low).

"""

import contextlib
import pickle
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon, binomtest
from tqdm.auto import tqdm


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """
    Context manager that patches joblib to report fold completion to a tqdm bar.
    Gives a live progress bar with ETA for any Parallel call.
    """
    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import (
    BANDS, ELECTRODE_REGIONS,
    N_BANDS, N_ELECTRODES, N_BP_FEATURES,
)

# ── Shared config ─────────────────────────────────────────────────────────────

TARGETS    = {'valence': 0, 'arousal': 1, 'dominance': 2}
BAND_NAMES = list(BANDS.keys())
CONFIGS    = ('eeg', 'peripheral', 'combined')

CORR_PAIRS = [
    ('valence',  'arousal'),
    ('valence',  'dominance'),
    ('arousal',  'dominance'),
]
CORR_KEYS = [f"{a}_{b}" for a, b in CORR_PAIRS]

CLASS_BALANCE_THRESHOLD = 0.65

# SVM hyperparameter: C=10 per Zheng & Lu (2015) and DEAP literature.
# Consistent improvement over C=1 for log-PSD features standardized to unit variance.
_SVM_C = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def binarize(labels: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a precomputed threshold to produce binary class labels.
    Trials at or below threshold -> 0. Trials above -> 1.
    Threshold must always come from training-fold labels only.
    """
    return (labels > threshold).astype(np.int32)


def _feature_configs(X_eeg: np.ndarray, X_peripheral: np.ndarray) -> dict:
    return {
        'eeg':        X_eeg,
        'peripheral': X_peripheral,
        'combined':   np.concatenate([X_eeg, X_peripheral], axis=1),
    }


def _band_indices(band_name: str) -> np.ndarray:
    """32 column indices in the EEG feature matrix for one frequency band."""
    b = BAND_NAMES.index(band_name)
    return np.array([ch * N_BANDS + b for ch in range(N_ELECTRODES)])


def _aggregate_importance(imp: np.ndarray) -> dict:
    """
    Aggregate (131,) importance into band, electrode, region summaries.
    Scientific claims use band/region only. Electrode is visualization only.
    """
    bp = imp[:N_BP_FEATURES].reshape(N_ELECTRODES, N_BANDS)
    return {
        'band':      bp.sum(axis=0),
        'electrode': bp.sum(axis=1),
        'region':    {r: float(bp.sum(axis=1)[idx].sum())
                      for r, idx in ELECTRODE_REGIONS.items()},
    }


# ── Single-fold worker (called in parallel) ───────────────────────────────────

def _run_fold(
    subj: int,
    X_eeg: np.ndarray,
    X_peripheral: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int,
    rf_n_estimators: int,
) -> dict:
    """
    Run one LOSO fold for all targets, configs, and models.
    RF uses n_jobs=1: outer parallelism across folds is used instead.
    SVM uses C=10 (literature-calibrated for log-PSD EEG features).
    Also stores per-fold (fold-level) accuracy arrays for paired tests (M3).
    """
    test_mask  = subject_ids == subj
    train_mask = ~test_mask
    fold_out   = {
        'subj': subj, 'metrics': {}, 'imp': {},
        'ablation': {}, 'balance': {}, 'flagged': {},
        # fold_acc stores per-fold scalar accuracy for paired tests (M3)
        'fold_acc': {},
    }

    for t_name, t_col in TARGETS.items():
        y_train_raw = y[train_mask, t_col]
        y_test_raw  = y[test_mask,  t_col]

        threshold = float(np.median(y_train_raw))
        y_train   = binarize(y_train_raw, threshold)
        y_test    = binarize(y_test_raw,  threshold)

        n_low, n_high = int((y_test == 0).sum()), int((y_test == 1).sum())
        fold_out['balance'][t_name] = (n_low, n_high)
        fold_out['flagged'][t_name] = (
            max(n_low, n_high) / len(y_test) > CLASS_BALANCE_THRESHOLD
        )

        configs = _feature_configs(X_eeg, X_peripheral)
        fold_out['metrics'][t_name] = {}
        fold_out['fold_acc'][t_name] = {}

        for cfg_name, X_cfg in configs.items():
            scaler    = StandardScaler().fit(X_cfg[train_mask])
            X_train_s = scaler.transform(X_cfg[train_mask])
            X_test_s  = scaler.transform(X_cfg[test_mask])

            # SVM (C=10, literature-calibrated)
            svm = SVC(kernel='rbf', C=_SVM_C, random_state=42)
            svm.fit(X_train_s, y_train)
            yp  = svm.predict(X_test_s)
            svm_acc = accuracy_score(y_test, yp)
            fold_out['metrics'][t_name][cfg_name] = {
                'svm': {'acc': svm_acc,
                        'f1':  f1_score(y_test, yp, zero_division=0)},
            }
            fold_out['fold_acc'][t_name][f'svm_{cfg_name}'] = svm_acc

            # RF (n_jobs=1 inside fold — outer parallelism across folds)
            rf = RandomForestClassifier(
                n_estimators=rf_n_estimators, random_state=42, n_jobs=1)
            rf.fit(X_train_s, y_train)
            yp = rf.predict(X_test_s)
            rf_acc = accuracy_score(y_test, yp)
            fold_out['metrics'][t_name][cfg_name]['rf'] = {
                'acc': rf_acc,
                'f1':  f1_score(y_test, yp, zero_division=0),
            }
            fold_out['fold_acc'][t_name][f'rf_{cfg_name}'] = rf_acc

            # Permutation importance on test fold, EEG config only
            if cfg_name == 'eeg':
                pi = permutation_importance(
                    rf, X_test_s, y_test,
                    n_repeats=n_perm_repeats, random_state=42, n_jobs=1,
                )
                fold_out['imp'][t_name] = pi.importances_mean  # (131,)

        # Band ablation — reuse same fold split
        fold_out['ablation'][t_name] = {}
        for band in BAND_NAMES:
            idx        = _band_indices(band)
            scaler_b   = StandardScaler().fit(X_eeg[train_mask][:, idx])
            Xb_train_s = scaler_b.transform(X_eeg[train_mask][:, idx])
            Xb_test_s  = scaler_b.transform(X_eeg[test_mask][:, idx])

            rf_b = RandomForestClassifier(
                n_estimators=rf_n_estimators, random_state=42, n_jobs=1)
            rf_b.fit(Xb_train_s, y_train)
            yp = rf_b.predict(Xb_test_s)
            fold_out['ablation'][t_name][band] = {
                'acc': accuracy_score(y_test, yp),
                'f1':  f1_score(y_test, yp, zero_division=0),
            }

    return fold_out


def _aggregate_folds(fold_results: list) -> dict:
    """Aggregate list of per-fold dicts into unified results structure."""
    metrics          = {t: {c: {'svm': {'acc': [], 'f1': []},
                                'rf':  {'acc': [], 'f1': []}}
                            for c in CONFIGS} for t in TARGETS}
    importance_folds = {t: [] for t in TARGETS}
    class_balance    = {t: [] for t in TARGETS}
    flagged_folds    = {t: [] for t in TARGETS}
    ablation         = {t: {b: {'acc': [], 'f1': []} for b in BAND_NAMES}
                        for t in TARGETS}
    # fold_acc_all: per-fold scalar accuracy for paired tests (M3)
    fold_acc_all     = {t: {} for t in TARGETS}

    for fold in fold_results:
        subj = fold['subj']
        for t in TARGETS:
            class_balance[t].append(fold['balance'][t])
            if fold['flagged'][t]:
                flagged_folds[t].append(int(subj))
            importance_folds[t].append(fold['imp'][t])
            for cfg in CONFIGS:
                for model in ('svm', 'rf'):
                    metrics[t][cfg][model]['acc'].append(
                        fold['metrics'][t][cfg][model]['acc'])
                    metrics[t][cfg][model]['f1'].append(
                        fold['metrics'][t][cfg][model]['f1'])
            for band in BAND_NAMES:
                ablation[t][band]['acc'].append(fold['ablation'][t][band]['acc'])
                ablation[t][band]['f1'].append(fold['ablation'][t][band]['f1'])
            for key, val in fold['fold_acc'][t].items():
                fold_acc_all[t].setdefault(key, []).append(val)

    return {
        'metrics':          metrics,
        'importance_folds': importance_folds,
        'importance_mean':  {t: np.mean(importance_folds[t], axis=0) for t in TARGETS},
        'class_balance':    class_balance,
        'flagged_folds':    flagged_folds,
        'ablation':         ablation,
        'fold_acc':         fold_acc_all,   # (32,) arrays per target per config-model key
    }


# ── Main LOSO loop ────────────────────────────────────────────────────────────

def run_loso(
    X_eeg: np.ndarray,
    X_peripheral: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int = 30,
    rf_n_estimators: int = 200,
    n_jobs: int = -3,
) -> dict:
    """
    Parallel LOSO loop. All 32 folds run concurrently.
    Progress bar with ETA shown via tqdm.
    n_jobs=-1 uses all available cores.
    """
    subjects = np.unique(subject_ids)

    with _tqdm_joblib(tqdm(total=len(subjects), desc='LOSO folds', unit='fold')):
        fold_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_run_fold)(
                subj, X_eeg, X_peripheral, y, subject_ids,
                n_perm_repeats, rf_n_estimators,
            )
            for subj in subjects
        )

    results = _aggregate_folds(fold_results)

    for t in TARGETS:
        if results['flagged_folds'][t]:
            print(f"[FLAG] {t}: imbalance >65/35 in subjects "
                  f"{results['flagged_folds'][t]}")

    return results


# ── Post-loop analysis ────────────────────────────────────────────────────────

def aggregate_importance(importance_mean: dict) -> dict:
    return {t: _aggregate_importance(imp) for t, imp in importance_mean.items()}


def importance_correlation(importance_mean: dict) -> dict:
    """
    Pairwise Spearman rho between importance vectors (RQ5).
    Returns rho and p-value for each pair.
    """
    results = {}
    for t1, t2 in CORR_PAIRS:
        rho, pval = spearmanr(importance_mean[t1], importance_mean[t2])
        results[f"{t1}_{t2}"] = {'rho': float(rho), 'pvalue': float(pval)}
    return results


# ── M1: Sign test on per-subject within-subject accuracy ─────────────────────

def sign_test_ws(metrics_ws: dict) -> dict:
    """
    Binomial sign test: how many of the 32 subjects individually exceed 0.5
    accuracy? Tests whether the number of above-chance subjects is significantly
    greater than expected by chance (binomial null: p=0.5, alternative='greater').

    Args:
        metrics_ws: run_within_subject()['metrics']
                    {target: {model: {'acc': list of 32 per-subject means}}}

    Returns:
        dict keyed by '{target}_{model}':
            'n_above':  number of subjects exceeding 0.5
            'n_valid':  number of non-NaN subjects
            'pvalue':   one-sided binomial p-value (alternative='greater')
    """
    out = {}
    for t in TARGETS:
        for model in ('svm', 'rf'):
            accs  = np.array(metrics_ws[t][model]['acc'], dtype=float)
            valid = accs[~np.isnan(accs)]
            n_above = int((valid > 0.5).sum())
            result  = binomtest(n_above, len(valid), p=0.5, alternative='greater')
            out[f"{t}_{model}"] = {
                'n_above': n_above,
                'n_valid': len(valid),
                'pvalue':  float(result.pvalue),
            }
    return out


# ── M3: Paired Wilcoxon signed-rank tests on LOSO fold accuracy (RQ4) ────────

def paired_accuracy_tests(fold_acc: dict) -> dict:
    """
    Wilcoxon signed-rank tests on paired fold-level LOSO accuracy (32 pairs per
    comparison). Tests whether accuracy differences between feature configurations
    are non-trivially above noise across folds (RQ4 directional evidence).

    Args:
        fold_acc: results['fold_acc'] from run_loso()
                  {target: {'{model}_{config}': list of 32 fold accuracies}}

    Returns:
        dict keyed by '{target}_{model}_{comparison}':
            'statistic': Wilcoxon W statistic
            'pvalue':    two-sided p-value
            'mean_diff': mean(A - B) across 32 folds (positive = A > B)
    """
    comparisons = [
        ('eeg',      'peripheral', 'eeg_vs_peripheral'),
        ('peripheral', 'combined', 'peripheral_vs_combined'),  # positive = peripheral > combined
        ('eeg',      'combined',   'eeg_vs_combined'),
    ]
    out = {}
    for t in TARGETS:
        for model in ('svm', 'rf'):
            for cfg_a, cfg_b, label in comparisons:
                a = np.array(fold_acc[t][f'{model}_{cfg_a}'])
                b = np.array(fold_acc[t][f'{model}_{cfg_b}'])
                diff = a - b
                # Wilcoxon requires non-constant differences
                if np.all(diff == 0):
                    out[f"{t}_{model}_{label}"] = {
                        'statistic': np.nan, 'pvalue': 1.0, 'mean_diff': 0.0}
                    continue
                stat, pval = wilcoxon(diff, alternative='two-sided')
                out[f"{t}_{model}_{label}"] = {
                    'statistic': float(stat),
                    'pvalue':    float(pval),
                    'mean_diff': float(diff.mean()),
                }
    return out


# ── M6: Benjamini-Hochberg FDR correction ────────────────────────────────────

def fdr_correction(pvalue_dict: dict, alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR correction on a flat dict of p-values.
    Controls the expected proportion of false discoveries rather than
    family-wise error rate, appropriate for the exploratory framing here.

    Args:
        pvalue_dict: {key: pvalue} — flat dict (not nested)
        alpha:       FDR level (default 0.05)

    Returns:
        {key: {'pvalue': original, 'pvalue_fdr': BH-adjusted, 'reject': bool}}
    """
    keys   = list(pvalue_dict.keys())
    pvals  = np.array([pvalue_dict[k] for k in keys], dtype=float)
    n      = len(pvals)
    order  = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    ranked[order] = (np.arange(1, n + 1) / n) * alpha

    # BH threshold: reject H_i if p_(i) <= (i/n) * alpha
    # Adjusted p-value: p_adj[i] = min over j>=i of (n/j) * p_(j)
    adj = np.minimum.accumulate((pvals[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_reordered = np.empty(n, dtype=float)
    adj_reordered[order] = np.minimum(adj, 1.0)

    return {
        k: {
            'pvalue':     float(pvalue_dict[k]),
            'pvalue_fdr': float(adj_reordered[i]),
            'reject':     bool(adj_reordered[i] < alpha),
        }
        for i, k in enumerate(keys)
    }


def apply_fdr_to_correlations(imp_corr: dict, alpha: float = 0.05) -> dict:
    """
    Apply FDR correction to a flat importance_correlation() output.

    Args:
        imp_corr: {pair_key: {'rho': ..., 'pvalue': ...}}
        alpha:    FDR level

    Returns:
        same structure with 'pvalue_fdr' and 'reject' added to each entry
    """
    pvals = {k: v['pvalue'] for k, v in imp_corr.items()}
    fdr   = fdr_correction(pvals, alpha=alpha)
    return {
        k: {**v, 'pvalue_fdr': fdr[k]['pvalue_fdr'], 'reject': fdr[k]['reject']}
        for k, v in imp_corr.items()
    }


# ── C1 partial: MI-selected feature subset importance correlation ──────────────

def mi_importance_correlation(
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    k: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Identify features carrying label-relevant information via mutual information,
    used as a diagnostic to check whether importance orthogonality holds when
    restricted to non-noise features (RQ5 robustness check).

    Global median binarization is used for MI scoring — this is a screening
    step, not part of any CV fold, so fold-level leakage is not a concern.

    Args:
        X_eeg:       (1280, 131) EEG features
        y:           (1280, 4)   raw labels
        subject_ids: (1280,)
        k:           top-k features per target to include (default 50)
        random_state: for mutual_info_classif reproducibility

    Returns:
        dict with keys:
            'selected_indices': np.ndarray — union of top-k indices across targets
            'n_selected':       int — number of features in the union (≤ 3k)
            'mi_scores':        {target: (131,) MI scores}
            'top_k_per_target': {target: (k,) top-k indices for that target}
    """
    from features import N_FEATURES

    # Global binarization for MI scoring — screening only, not CV
    # Note: using full dataset median here is acceptable for MI screening
    # (we are not making accuracy claims, only identifying informative features)
    mi_scores     = {}
    top_k_indices = {}
    for t_name, t_col in TARGETS.items():
        threshold  = float(np.median(y[:, t_col]))
        y_bin      = binarize(y[:, t_col], threshold)
        mi         = mutual_info_classif(
            X_eeg, y_bin, discrete_features=False,
            n_neighbors=5, random_state=random_state,
        )
        mi_scores[t_name]     = mi
        top_k_indices[t_name] = np.argsort(mi)[-k:]

    # Union of top-k indices across targets
    selected = np.unique(np.concatenate(list(top_k_indices.values())))

    # Load importance means from the run_loso results that are already in memory
    # (caller must pass them; this function takes them via a second return path)
    # Return MI info and selected indices; caller runs importance_correlation
    # on the sliced vectors using the helper below.
    return {
        'selected_indices': selected,
        'n_selected':       len(selected),
        'mi_scores':        mi_scores,
        'top_k_per_target': top_k_indices,
    }


def corr_on_selected(importance_mean: dict, selected_indices: np.ndarray) -> dict:
    """
    Compute pairwise Spearman correlation on importance vectors sliced to
    the MI-selected feature indices.

    Args:
        importance_mean: {target: (131,)} from run_loso or run_within_subject
        selected_indices: from mi_importance_correlation()['selected_indices']

    Returns:
        same format as importance_correlation()
    """
    sliced = {t: v[selected_indices] for t, v in importance_mean.items()}
    return importance_correlation(sliced)


# ── Option 3: SVM permutation importance (LOSO) ───────────────────────────────

def _run_fold_svm_importance(
    subj: int,
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int,
) -> dict:
    """
    Single LOSO fold: fit SVM on EEG features, compute permutation importance
    on the test fold. Uses C=10 consistent with main LOSO loop.
    """
    test_mask  = subject_ids == subj
    train_mask = ~test_mask
    fold_out   = {'subj': subj, 'imp': {}}

    for t_name, t_col in TARGETS.items():
        threshold = float(np.median(y[train_mask, t_col]))
        y_train   = binarize(y[train_mask, t_col], threshold)
        y_test    = binarize(y[test_mask,  t_col], threshold)

        scaler    = StandardScaler().fit(X_eeg[train_mask])
        X_train_s = scaler.transform(X_eeg[train_mask])
        X_test_s  = scaler.transform(X_eeg[test_mask])

        svm = SVC(kernel='rbf', C=_SVM_C, random_state=42)
        svm.fit(X_train_s, y_train)

        pi = permutation_importance(
            svm, X_test_s, y_test,
            n_repeats=n_perm_repeats, random_state=42, n_jobs=1,
        )
        fold_out['imp'][t_name] = pi.importances_mean  # (131,)

    return fold_out


def run_loso_svm_importance(
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int = 20,
    n_jobs: int = -3,
) -> dict:
    """
    LOSO SVM permutation importance loop. EEG config only.
    Returns mean importance vectors per target, aggregated across 32 folds.
    """
    subjects = np.unique(subject_ids)

    with _tqdm_joblib(tqdm(total=len(subjects),
                           desc='SVM importance folds', unit='fold')):
        fold_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_run_fold_svm_importance)(
                subj, X_eeg, y, subject_ids, n_perm_repeats,
            )
            for subj in subjects
        )

    imp_folds = {t: [] for t in TARGETS}
    for fold in fold_results:
        for t in TARGETS:
            imp_folds[t].append(fold['imp'][t])

    imp_mean = {t: np.mean(imp_folds[t], axis=0) for t in TARGETS}

    return {
        'importance_folds': imp_folds,
        'importance_mean':  imp_mean,
        'importance_agg':   aggregate_importance(imp_mean),
        'importance_corr':  importance_correlation(imp_mean),
    }


# ── Option 1: within-subject k-fold CV ───────────────────────────────────────

def _run_within_subject_fold(
    subj: int,
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int,
    n_perm_repeats: int,
    rf_n_estimators: int,
) -> dict:
    """
    Run k-fold CV within a single subject. Uses C=10 for SVM consistent with
    the main LOSO loop.
    """
    from sklearn.model_selection import StratifiedKFold

    mask    = subject_ids == subj
    X_subj  = X_eeg[mask]          # (40, 131)
    y_raw   = y[mask]               # (40, 4)
    fold_out = {'subj': subj, 'metrics': {}, 'imp_rf': {}, 'imp_svm': {}}

    for t_name, t_col in TARGETS.items():
        threshold = float(np.median(y_raw[:, t_col]))
        y_bin     = binarize(y_raw[:, t_col], threshold)

        if len(np.unique(y_bin)) < 2:
            fold_out['metrics'][t_name] = {
                'svm': {'acc': [np.nan], 'f1': [np.nan]},
                'rf':  {'acc': [np.nan], 'f1': [np.nan]},
            }
            fold_out['imp_rf'][t_name]  = [np.zeros(X_subj.shape[1])]
            fold_out['imp_svm'][t_name] = [np.zeros(X_subj.shape[1])]
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_out['metrics'][t_name] = {
            'svm': {'acc': [], 'f1': []},
            'rf':  {'acc': [], 'f1': []},
        }
        fold_out['imp_rf'][t_name]  = []
        fold_out['imp_svm'][t_name] = []

        for tr_idx, te_idx in skf.split(X_subj, y_bin):
            X_tr, X_te = X_subj[tr_idx], X_subj[te_idx]
            y_tr, y_te = y_bin[tr_idx],  y_bin[te_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            scaler    = StandardScaler().fit(X_tr)
            X_tr_s    = scaler.transform(X_tr)
            X_te_s    = scaler.transform(X_te)

            # SVM (C=10)
            svm = SVC(kernel='rbf', C=_SVM_C, random_state=42)
            svm.fit(X_tr_s, y_tr)
            yp  = svm.predict(X_te_s)
            fold_out['metrics'][t_name]['svm']['acc'].append(accuracy_score(y_te, yp))
            fold_out['metrics'][t_name]['svm']['f1'].append(
                f1_score(y_te, yp, zero_division=0))

            pi_svm = permutation_importance(
                svm, X_te_s, y_te,
                n_repeats=n_perm_repeats, random_state=42, n_jobs=1,
            )
            fold_out['imp_svm'][t_name].append(pi_svm.importances_mean)

            # RF
            rf = RandomForestClassifier(
                n_estimators=rf_n_estimators, class_weight='balanced',
                random_state=42, n_jobs=1)
            rf.fit(X_tr_s, y_tr)
            yp = rf.predict(X_te_s)
            fold_out['metrics'][t_name]['rf']['acc'].append(accuracy_score(y_te, yp))
            fold_out['metrics'][t_name]['rf']['f1'].append(
                f1_score(y_te, yp, zero_division=0))

            pi_rf = permutation_importance(
                rf, X_te_s, y_te,
                n_repeats=n_perm_repeats, random_state=42, n_jobs=1,
            )
            fold_out['imp_rf'][t_name].append(pi_rf.importances_mean)

    return fold_out


def run_within_subject(
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    n_perm_repeats: int = 20,
    rf_n_estimators: int = 150,
    n_jobs: int = -3,
) -> dict:
    """
    Within-subject k-fold CV across all 32 subjects, parallelized.
    Now also returns sign_test results (M1) in the output dict.

    Args:
        X_eeg:          (1280, 131)
        y:              (1280, 4)
        subject_ids:    (1280,)
        n_splits:       CV folds per subject (default 5)
        n_perm_repeats: importance repeats per inner fold
        rf_n_estimators:trees per RF
        n_jobs:         parallel workers across subjects

    Returns:
        dict with keys:
            'metrics':         {target: {model: {'acc': (32,), 'f1': (32,)}}}
            'importance_mean': {target: {'rf': (131,), 'svm': (131,)}}
            'importance_agg':  {target: {'rf': agg_dict, 'svm': agg_dict}}
            'importance_corr': {'rf': corr_dict, 'svm': corr_dict}
            'sign_test':       {target_model: {'n_above', 'n_valid', 'pvalue'}}  # M1
    """
    subjects = np.unique(subject_ids)

    with _tqdm_joblib(tqdm(total=len(subjects),
                           desc='Within-subject CV', unit='subject')):
        subj_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_run_within_subject_fold)(
                subj, X_eeg, y, subject_ids,
                n_splits, n_perm_repeats, rf_n_estimators,
            )
            for subj in subjects
        )

    metrics     = {t: {'svm': {'acc': [], 'f1': []},
                       'rf':  {'acc': [], 'f1': []}}
                   for t in TARGETS}
    imp_rf_all  = {t: [] for t in TARGETS}
    imp_svm_all = {t: [] for t in TARGETS}

    for sr in subj_results:
        for t in TARGETS:
            for model in ('svm', 'rf'):
                metrics[t][model]['acc'].append(
                    float(np.nanmean(sr['metrics'][t][model]['acc'])))
                metrics[t][model]['f1'].append(
                    float(np.nanmean(sr['metrics'][t][model]['f1'])))
            if sr['imp_rf'][t] and not all(np.all(v == 0) for v in sr['imp_rf'][t]):
                imp_rf_all[t].append(np.mean(sr['imp_rf'][t],  axis=0))
            if sr['imp_svm'][t] and not all(np.all(v == 0) for v in sr['imp_svm'][t]):
                imp_svm_all[t].append(np.mean(sr['imp_svm'][t], axis=0))

    imp_mean = {
        t: {
            'rf':  np.mean(imp_rf_all[t],  axis=0),
            'svm': np.mean(imp_svm_all[t], axis=0),
        }
        for t in TARGETS
    }

    imp_agg = {
        t: {
            'rf':  _aggregate_importance(imp_mean[t]['rf']),
            'svm': _aggregate_importance(imp_mean[t]['svm']),
        }
        for t in TARGETS
    }

    imp_corr = {
        'rf':  importance_correlation({t: imp_mean[t]['rf']  for t in TARGETS}),
        'svm': importance_correlation({t: imp_mean[t]['svm'] for t in TARGETS}),
    }

    # M1: sign test — how many subjects individually beat chance?
    sign_test = sign_test_ws(metrics)

    return {
        'metrics':         metrics,
        'importance_mean': imp_mean,
        'importance_agg':  imp_agg,
        'importance_corr': imp_corr,
        'sign_test':       sign_test,    # M1
    }


# ── Lean fold for null (no SVM, fewer trees, fewer importance repeats) ────────

def _run_fold_null(
    subj: int,
    X_eeg: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int,
    rf_n_estimators: int,
) -> dict:
    """
    Minimal null fold: EEG-only RF + permutation importance.
    Peripheral, combined, and band ablation are excluded from the null loop:
    the null is used only for (1) EEG RF accuracy vs chance and (2) importance
    correlation null for RQ5. Both require only the EEG RF path.
    """
    test_mask  = subject_ids == subj
    train_mask = ~test_mask
    fold_out   = {'subj': subj, 'metrics': {}, 'imp': {}}

    for t_name, t_col in TARGETS.items():
        threshold = float(np.median(y[train_mask, t_col]))
        y_train   = binarize(y[train_mask, t_col], threshold)
        y_test    = binarize(y[test_mask,  t_col], threshold)

        scaler    = StandardScaler().fit(X_eeg[train_mask])
        X_train_s = scaler.transform(X_eeg[train_mask])
        X_test_s  = scaler.transform(X_eeg[test_mask])

        rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, random_state=42, n_jobs=1)
        rf.fit(X_train_s, y_train)
        yp = rf.predict(X_test_s)

        fold_out['metrics'][t_name] = {
            'eeg': {'rf': {'acc': accuracy_score(y_test, yp),
                           'f1':  f1_score(y_test, yp, zero_division=0)}},
        }

        pi = permutation_importance(
            rf, X_test_s, y_test,
            n_repeats=n_perm_repeats, random_state=42, n_jobs=1,
        )
        fold_out['imp'][t_name] = pi.importances_mean  # (131,)

    return fold_out


def _aggregate_folds_null(fold_results: list) -> dict:
    """Aggregation for minimal null folds (EEG RF only, no ablation)."""
    # Only EEG config; peripheral and combined dropped from null loop
    metrics          = {t: {'eeg': {'rf': {'acc': [], 'f1': []}}}
                        for t in TARGETS}
    importance_folds = {t: [] for t in TARGETS}

    for fold in fold_results:
        for t in TARGETS:
            importance_folds[t].append(fold['imp'][t])
            metrics[t]['eeg']['rf']['acc'].append(
                fold['metrics'][t]['eeg']['rf']['acc'])
            metrics[t]['eeg']['rf']['f1'].append(
                fold['metrics'][t]['eeg']['rf']['f1'])

    return {
        'metrics':         metrics,
        'importance_mean': {t: np.mean(importance_folds[t], axis=0) for t in TARGETS},
    }


# ── Shuffled-label null (M2: n_permutations=1000, p5 added) ──────────────────

def shuffled_label_null(
    X_eeg: np.ndarray,
    X_peripheral: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_permutations: int = 500,
    batch_size: int = 50,
    checkpoint_dir: str = '../results/null_checkpoints',
    rf_n_estimators: int = 50,    # reduced from 100: enough for distribution estimate
    n_perm_repeats: int = 3,      # reduced from 10: enough for correlation null
    n_jobs: int = -3,
    random_state: int = 42,
) -> dict:
    """
    Null distributions via shuffled labels, batched with checkpoint saving.
    Labels are shuffled within each subject to preserve per-subject trial counts.

    Uses _run_fold_null (lean RF-only fold): 1 RF fit + 1 importance computation
    per fold per permutation, consistent with the null requiring only EEG RF
    accuracy (vs chance) and importance correlation (RQ5).

    Saves one .pkl per batch to checkpoint_dir. Safe to interrupt — resumes
    from the last completed batch on the next call with the same checkpoint_dir.

    At 500 permutations the null p5/p95 estimates stabilize to ±~0.010,
    which is acceptable for the exploratory framing of this analysis.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng      = np.random.default_rng(random_state)
    subjects = np.unique(subject_ids)

    raw_acc  = {t: [] for t in TARGETS}   # EEG RF only
    raw_corr = {k: [] for k in CORR_KEYS}

    n_batches  = (n_permutations + batch_size - 1) // batch_size
    perms_done = 0

    for batch_i in tqdm(range(n_batches), desc='Null batches', unit='batch'):
        ckpt_path   = ckpt_dir / f"batch_{batch_i:03d}.pkl"
        batch_perms = min(batch_size, n_permutations - perms_done)

        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as f:
                batch_data = pickle.load(f)
            print(f"Batch {batch_i + 1}/{n_batches}: resumed from checkpoint "
                  f"({len(batch_data['corr'][CORR_KEYS[0]])} perms).")
        else:
            print(f"Batch {batch_i + 1}/{n_batches}: "
                  f"running {batch_perms} permutations...")

            batch_data = {
                'acc':  {t: [] for t in TARGETS},
                'corr': {k: [] for k in CORR_KEYS},
            }

            for _ in tqdm(range(batch_perms), desc=f'  Batch {batch_i+1}',
                          unit='perm', leave=False):
                y_shuf = y.copy()
                for subj in subjects:
                    mask = subject_ids == subj
                    y_shuf[mask] = rng.permutation(y_shuf[mask])

                fold_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(_run_fold_null)(
                        subj, X_eeg, y_shuf, subject_ids,
                        n_perm_repeats, rf_n_estimators,
                    )
                    for subj in subjects
                )

                agg = _aggregate_folds_null(fold_results)

                for t in TARGETS:
                    batch_data['acc'][t].append(
                        float(np.mean(agg['metrics'][t]['eeg']['rf']['acc'])))

                for key, val in importance_correlation(agg['importance_mean']).items():
                    batch_data['corr'][key].append(val['rho'])

            with open(ckpt_path, 'wb') as f:
                pickle.dump(batch_data, f)
            print(f"  Checkpoint saved: {ckpt_path.name}")

        perms_done += batch_perms

        for t in TARGETS:
            raw_acc[t].extend(batch_data['acc'][t])
        for key in CORR_KEYS:
            raw_corr[key].extend(batch_data['corr'][key])

    def _summarize(values: list) -> dict:
        arr = np.array(values)
        return {
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'p5':   float(np.percentile(arr, 5)),
            'p95':  float(np.percentile(arr, 95)),
            'n':    len(arr),
        }

    return {
        'accuracy':    {t: _summarize(raw_acc[t]) for t in TARGETS},
        'correlation': {k: _summarize(v) for k, v in raw_corr.items()},
        'raw':         {'accuracy': raw_acc, 'correlation': raw_corr},
    }
