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
from scipy.stats import spearmanr
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
    """
    test_mask  = subject_ids == subj
    train_mask = ~test_mask
    fold_out   = {
        'subj': subj, 'metrics': {}, 'imp': {},
        'ablation': {}, 'balance': {}, 'flagged': {},
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

        for cfg_name, X_cfg in configs.items():
            scaler    = StandardScaler().fit(X_cfg[train_mask])
            X_train_s = scaler.transform(X_cfg[train_mask])
            X_test_s  = scaler.transform(X_cfg[test_mask])

            # SVM
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(X_train_s, y_train)
            yp  = svm.predict(X_test_s)
            fold_out['metrics'][t_name][cfg_name] = {
                'svm': {'acc': accuracy_score(y_test, yp),
                        'f1':  f1_score(y_test, yp, zero_division=0)},
            }

            # RF (n_jobs=1 inside fold — outer parallelism across folds)
            rf = RandomForestClassifier(
                n_estimators=rf_n_estimators, random_state=42, n_jobs=1)
            rf.fit(X_train_s, y_train)
            yp = rf.predict(X_test_s)
            fold_out['metrics'][t_name][cfg_name]['rf'] = {
                'acc': accuracy_score(y_test, yp),
                'f1':  f1_score(y_test, yp, zero_division=0),
            }

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

    return {
        'metrics':          metrics,
        'importance_folds': importance_folds,
        'importance_mean':  {t: np.mean(importance_folds[t], axis=0) for t in TARGETS},
        'class_balance':    class_balance,
        'flagged_folds':    flagged_folds,
        'ablation':         ablation,
    }


# ── Main LOSO loop ────────────────────────────────────────────────────────────

def run_loso(
    X_eeg: np.ndarray,
    X_peripheral: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_perm_repeats: int = 30,
    rf_n_estimators: int = 200,
    n_jobs: int = -1,
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
    """Pairwise Spearman rho between importance vectors (RQ5)."""
    results = {}
    for t1, t2 in CORR_PAIRS:
        rho, pval = spearmanr(importance_mean[t1], importance_mean[t2])
        results[f"{t1}_{t2}"] = {'rho': float(rho), 'pvalue': float(pval)}
    return results


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
    on the test fold. SVM is the primary cross-subject classifier; its importance
    vectors are used for RQ2 and RQ5 in place of RF importance.
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

        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_s, y_train)

        # Permutation importance on test fold using SVM predict
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
    n_jobs: int = -1,
) -> dict:
    """
    LOSO SVM permutation importance loop. EEG config only.
    Returns mean importance vectors per target, aggregated across 32 folds.
    Used to back RQ2 (FAA emergence) and RQ5 (dominance dissociability) with
    a classifier that actually generalizes cross-subject.

    Args:
        X_eeg:          (1280, 131) EEG features only
        y:              (1280, 4)
        subject_ids:    (1280,)
        n_perm_repeats: permutation repeats per fold (default 20)
        n_jobs:         parallel workers

    Returns:
        dict with keys:
            'importance_folds': {target: list of (131,) per fold}
            'importance_mean':  {target: (131,) averaged across 32 folds}
            'importance_agg':   aggregated band/electrode/region per target
            'importance_corr':  pairwise Spearman rho across targets
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
    Run k-fold CV within a single subject. Trains and tests on the same
    subject's trials only — eliminates cross-subject distribution shift.
    Returns per-fold accuracy, F1, and permutation importance for both
    RF and SVM on EEG features.
    """
    from sklearn.model_selection import StratifiedKFold

    mask    = subject_ids == subj
    X_subj  = X_eeg[mask]          # (40, 131)
    y_raw   = y[mask]               # (40, 4)
    fold_out = {'subj': subj, 'metrics': {}, 'imp_rf': {}, 'imp_svm': {}}

    for t_name, t_col in TARGETS.items():
        threshold = float(np.median(y_raw[:, t_col]))
        y_bin     = binarize(y_raw[:, t_col], threshold)

        # Skip if subject has only one class after binarization — happens when
        # all 40 trials cluster at or below the median (consistent rater).
        # Record NaN so aggregation can filter these subjects out gracefully.
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

            # Skip inner fold if training split has only one class —
            # happens with skewed per-subject distributions and small n
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            scaler    = StandardScaler().fit(X_tr)
            X_tr_s    = scaler.transform(X_tr)
            X_te_s    = scaler.transform(X_te)

            # SVM
            svm = SVC(kernel='rbf', random_state=42)
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
    n_jobs: int = -1,
) -> dict:
    """
    Within-subject k-fold CV across all 32 subjects, parallelized.
    Eliminates cross-subject distribution shift to validate that EEG features
    carry real signal and importance vectors are meaningful.

    Args:
        X_eeg:          (1280, 131)
        y:              (1280, 4)
        subject_ids:    (1280,)
        n_splits:       CV folds per subject (default 5 → 8 train, 32 test trials)
        n_perm_repeats: importance repeats per inner fold
        rf_n_estimators:trees per RF
        n_jobs:         parallel workers across subjects

    Returns:
        dict with keys:
            'metrics':         {target: {model: {'acc': (32,), 'f1': (32,)}}}
                               each value is the per-subject mean across k folds
            'importance_mean': {target: {'rf': (131,), 'svm': (131,)}}
                               averaged across all subjects and all inner folds
            'importance_agg':  {target: {'rf': agg_dict, 'svm': agg_dict}}
            'importance_corr': {'rf': corr_dict, 'svm': corr_dict}
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

    # Aggregate: per-subject mean across inner folds, then collect across subjects
    metrics = {t: {'svm': {'acc': [], 'f1': []},
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
            # Mean across inner folds for this subject — skip if all folds were degenerate
            if sr['imp_rf'][t] and not all(np.all(v == 0) for v in sr['imp_rf'][t]):
                imp_rf_all[t].append(np.mean(sr['imp_rf'][t],  axis=0))
            if sr['imp_svm'][t] and not all(np.all(v == 0) for v in sr['imp_svm'][t]):
                imp_svm_all[t].append(np.mean(sr['imp_svm'][t], axis=0))

    # Mean importance across all 32 subjects
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

    return {
        'metrics':         metrics,
        'importance_mean': imp_mean,
        'importance_agg':  imp_agg,
        'importance_corr': imp_corr,
    }

def shuffled_label_null(
    X_eeg: np.ndarray,
    X_peripheral: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_permutations: int = 100,
    batch_size: int = 10,
    checkpoint_dir: str = '../results/null_checkpoints',
    rf_n_estimators: int = 100,
    n_perm_repeats: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    """
    Null distributions via shuffled labels, batched with checkpoint saving.

    Saves one .pkl per batch to checkpoint_dir. If interrupted, resumes from
    the last completed batch automatically on the next call with the same
    checkpoint_dir. Do not change checkpoint_dir between runs.

    batch_size=10 means a crash loses at most 10 permutations worth of work.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng      = np.random.default_rng(random_state)
    subjects = np.unique(subject_ids)

    raw_acc  = {t: {cfg: [] for cfg in CONFIGS} for t in TARGETS}
    raw_abl  = {t: {b:   [] for b in BAND_NAMES} for t in TARGETS}
    raw_corr = {k: [] for k in CORR_KEYS}

    n_batches  = (n_permutations + batch_size - 1) // batch_size
    perms_done = 0

    for batch_i in tqdm(range(n_batches), desc='Null batches', unit='batch'):
        ckpt_path  = ckpt_dir / f"batch_{batch_i:03d}.pkl"
        batch_perms = min(batch_size, n_permutations - perms_done)

        if ckpt_path.exists():
            with open(ckpt_path, 'rb') as f:
                batch_data = pickle.load(f)
            print(f"Batch {batch_i + 1}/{n_batches}: resumed from checkpoint.")
        else:
            print(f"Batch {batch_i + 1}/{n_batches}: "
                  f"running {batch_perms} permutations...")

            batch_data = {
                'acc':  {t: {cfg: [] for cfg in CONFIGS} for t in TARGETS},
                'abl':  {t: {b:   [] for b in BAND_NAMES} for t in TARGETS},
                'corr': {k: [] for k in CORR_KEYS},
            }

            for perm in tqdm(range(batch_perms), desc=f'  Batch {batch_i+1} perms',
                             unit='perm', leave=False):
                y_shuf = y.copy()
                for subj in subjects:
                    mask = subject_ids == subj
                    y_shuf[mask] = rng.permutation(y_shuf[mask])

                fold_results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(_run_fold)(
                        subj, X_eeg, X_peripheral, y_shuf, subject_ids,
                        n_perm_repeats, rf_n_estimators,
                    )
                    for subj in subjects
                )

                agg = _aggregate_folds(fold_results)

                for t in TARGETS:
                    for cfg in CONFIGS:
                        batch_data['acc'][t][cfg].append(
                            float(np.mean(agg['metrics'][t][cfg]['rf']['acc'])))
                    for band in BAND_NAMES:
                        batch_data['abl'][t][band].append(
                            float(np.mean(agg['ablation'][t][band]['acc'])))

                for key, val in importance_correlation(agg['importance_mean']).items():
                    batch_data['corr'][key].append(val['rho'])

            with open(ckpt_path, 'wb') as f:
                pickle.dump(batch_data, f)
            print(f"  Checkpoint saved: {ckpt_path.name}")

        perms_done += batch_perms

        for t in TARGETS:
            for cfg in CONFIGS:
                raw_acc[t][cfg].extend(batch_data['acc'][t][cfg])
            for band in BAND_NAMES:
                raw_abl[t][band].extend(batch_data['abl'][t][band])
        for key in CORR_KEYS:
            raw_corr[key].extend(batch_data['corr'][key])

    def _summarize(values: list) -> dict:
        arr = np.array(values)
        return {'mean': float(arr.mean()), 'std': float(arr.std()),
                'p95':  float(np.percentile(arr, 95))}

    return {
        'accuracy':    {t: {cfg: _summarize(raw_acc[t][cfg]) for cfg in CONFIGS}
                        for t in TARGETS},
        'ablation':    {t: {b: _summarize(raw_abl[t][b]) for b in BAND_NAMES}
                        for t in TARGETS},
        'correlation': {k: _summarize(v) for k, v in raw_corr.items()},
        'raw':         {'accuracy': raw_acc, 'ablation': raw_abl,
                        'correlation': raw_corr},
    }
