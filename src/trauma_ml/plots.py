"""Diagnostic plots for every trained model.

Generated per model
-------------------
* ROC curve  (test + holdout on same axes)
* PR  curve  (test + holdout on same axes)
* Confusion matrix  (separate figure per partition: test / holdout)
* SHAP summary (beeswarm) and bar chart

All figures are saved to ``<output_dir>/plots/<model_id>/``.

Usage (called from Trainer._evaluate after fitting)
----------------------------------------------------
    from trauma_ml.plots import save_model_plots
    save_model_plots(
        model=self.model,
        X_test=X_te, y_test=y_te,
        X_holdout=X_ho, y_holdout=y_ho,   # may be None
        feature_names=self.predictors,
        output_dir=outputs_root / "plots" / self.cfg.model_id,
        model_id=self.cfg.model_id,
    )
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _proba(model, X: pd.DataFrame) -> np.ndarray | None:
    """Return positive-class probability scores, or None."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            s = model.decision_function(X)
            return (s - s.min()) / (s.max() - s.min() + 1e-12)
        except Exception:
            pass
    return None


def _roc(y_true, y_score):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc


def _pr(y_true, y_score):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return rec, prec, ap


def _auc_ci(y_true, y_score, kind: str, n_boot: int, alpha: float = 0.05,
            seed: int = 42):
    """Stratified percentile bootstrap CI for AUROC ('roc') or AUPRC ('pr').

    Returns (lo, hi) or None if n_boot<=0 or the data is degenerate.  Same
    method as evaluation._bootstrap_cis (resample within each class), kept
    here so the plot legends can annotate every curve — model AND baselines —
    with its interval.
    """
    if not n_boot or n_boot <= 0:
        return None
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return None
    metric = roc_auc_score if kind == "roc" else average_precision_score
    rng = np.random.default_rng(seed)
    strata = [np.where(y_true == c)[0] for c in np.unique(y_true)]
    vals = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(float(metric(yt, y_score[idx])))
        except ValueError:
            continue
    if len(vals) < max(20, n_boot // 5):
        return None
    arr = np.asarray(vals)
    return (float(np.percentile(arr, 100 * alpha / 2)),
            float(np.percentile(arr, 100 * (1 - alpha / 2))))


def _ci_suffix(ci):
    """Format a CI tuple as ' [lo-hi]' for legend labels (empty if None)."""
    return "" if ci is None else f" [{ci[0]:.3f}-{ci[1]:.3f}]"


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------
def _plot_roc_pr_one_panel(
    score_test: np.ndarray,
    y_test: np.ndarray,
    score_holdout: np.ndarray | None,
    y_holdout: np.ndarray | None,
    output_path: Path,
    title_prefix: str,
    test_n: int,
    holdout_n: int | None,
    baseline_scores_test: dict[str, np.ndarray] | None,
    baseline_scores_holdout: dict[str, np.ndarray] | None,
    test_mask: np.ndarray | None = None,
    holdout_mask: np.ndarray | None = None,
    n_bootstrap: int = 0,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> None:
    """Internal: render one ROC + PR figure to disk.

    Parameters
    ----------
    score_test / score_holdout :
        Model probability scores already filtered through any subset masks.
    y_test / y_holdout :
        True labels already filtered through any subset masks.
    test_n / holdout_n :
        Sample size shown in the model's legend entry (the size of the
        subset the model curve is computed on, AFTER any subset filtering).
    baseline_scores_test / baseline_scores_holdout :
        Already-aligned-with-y baseline score arrays.  Each is plotted on
        its own non-NaN subset.
    """
    import matplotlib.pyplot as plt

    BASELINE_COLOURS = {
        "ISS":   ("#2ca02c", "#98df8a"),
        "NISS":  ("#d62728", "#ff9896"),
        "TRISS": ("#9467bd", "#c5b0d5"),
        "RTS":   ("#8c564b", "#c49c94"),
        "MGAP":  ("#e377c2", "#f7b6d2"),
        "MREMS": ("#17becf", "#9edae5"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold")

    # === ROC ===
    ax = axes[0]
    fpr, tpr, auc = _roc(y_test, score_test)
    ax.plot(fpr, tpr, lw=2.5,
            label=f"Model (test AUC={auc:.3f}{_ci_suffix(_auc_ci(y_test, score_test, 'roc', n_bootstrap, ci_alpha, bootstrap_seed))}, n={test_n})",
            color="#1f77b4", zorder=10)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    if (score_holdout is not None and y_holdout is not None
            and len(np.unique(y_holdout)) > 1):
        fpr_h, tpr_h, auc_h = _roc(y_holdout, score_holdout)
        ax.plot(fpr_h, tpr_h, lw=2.5, linestyle="--",
                label=f"Model (holdout AUC={auc_h:.3f}{_ci_suffix(_auc_ci(y_holdout, score_holdout, 'roc', n_bootstrap, ci_alpha, bootstrap_seed))}, n={holdout_n})",
                color="#ff7f0e", zorder=9)

    if baseline_scores_test:
        for name, scores in baseline_scores_test.items():
            scores = np.asarray(scores)
            valid = ~np.isnan(scores)
            if valid.sum() < 30 or len(np.unique(y_test[valid])) < 2:
                continue
            try:
                fpr_b, tpr_b, auc_b = _roc(y_test[valid], scores[valid])
                col = BASELINE_COLOURS.get(name.upper(), ("#7f7f7f", "#c7c7c7"))[0]
                ax.plot(fpr_b, tpr_b, lw=1.5, color=col, alpha=0.85,
                        label=f"{name} (test AUC={auc_b:.3f}{_ci_suffix(_auc_ci(y_test[valid], scores[valid], 'roc', n_bootstrap, ci_alpha, bootstrap_seed))}, n={int(valid.sum())})")
            except Exception:
                pass

    if (baseline_scores_holdout and score_holdout is not None
            and y_holdout is not None
            and len(np.unique(y_holdout)) > 1):
        for name, scores in baseline_scores_holdout.items():
            scores = np.asarray(scores)
            valid = ~np.isnan(scores)
            if valid.sum() < 30 or len(np.unique(y_holdout[valid])) < 2:
                continue
            try:
                fpr_b, tpr_b, auc_b = _roc(y_holdout[valid], scores[valid])
                col = BASELINE_COLOURS.get(name.upper(), ("#7f7f7f", "#c7c7c7"))[1]
                ax.plot(fpr_b, tpr_b, lw=1.5, linestyle="--",
                        color=col, alpha=0.85,
                        label=f"{name} (holdout AUC={auc_b:.3f}{_ci_suffix(_auc_ci(y_holdout[valid], scores[valid], 'roc', n_bootstrap, ci_alpha, bootstrap_seed))}, n={int(valid.sum())})")
            except Exception:
                pass

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    # === PR ===
    ax = axes[1]
    rec, prec, ap = _pr(y_test, score_test)
    ax.plot(rec, prec, lw=2.5,
            label=f"Model (test AP={ap:.3f}{_ci_suffix(_auc_ci(y_test, score_test, 'pr', n_bootstrap, ci_alpha, bootstrap_seed))}, n={test_n})",
            color="#1f77b4", zorder=10)
    baseline_prev = float(np.mean(y_test))
    ax.axhline(baseline_prev, color="k", linestyle="--", lw=0.8, alpha=0.5,
               label=f"No-skill (prev={baseline_prev:.3f})")

    if (score_holdout is not None and y_holdout is not None
            and len(np.unique(y_holdout)) > 1):
        rec_h, prec_h, ap_h = _pr(y_holdout, score_holdout)
        ax.plot(rec_h, prec_h, lw=2.5, linestyle="--",
                label=f"Model (holdout AP={ap_h:.3f}{_ci_suffix(_auc_ci(y_holdout, score_holdout, 'pr', n_bootstrap, ci_alpha, bootstrap_seed))}, n={holdout_n})",
                color="#ff7f0e", zorder=9)

    if baseline_scores_test:
        for name, scores in baseline_scores_test.items():
            scores = np.asarray(scores)
            valid = ~np.isnan(scores)
            if valid.sum() < 30 or len(np.unique(y_test[valid])) < 2:
                continue
            try:
                rec_b, prec_b, ap_b = _pr(y_test[valid], scores[valid])
                col = BASELINE_COLOURS.get(name.upper(), ("#7f7f7f", "#c7c7c7"))[0]
                ax.plot(rec_b, prec_b, lw=1.5, color=col, alpha=0.85,
                        label=f"{name} (test AP={ap_b:.3f}{_ci_suffix(_auc_ci(y_test[valid], scores[valid], 'pr', n_bootstrap, ci_alpha, bootstrap_seed))}, n={int(valid.sum())})")
            except Exception:
                pass

    if (baseline_scores_holdout and score_holdout is not None
            and y_holdout is not None
            and len(np.unique(y_holdout)) > 1):
        for name, scores in baseline_scores_holdout.items():
            scores = np.asarray(scores)
            valid = ~np.isnan(scores)
            if valid.sum() < 30 or len(np.unique(y_holdout[valid])) < 2:
                continue
            try:
                rec_b, prec_b, ap_b = _pr(y_holdout[valid], scores[valid])
                col = BASELINE_COLOURS.get(name.upper(), ("#7f7f7f", "#c7c7c7"))[1]
                ax.plot(rec_b, prec_b, lw=1.5, linestyle="--",
                        color=col, alpha=0.85,
                        label=f"{name} (holdout AP={ap_b:.3f}{_ci_suffix(_auc_ci(y_holdout[valid], scores[valid], 'pr', n_bootstrap, ci_alpha, bootstrap_seed))}, n={int(valid.sum())})")
            except Exception:
                pass

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_holdout: pd.DataFrame | None,
    y_holdout: np.ndarray | None,
    output_dir: Path,
    model_id: str,
    baseline_scores_test: dict[str, np.ndarray] | None = None,
    baseline_scores_holdout: dict[str, np.ndarray] | None = None,
    n_bootstrap: int = 0,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> None:
    """ROC and PR curves — produces TWO files when baselines are supplied:

    * ``roc_pr_curves__full.png`` — model on the full test/holdout sets;
      each baseline plotted on its OWN non-NaN subset (sizes differ across
      baselines and from the model).  The model line is the truthful
      operating-curve evaluated on every patient predicted at runtime;
      baseline lines convey "what would the score have done on patients
      where it was computable?".

    * ``roc_pr_curves__baseline_complete.png`` — restricted to rows where
      ALL listed baselines are computable simultaneously.  Model AND every
      baseline are evaluated on the same denominator → fully comparable
      AUCs / APs.  This is the apples-to-apples view.

    Round 30: legend always shows ``n=`` for both model and baseline lines.

    Multiclass tasks: skipped silently (binary-only metrics).
    """
    if len(np.unique(y_test)) > 2:
        log.info("[%s] Skipping ROC/PR plot for multiclass task", model_id)
        return
    score_test = _proba(model, X_test)
    if score_test is None or len(np.unique(y_test)) < 2:
        log.warning("[%s] Cannot plot ROC/PR: no proba or single class", model_id)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_holdout = None
    if X_holdout is not None and y_holdout is not None and len(y_holdout) > 0:
        score_holdout = _proba(model, X_holdout)

    # ── Plot 1: full sets, baselines on their own non-NaN subsets ────
    _plot_roc_pr_one_panel(
        score_test=score_test,
        y_test=y_test,
        score_holdout=score_holdout,
        y_holdout=y_holdout,
        output_path=output_dir / "roc_pr_curves__full.png",
        title_prefix=f"Model: {model_id} — full test/holdout sets",
        test_n=int(len(y_test)),
        holdout_n=int(len(y_holdout)) if y_holdout is not None else None,
        baseline_scores_test=baseline_scores_test,
        baseline_scores_holdout=baseline_scores_holdout,
        n_bootstrap=n_bootstrap, ci_alpha=ci_alpha, bootstrap_seed=bootstrap_seed,
    )

    # ── Plot 2: baseline-complete subset (all baselines computable) ──
    if baseline_scores_test:
        # Mask = rows where every supplied baseline is non-NaN
        test_mask = np.ones(len(y_test), dtype=bool)
        for name, scores in baseline_scores_test.items():
            test_mask &= ~np.isnan(np.asarray(scores))

        holdout_mask = None
        if (baseline_scores_holdout and score_holdout is not None
                and y_holdout is not None):
            holdout_mask = np.ones(len(y_holdout), dtype=bool)
            for name, scores in baseline_scores_holdout.items():
                holdout_mask &= ~np.isnan(np.asarray(scores))

        # Bail out if test subset is unusable
        if test_mask.sum() < 30 or len(np.unique(y_test[test_mask])) < 2:
            log.info(
                "[%s] baseline-complete subset has %d test rows / %d positive — "
                "skipping baseline-complete plot",
                model_id, int(test_mask.sum()),
                int(np.unique(y_test[test_mask]).size),
            )
            return

        # Filter scores + labels to the masked subset
        score_test_sub = score_test[test_mask]
        y_test_sub = y_test[test_mask]
        bl_test_sub = {
            name: np.asarray(scores)[test_mask]
            for name, scores in baseline_scores_test.items()
        }

        score_ho_sub = None
        y_ho_sub = None
        bl_ho_sub = None
        if (holdout_mask is not None and holdout_mask.sum() >= 30
                and y_holdout is not None
                and len(np.unique(y_holdout[holdout_mask])) >= 2):
            score_ho_sub = score_holdout[holdout_mask]
            y_ho_sub = y_holdout[holdout_mask]
            bl_ho_sub = {
                name: np.asarray(scores)[holdout_mask]
                for name, scores in (baseline_scores_holdout or {}).items()
            }

        _plot_roc_pr_one_panel(
            score_test=score_test_sub,
            y_test=y_test_sub,
            score_holdout=score_ho_sub,
            y_holdout=y_ho_sub,
            output_path=output_dir / "roc_pr_curves__baseline_complete.png",
            title_prefix=(f"Model: {model_id} — baseline-complete subset "
                          f"(all of {', '.join(baseline_scores_test)} computable)"),
            test_n=int(test_mask.sum()),
            holdout_n=int(holdout_mask.sum()) if holdout_mask is not None else None,
            baseline_scores_test=bl_test_sub,
            baseline_scores_holdout=bl_ho_sub,
            n_bootstrap=n_bootstrap, ci_alpha=ci_alpha, bootstrap_seed=bootstrap_seed,
        )
        log.info(
            "[%s] baseline-complete plot: test n=%d/%d (%.1f%%), "
            "holdout n=%s/%s",
            model_id, int(test_mask.sum()), len(y_test),
            100 * test_mask.sum() / max(1, len(y_test)),
            f"{int(holdout_mask.sum())}" if holdout_mask is not None else "—",
            f"{len(y_holdout)}" if y_holdout is not None else "—",
        )


def plot_slice(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path,
    model_id: str,
    suffix: str,
    baseline_scores: dict[str, np.ndarray] | None = None,
    class_names: list[str] | None = None,
    n_bootstrap: int = 0,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> None:
    """Diagnostic figures for an arbitrary row subset (e.g. the phase-complete
    slice).  Writes ``confusion_matrix_<suffix>.png`` (all tasks) and, for
    binary targets, ``roc_pr_curves__<suffix>.png`` (model + any baselines)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if X is None or y is None or len(y) == 0:
        log.info("[%s] slice %s: empty subset, nothing to plot", model_id, suffix)
        return
    try:
        plot_confusion_matrix(model, X, y, suffix, output_dir, model_id, class_names)
    except Exception as exc:
        log.warning("[%s] slice confusion (%s) failed: %s", model_id, suffix, exc)
    if len(np.unique(y)) == 2:
        score = _proba(model, X)
        if score is not None:
            try:
                _plot_roc_pr_one_panel(
                    score_test=score, y_test=np.asarray(y),
                    score_holdout=None, y_holdout=None,
                    output_path=output_dir / f"roc_pr_curves__{suffix}.png",
                    title_prefix=f"Model: {model_id} — {suffix}",
                    test_n=int(len(y)), holdout_n=None,
                    baseline_scores_test=baseline_scores,
                    baseline_scores_holdout=None,
                    n_bootstrap=n_bootstrap, ci_alpha=ci_alpha,
                    bootstrap_seed=bootstrap_seed,
                )
            except Exception as exc:
                log.warning("[%s] slice ROC/PR (%s) failed: %s",
                            model_id, suffix, exc)


def plot_confusion_matrix(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    partition: str,
    output_dir: Path,
    model_id: str,
    class_names: list[str] | None = None,
) -> None:
    """Single confusion matrix for one partition."""
    from sklearn.metrics import confusion_matrix as _cm
    import matplotlib.ticker as ticker

    y_pred = model.predict(X)
    classes = sorted(np.unique(np.concatenate([y, y_pred])))
    cm = _cm(y, y_pred, labels=classes)
    if class_names is None:
        class_names = [str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {partition}\n{model_id}",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{partition}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("[%s] Saved confusion matrix (%s) → %s", model_id, partition, path)


def plot_shap(
    model,
    X: pd.DataFrame,
    output_dir: Path,
    model_id: str,
    max_display: int = 20,
    max_samples: int = 2000,
) -> None:
    """SHAP beeswarm + bar summary plot."""
    try:
        import shap
    except ImportError:
        log.warning("[%s] shap not installed; skipping SHAP plots. "
                    "pip install shap", model_id)
        return

    # Sub-sample for speed
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X, check_additivity=False)
        except Exception:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X)
            except Exception:
                try:
                    explainer = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, "predict_proba")
                        else model.predict,
                        shap.sample(X, 100),
                    )
                    shap_values = explainer.shap_values(X)
                    # KernelExplainer returns array; wrap in Explanation
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    shap_values = shap.Explanation(
                        values=shap_values, data=X.values,
                        feature_names=list(X.columns),
                    )
                except Exception as e:
                    log.warning("[%s] SHAP failed: %s", model_id, e)
                    return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Multiclass SHAP returns (samples, features, classes) — collapse to
    # mean |shap| across classes for the summary plots.  This loses
    # per-class direction info but gives a single picture of feature
    # importance.  Per-class plots can be added separately if needed.
    sv_arr = getattr(shap_values, "values", shap_values)
    is_multiclass = (hasattr(sv_arr, "ndim") and sv_arr.ndim == 3)
    if is_multiclass:
        log.info("[%s] Multiclass SHAP detected — collapsing to mean across classes",
                 model_id)
        import numpy as _np
        collapsed = _np.abs(sv_arr).mean(axis=2)
        shap_values = shap.Explanation(
            values=collapsed,
            data=X.values,
            feature_names=list(X.columns),
        )

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm — {model_id}", fontsize=11)
    plt.tight_layout()
    path = output_dir / "shap_beeswarm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("[%s] Saved SHAP beeswarm → %s", model_id, path)

    # Bar (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Mean |value| — {model_id}", fontsize=11)
    plt.tight_layout()
    path = output_dir / "shap_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("[%s] Saved SHAP bar → %s", model_id, path)


# ---------------------------------------------------------------------------
# Main entry point called from Trainer
# ---------------------------------------------------------------------------
def save_model_plots(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    model_id: str,
    X_holdout: pd.DataFrame | None = None,
    y_holdout: np.ndarray | None = None,
    class_names: list[str] | None = None,
    enable_shap: bool = True,
    baseline_scores_test: dict[str, np.ndarray] | None = None,
    baseline_scores_holdout: dict[str, np.ndarray] | None = None,
    n_bootstrap: int = 0,
    ci_alpha: float = 0.05,
    bootstrap_seed: int = 42,
) -> None:
    """Generate and save all diagnostic plots for one model.

    Round 26: optional ``baseline_scores_test`` / ``baseline_scores_holdout``
    overlay clinical-baseline (ISS / NISS / TRISS) ROC/PR curves on the
    same axes as the model's, computed on the same y_true.  Pass dicts
    keyed by score name → 1D numpy array of scores aligned with
    ``X_test`` / ``X_holdout``.

    Set ``enable_shap=False`` to skip SHAP — recommended for tree
    ensembles with many estimators (e.g. random_forest with n_estimators
    >= 500) on glibc<2.28 systems where SHAP's TreeExplainer can
    segfault on the underlying native code.  Round-16 fix.
    """
    output_dir = Path(output_dir)

    plot_roc_pr(model, X_test, y_test, X_holdout, y_holdout, output_dir, model_id,
                baseline_scores_test=baseline_scores_test,
                baseline_scores_holdout=baseline_scores_holdout,
                n_bootstrap=n_bootstrap, ci_alpha=ci_alpha,
                bootstrap_seed=bootstrap_seed)

    plot_confusion_matrix(model, X_test, y_test, "test", output_dir, model_id,
                          class_names=class_names)
    if X_holdout is not None and y_holdout is not None:
        plot_confusion_matrix(model, X_holdout, y_holdout, "holdout", output_dir,
                              model_id, class_names=class_names)

    if enable_shap:
        plot_shap(model, X_test, output_dir, model_id)
    else:
        log.info("[%s] SHAP disabled (enable_shap=False)", model_id)
    # Final cleanup — release ALL matplotlib figures even if a plot fn forgot.
    try:
        plt.close("all")
    except Exception:
        pass
