"""Hierarchical COMPASS-ladder result visualizations (notebook 03).

Consumes results/compass/ladder/predictions.json (written by utils.run_ladder) and
renders the tier ladder, the diagnosis recovery, and the nested symptom-severity
recovery. Metrics on 10 subjects are illustrative of the pipeline, not definitive.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, spearmanr

from .compass_task import (ALL_OUTPUTS, BPRS_TOTAL, CASE_LABEL, CONTROL_LABEL,
                           SANS_GLOBALS, SAPS_GLOBALS)

TIER_ORDER = ["T1_demographic_socioeconomic", "T2_clinical_profile", "T3_multimodal_full",
              "T4_eeg_lean", "T5_eeg_rich"]
TIER_SHORT = {"T1_demographic_socioeconomic": "T1\ndemo+SES",
              "T2_clinical_profile": "T2\n+clinical",
              "T3_multimodal_full": "T3\n+EEG (full)",
              "T4_eeg_lean": "T4\nEEG lean",
              "T5_eeg_rich": "T5\nEEG rich"}
TIER_COLORS = {"T1_demographic_socioeconomic": "#9c755f", "T2_clinical_profile": "#4e79a7",
               "T3_multimodal_full": "#59a14f", "T4_eeg_lean": "#f28e2b",
               "T5_eeg_rich": "#e15759"}


def load_predictions(path) -> dict[str, Any]:
    return json.loads(path.read_text())


def to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in payload["predictions"]:
        if r.get("error"):
            continue
        pred = r["prediction"]
        prob = (pred.get("diagnosis_probability") or {}).get(CASE_LABEL)
        row = {"tier": r["tier"], "recording_id": r["recording_id"],
               "true_group": r["true_group"],
               "true_case": 1 if r["true_group"] == "Psychosis" else 0,
               "pred_label": pred.get("diagnosis_label"),
               "p_case": prob if prob is not None else np.nan}
        for c in ALL_OUTPUTS:
            row[f"pred__{c}"] = pred.get("regression", {}).get(c, np.nan)
            gt = r["ground_truth"].get(c)
            row[f"true__{c}"] = gt if gt is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _balanced_accuracy(df_tier: pd.DataFrame) -> float:
    pred_case = (df_tier["pred_label"] == CASE_LABEL).astype(int)
    out = []
    for cls in (0, 1):
        m = df_tier["true_case"] == cls
        if m.sum():
            out.append((pred_case[m] == cls).mean())
    return float(np.mean(out)) if out else np.nan


def _auroc(df_tier: pd.DataFrame) -> float:
    y = df_tier["true_case"].to_numpy()
    s = df_tier["p_case"].to_numpy()
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return np.nan
    pos = s[y == 1]; neg = s[y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def diagnosis_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIER_ORDER:
        d = frame[frame["tier"] == tier]
        if d.empty:
            continue
        rows.append({"tier": tier, "balanced_accuracy": _balanced_accuracy(d),
                     "auroc": _auroc(d), "n": len(d)})
    return pd.DataFrame(rows)


def fig_diagnosis_ladder(frame: pd.DataFrame):
    metrics = diagnosis_metrics(frame)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    fig.suptitle("Diagnosis recovery across the evidence ladder", fontsize=14, fontweight="bold")
    order = [t for t in TIER_ORDER if t in metrics["tier"].values]
    x = np.arange(len(order))
    for ax, col, title in [(axes[0], "balanced_accuracy", "Balanced accuracy"),
                           (axes[1], "auroc", "AUROC")]:
        vals = [metrics.set_index("tier").loc[t, col] for t in order]
        ax.bar(x, vals, color=[TIER_COLORS[t] for t in order])
        ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
        ax.set_xticks(x); ax.set_xticklabels([TIER_SHORT[t] for t in order], fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=11); ax.legend(frameon=False, fontsize=8)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    return fig


def fig_diagnosis_heatmap(frame: pd.DataFrame, payload: dict[str, Any]):
    order = [t for t in TIER_ORDER if t in frame["tier"].values]
    ids = payload["cohort"]["psychosis"] + payload["cohort"]["control"]
    mat = np.full((len(ids), len(order)), np.nan)
    for i, rid in enumerate(ids):
        for j, tier in enumerate(order):
            row = frame[(frame["recording_id"] == rid) & (frame["tier"] == tier)]
            if not row.empty:
                mat[i, j] = row["p_case"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([TIER_SHORT[t] for t in order], fontsize=8)
    ax.set_yticks(range(len(ids)))
    truth = {rid: ("Psychosis" if rid in payload["cohort"]["psychosis"] else "Control") for rid in ids}
    ax.set_yticklabels([f"{rid.split('_')[-1]} ({truth[rid][0]})" for rid in ids], fontsize=7)
    ax.axhline(len(payload["cohort"]["psychosis"]) - 0.5, color="k", lw=1.5)
    for i in range(len(ids)):
        for j in range(len(order)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j] - 0.5) > 0.3 else "k")
    ax.set_title("P(First-Episode Psychosis) by subject and tier\n(top block = true cases, bottom = controls)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="P(case)")
    return fig


def fig_symptom_recovery(frame: pd.DataFrame):
    """Predicted vs true BPRS total, for rated (psychosis) subjects, across tiers."""
    order = [t for t in TIER_ORDER if t in frame["tier"].values]
    fig, axes = plt.subplots(1, len(order), figsize=(3.1 * len(order), 3.6),
                             constrained_layout=True, sharex=True, sharey=True)
    if len(order) == 1:
        axes = [axes]
    fig.suptitle("BPRS total severity recovery (rated subjects) across tiers",
                 fontsize=14, fontweight="bold")
    for ax, tier in zip(axes, order):
        d = frame[frame["tier"] == tier]
        t = pd.to_numeric(d[f"true__{BPRS_TOTAL}"], errors="coerce")
        p = pd.to_numeric(d[f"pred__{BPRS_TOTAL}"], errors="coerce")
        ok = t.notna() & p.notna()
        ax.scatter(t[ok], p[ok], color=TIER_COLORS[tier], s=45, alpha=0.85, edgecolor="k", lw=0.4)
        lims = [19, 100]
        ax.plot(lims, lims, "k--", lw=0.7)
        r = pearsonr(t[ok], p[ok])[0] if ok.sum() >= 3 else np.nan
        ax.set_title(f"{TIER_SHORT[tier].splitlines()[0]}  r={r:.2f}" if np.isfinite(r)
                     else TIER_SHORT[tier].splitlines()[0], fontsize=10)
        ax.set_xlabel("true BPRS"); ax.set_xlim(19, 100); ax.set_ylim(19, 100)
    axes[0].set_ylabel("predicted BPRS")
    return fig


def fig_symptom_dimensions(frame: pd.DataFrame, tier: str = "T3_multimodal_full"):
    """SAPS positive and SANS negative global-rating recovery for one tier."""
    d = frame[frame["tier"] == tier]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Symptom-dimension recovery at {TIER_SHORT[tier].splitlines()[0].strip()} "
                 f"({tier})", fontsize=14, fontweight="bold")
    for ax, mapping, name, color in [
            (axes[0], SAPS_GLOBALS, "SAPS positive globals", "#e15759"),
            (axes[1], SANS_GLOBALS, "SANS negative globals", "#4e79a7")]:
        ts, ps = [], []
        for col in mapping.values():
            t = pd.to_numeric(d[f"true__{col}"], errors="coerce")
            p = pd.to_numeric(d[f"pred__{col}"], errors="coerce")
            ok = t.notna() & p.notna()
            ts += t[ok].tolist(); ps += p[ok].tolist()
        ax.scatter(ts, ps, color=color, s=40, alpha=0.7, edgecolor="k", lw=0.4)
        ax.plot([0, 5], [0, 5], "k--", lw=0.7)
        r = pearsonr(ts, ps)[0] if len(ts) >= 3 else np.nan
        rho = spearmanr(ts, ps)[0] if len(ts) >= 3 else np.nan
        ax.set_title(f"{name}\nPearson r={r:.2f}, Spearman rho={rho:.2f}" if np.isfinite(r) else name,
                     fontsize=10)
        ax.set_xlabel("true global rating (0-5)"); ax.set_ylabel("predicted")
        ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.3, 5.3)
    return fig


def fig_ladder_reading(frame: pd.DataFrame):
    """The tier ladder read as balanced accuracy with the key contrasts annotated."""
    metrics = diagnosis_metrics(frame).set_index("tier")
    order = [t for t in TIER_ORDER if t in metrics.index]
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    x = np.arange(len(order))
    ba = [metrics.loc[t, "balanced_accuracy"] for t in order]
    ax.plot(x, ba, "o-", color="#333333", lw=2, markersize=9)
    for xi, t in zip(x, order):
        ax.scatter(xi, metrics.loc[t, "balanced_accuracy"], s=140, color=TIER_COLORS[t], zorder=3)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([TIER_SHORT[t] for t in order], fontsize=9)
    ax.set_ylabel("balanced accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Evidence ladder: what each modality adds to diagnosis recovery",
                 fontsize=13, fontweight="bold")
    notes = [("T1_demographic_socioeconomic", "T2_clinical_profile", "proxy gain"),
             ("T2_clinical_profile", "T3_multimodal_full", "neural lift"),
             ("T4_eeg_lean", "T5_eeg_rich", "lean vs rich")]
    idx = {t: i for i, t in enumerate(order)}
    for a, b, label in notes:
        if a in idx and b in idx:
            xa, xb = idx[a], idx[b]
            ymid = (metrics.loc[a, "balanced_accuracy"] + metrics.loc[b, "balanced_accuracy"]) / 2
            ax.annotate("", xy=(xb, metrics.loc[b, "balanced_accuracy"]),
                        xytext=(xa, metrics.loc[a, "balanced_accuracy"]),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
            ax.text((xa + xb) / 2, ymid + 0.04, label, fontsize=8, color="gray",
                    ha="center", style="italic")
    return fig


def fig_hierarchy_tree(frame: pd.DataFrame, payload: dict[str, Any],
                       tier: str = "T3_multimodal_full", recording_id: str | None = None):
    """The full mixed-type prediction tree for one subject: diagnosis -> BPRS ->
    SAPS/SANS globals, predicted against ground truth."""
    if recording_id is None:
        recording_id = payload["cohort"]["psychosis"][len(payload["cohort"]["psychosis"]) // 2]
    row = frame[(frame["tier"] == tier) & (frame["recording_id"] == recording_id)]
    row = row.iloc[0]
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax.axis("off")
    ax.set_title(f"Hierarchical prediction for {recording_id} at {tier}\n"
                 f"(true group: {row['true_group']})", fontsize=13, fontweight="bold")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="k",
                                   lw=1.1, alpha=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, zorder=3)

    prob = row["p_case"]
    box(0.36, 0.86, 0.28, 0.1,
        f"DIAGNOSIS\npred: {row['pred_label']}\nP(case)={prob:.2f}" if np.isfinite(prob)
        else f"DIAGNOSIS\npred: {row['pred_label']}", "#bdd7e7")
    bt, bp = row[f"true__{BPRS_TOTAL}"], row[f"pred__{BPRS_TOTAL}"]
    box(0.36, 0.66, 0.28, 0.1,
        f"BPRS total\npred {bp:.0f}  (true {bt:.0f})" if np.isfinite(bt) else f"BPRS total\npred {bp:.0f}",
        "#c7e9c0")
    ax.plot([0.5, 0.5], [0.86, 0.76], "k-", lw=1)

    def leaf_block(x0, mapping, title, color):
        y = 0.46
        box(x0, y + 0.06, 0.30, 0.06, title, color)
        ax.plot([0.5, x0 + 0.15], [0.66, y + 0.12], "k-", lw=0.8)
        for i, (name, col) in enumerate(mapping.items()):
            t, p = row[f"true__{col}"], row[f"pred__{col}"]
            txt = f"{name.replace('_',' ')}: pred {p:.1f}" + (f" (true {t:.0f})" if np.isfinite(t) else "")
            ax.text(x0 + 0.02, y - i * 0.055, txt, fontsize=7.5, va="top")
    leaf_block(0.04, SAPS_GLOBALS, "SAPS positive globals", "#fcae91")
    leaf_block(0.66, SANS_GLOBALS, "SANS negative globals", "#bdd7e7")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig
