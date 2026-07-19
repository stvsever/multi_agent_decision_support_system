#!/usr/bin/env python3
"""
Generate and execute the AOMIC ID1000 preprocessing notebooks.

Builds two clean, professional notebooks with embedded visualisations and runs
them (nbclient) so the committed .ipynb files render without re-execution:

  01_tabular_data_exploration.ipynb   phenotype distributions, missingness,
                                       feature-feature and feature-target correlations
  02_brain_preprocessing.ipynb        FreeSurfer morphometry distributions/correlations
                                       and nilearn connectome matrices + glass brain

Run after the pipeline has produced the ontology and brain features:
  python build_notebooks.py
"""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


# --------------------------------------------------------------------------- #
# Notebook 1: tabular data exploration
# --------------------------------------------------------------------------- #
def tabular_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# AOMIC-ID1000 - Tabular Phenotype Exploration

Prediction target: **total intelligence** (`IST_intelligence_total`, IST 2000-R composite).
This notebook profiles the self-report phenotype that feeds the lower complexity tiers:
target distribution, missingness, feature-feature correlation structure, and which
features actually track intelligence. All values shown here are the raw dataset values
(before the ontology projection and z-scoring done by the ingestion pipeline).
"""),
        code("""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="notebook")
ROOT = Path.cwd().parent
df = pd.read_csv(ROOT / "dataset" / "participants.tsv", sep="\\t", na_values=["n/a","N/A",""])
manifest = json.load(open(ROOT / "ontology" / "feature_manifest.json"))
target = manifest["target"]["column"]
print("participants:", len(df), "| target:", target)
"""),
        md("## Target: total intelligence and its subscales"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
t = pd.to_numeric(df[target], errors="coerce").dropna()
ax[0].hist(t, bins=30, color="#6366f1", edgecolor="white")
ax[0].axvline(t.mean(), color="#ef4444", ls="--", label=f"mean {t.mean():.0f}")
ax[0].set(title="Total intelligence (IST 2000-R)", xlabel="score", ylabel="participants"); ax[0].legend()
for c, col in zip(["#10b981","#f59e0b","#38bdf8"], ["IST_fluid","IST_memory","IST_crystallised"]):
    ax[1].hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=25, alpha=0.6, label=col, color=c)
ax[1].set(title="IST subscales (excluded as predictors: circular)", xlabel="score"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md("""
The three IST subscales sum to the total, so they are excluded from the predictors.
Everything the tiers use is *non-cognitive*: personality, motivation, affect,
demographics, identity, and later brain structure and connectivity.
"""),
        md("## Predictor groups and missingness"),
        code("""
preds = pd.DataFrame(manifest["predictors"])
groups = preds.groupby("column").first()
order = preds.sort_values("coverage_pct")
fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#ef4444" if c < 70 else "#f59e0b" if c < 95 else "#10b981" for c in order["coverage_pct"]]
ax.barh(order["label"], order["coverage_pct"], color=colors)
ax.set(title="Feature coverage (percent of participants present)", xlabel="coverage %")
plt.tight_layout(); plt.show()
print("Sparsest features:", ", ".join(order.head(4)["column"]))
"""),
        md("""
Missingness varies a lot: personality and demographics are near-complete, the
sexual/gender identity ratings are ~60% present, and `religious_importance` is
only ~21% present. The ingestion pipeline encodes missing leaves explicitly, so the
engine sees which evidence is absent rather than silently imputed.
"""),
        md("## Feature-feature correlation structure"),
        code("""
# Tabular self-report features only (brain modalities are covered in notebook 2).
num_cols = [p["column"] for p in manifest["predictors"]
            if p["stat_type"] == "numeric" and p["column"] in df.columns
            and not p["column"].startswith(("fs_", "fc_"))]
corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True, linewidths=.5,
            cbar_kws={"shrink": .7}, ax=ax)
ax.set_title("Correlation among numeric self-report features")
plt.tight_layout(); plt.show()
"""),
        md("""
The block structure (BAS scales together, NEO scales, etc.) is exactly the
instrument grouping the ontology encodes as subdomains. A clean subclass ontology
is not cosmetic: it gives the engine that block structure explicitly so it can reason
per construct instead of over a flat list of numbers.
"""),
        md("## Which features track intelligence?"),
        code("""
t = pd.to_numeric(df[target], errors="coerce")
rows = []
for c in num_cols:
    x = pd.to_numeric(df[c], errors="coerce")
    m = x.notna() & t.notna()
    if m.sum() > 30:
        rows.append((c, float(np.corrcoef(x[m], t[m])[0,1])))
rho = pd.DataFrame(rows, columns=["feature","r"]).sort_values("r")
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(rho["feature"], rho["r"], color=["#ef4444" if v<0 else "#10b981" for v in rho["r"]])
ax.axvline(0, color="#333"); ax.set(title="Pearson r with total intelligence", xlabel="r")
plt.tight_layout(); plt.show()
"""),
        md("""
Correlations with intelligence are individually weak (education and openness are the
usual positive signals, consistent with the literature). This is why the engine is
asked to recover the *ranking* of participants rather than exact scores, and why
adding brain modalities in later tiers is worth testing.
"""),
    ]
    return nb


# --------------------------------------------------------------------------- #
# Notebook 2: brain preprocessing
# --------------------------------------------------------------------------- #
def brain_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# AOMIC-ID1000 - Brain Preprocessing and Visualisation

Two brain modalities feed the highest complexity tiers, kept as separate ontology
domains:

* **Morphometry (FreeSurfer)** - subcortical volumes, cortical thickness by lobe, and
  global brain measures, parsed from the per-subject stats tables.
* **Connectome (movie-watching fMRI)** - Schaefer-100 / Yeo-7 parcellation reduced to
  network-level functional connectivity.

This notebook visualises both and their relationship to intelligence.
"""),
        code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, datasets
sns.set_theme(style="whitegrid", context="notebook")
ROOT = Path.cwd().parent
part = pd.read_csv(ROOT / "dataset" / "participants.tsv", sep="\\t", na_values=["n/a",""])
target = pd.to_numeric(part.set_index("participant_id")["IST_intelligence_total"], errors="coerce")
morph = pd.read_csv(ROOT / "brain" / "freesurfer" / "morphometry_features.csv", index_col=0)
print("morphometry:", morph.shape, "| connectome subjects available:",
      len(list((ROOT / "brain" / "connectome" / "network_fc").glob("*.npy"))))
"""),
        md("## FreeSurfer morphometry: subcortical volumes"),
        code("""
sub_cols = [c for c in morph.columns if c.startswith("fs_vol_")]
melt = morph[sub_cols].melt(var_name="structure", value_name="volume")
melt["structure"] = melt["structure"].str.replace("fs_vol_","",regex=False)
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=melt, x="structure", y="volume", ax=ax, color="#6366f1")
ax.set(title="Subcortical volumes across the reference cohort", ylabel="mm^3")
plt.xticks(rotation=60, ha="right"); plt.tight_layout(); plt.show()
"""),
        md("## Morphometry correlation structure and link to intelligence"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":[1.3,1]})
sns.heatmap(morph.corr(), cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink":.6}, ax=ax[0])
ax[0].set_title("Morphometry feature correlations")
rows = []
for c in morph.columns:
    m = morph[c].notna() & target.reindex(morph.index).notna()
    if m.sum() > 30:
        rows.append((c, float(np.corrcoef(morph[c][m], target.reindex(morph.index)[m])[0,1])))
rho = pd.DataFrame(rows, columns=["feature","r"]).sort_values("r").tail(12)
ax[1].barh(rho["feature"], rho["r"], color=["#ef4444" if v<0 else "#10b981" for v in rho["r"]])
ax[1].axvline(0, color="#333"); ax[1].set_title("Top morphometry correlations with IQ")
plt.tight_layout(); plt.show()
"""),
        md("""
Global volume measures (intracranial and gray-matter volume) carry the familiar weak
positive brain-size / intelligence association. Individual regions are noisier at this
sample size, which is why morphometry enters as its own tier to see its marginal value.
"""),
        md("## Functional connectome: network-level FC matrix"),
        code("""
YEO7 = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
LBL = ["Visual","Somatomotor","DorsAttn","Salience/VentAttn","Limbic","Frontoparietal","Default"]
fc_dir = ROOT / "brain" / "connectome" / "network_fc"
mats = np.stack([np.load(p) for p in sorted(fc_dir.glob("*.npy"))])
group_fc = np.nanmean(mats, axis=0)
fig = plt.figure(figsize=(6.5, 5.5))
plotting.plot_matrix(group_fc, labels=LBL, colorbar=True, vmin=-1, vmax=1,
                     title="Group-average Yeo-7 network FC", reorder=False)
plt.show()
"""),
        md("""
This 7x7 matrix is the reduced connectome the engine sees: 7 within-network and 21
between-network mean correlations (28 features). Reducing ~5000 parcel edges to 28
interpretable, network-labelled features is what makes the connectome usable by an
LLM-reasoning engine without losing the meaning of each value.
"""),
        md("## Glass-brain connectome (one subject, parcel level)"),
        code("""
# Recompute one subject's parcel-level FC to visualise the full network on a glass brain.
repo_root = next(p for p in [ROOT, *ROOT.parents] if (p / "src" / "full_stack").is_dir())
sys.path.insert(0, str(repo_root))  # for validation.common
from validation.common import connectome as conn
sub = sorted(p.stem for p in fc_dir.glob("*.npy"))[0]
atlas = conn.load_atlas(100, 7, 2)
coords = plotting.find_parcellation_cut_coords(atlas.maps)
paths = conn.download_func(sub, "ds003097", ROOT / "brain" / "_cache" / "connectome")
ts = conn.parcel_timeseries(paths, atlas)
parcel_fc = np.corrcoef(ts.T)
plotting.plot_connectome(parcel_fc, coords, edge_threshold="99.2%", node_size=18,
                         title=f"Movie-watching FC ({sub}, top 0.8% edges)")
plt.show()
for k in ("bold","mask"):
    p = paths.get(k)
    if p and Path(p).exists(): Path(p).unlink()  # tidy the large download
"""),
        md("""
The glass brain shows the strongest parcel-to-parcel connections for one participant,
computed by the exact same pipeline used for feature extraction. The committed engine
features are the network-level summary above; this full view is for inspection only.
"""),
    ]
    return nb


def main():
    for name, builder in [("01_tabular_data_exploration", tabular_notebook),
                          ("02_brain_preprocessing", brain_notebook)]:
        nb = builder()
        print(f"[notebooks] executing {name} ...")
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(f"[notebooks] wrote {name}.ipynb")


if __name__ == "__main__":
    main()
