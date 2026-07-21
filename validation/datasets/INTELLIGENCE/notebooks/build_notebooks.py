#!/usr/bin/env python3
"""
Generate and execute the AOMIC ID1000 exploration/preprocessing notebooks.

Builds three richly-visualised, self-contained notebooks and runs them (nbclient)
so the committed .ipynb files render without re-execution:

  01_tabular_data_exploration.ipynb   phenotype distributions, missingness, correlation
                                       structure and dendrogram, IQ associations, PCA
  02_brain_preprocessing.ipynb        FreeSurfer morphometry + fMRI connectome, with
                                       nilearn matrices, glass brain, and IQ links
  03_ontology_and_results.ipynb       automated exploration, the ontology graph,
                                       cluster agreement, tier ladder, per-tier results

Run after the pipeline has produced the ontology, brain features, and tier results:
  python build_notebooks.py
"""

import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


PRELUDE = """
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
ROOT = Path.cwd().parent
IND = "#6366f1"; GRN = "#10b981"; ORG = "#f59e0b"; RED = "#ef4444"; BLU = "#38bdf8"

def _leaves(node):
    ch = node.get("children")
    if not ch:
        return [node]
    out = []
    for c in ch:
        out += _leaves(c)
    return out

def n_leaves(node):
    return len(_leaves(node))

def mm_leaves(node, out):
    # Collect every _leaves record from an arbitrary-depth multimodal tree.
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "_leaves":
                out.extend(v)
            elif isinstance(v, dict):
                mm_leaves(v, out)
"""


# =========================================================================== #
# Notebook 1: tabular exploration
# =========================================================================== #
def nb_tabular():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# AOMIC-ID1000 - Tabular Phenotype Exploration

Target: **total intelligence** (`IST_intelligence_total`). This notebook profiles the
self-report phenotype that feeds the lower complexity tiers, with distributions,
missingness structure, correlation structure, associations with intelligence, and a
low-dimensional view of participants. Raw dataset values are shown here (before the
ontology projection and z-scoring done by the ingestion pipeline).
"""),
        code(PRELUDE + """
df = pd.read_csv(ROOT / "dataset" / "participants.tsv", sep="\\t", na_values=["n/a","N/A",""])
manifest = json.load(open(ROOT / "ontology" / "feature_manifest.json"))
explore = json.load(open(ROOT / "ontology" / "exploration_report.json"))
target = manifest["target"]["column"]
tab = [p["column"] for p in manifest["predictors"] if not p["column"].startswith(("fs_","fc_"))]
tab_num = [p["column"] for p in manifest["predictors"]
           if p["stat_type"]=="numeric" and p["column"] in df.columns and not p["column"].startswith(("fs_","fc_"))]
print(len(df), "participants |", len(tab), "tabular predictors")
"""),
        md("## 1. Target and its subscales"),
        code("""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
t = pd.to_numeric(df[target], errors="coerce")
ax[0].hist(t.dropna(), bins=30, color=IND, edgecolor="white")
ax[0].axvline(t.mean(), color=RED, ls="--", label=f"mean {t.mean():.0f}")
ax[0].set(title="Total intelligence", xlabel="IST score"); ax[0].legend()
for c, col in zip([GRN,ORG,BLU], ["IST_fluid","IST_memory","IST_crystallised"]):
    ax[1].hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=25, alpha=0.6, label=col, color=c)
ax[1].set(title="IST subscales (excluded: circular)", xlabel="score"); ax[1].legend(fontsize=8)
sub = df[["IST_fluid","IST_memory","IST_crystallised"]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
ax[2].scatter(sub, t, s=10, alpha=0.4, color=IND)
ax[2].plot([t.min(),t.max()],[t.min(),t.max()], color=RED, ls="--")
ax[2].set(title="total vs sum(subscales)", xlabel="fluid+memory+crystallised", ylabel="total")
plt.tight_layout(); plt.show()
"""),
        md("## 2. Coverage and missingness co-occurrence"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":[1,1.1]})
prof = pd.DataFrame(manifest["predictors"])
tt = prof[prof["column"].isin(tab)].sort_values("coverage_pct")
cols = [RED if c<70 else ORG if c<95 else GRN for c in tt["coverage_pct"]]
ax[0].barh(tt["label"], tt["coverage_pct"], color=cols)
ax[0].set(title="Feature coverage (%)", xlabel="present %")
miss = df[tab].isna().astype(int)
co = miss.corr().fillna(0)
sns.heatmap(co, cmap="rocket", ax=ax[1], cbar_kws={"shrink":.6}, xticklabels=False)
ax[1].set_title("Missingness co-occurrence (features missing together)")
plt.tight_layout(); plt.show()
"""),
        md("## 3. Distributions of every self-report feature"),
        code("""
n = len(tab_num); ncol = 4; nrow = int(np.ceil(n/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6*nrow))
for ax, c in zip(axes.flat, tab_num):
    v = pd.to_numeric(df[c], errors="coerce").dropna()
    ax.hist(v, bins=20, color=IND, edgecolor="white", alpha=0.85)
    ax.set_title(c, fontsize=8); ax.tick_params(labelsize=7)
for ax in axes.flat[n:]:
    ax.axis("off")
plt.suptitle("Numeric self-report feature distributions", y=1.005); plt.tight_layout(); plt.show()
"""),
        md("## 4. Categorical features"),
        code("""
cats = [p["column"] for p in manifest["predictors"]
        if p["stat_type"] in ("binary","ordinal","nominal") and p["column"] in df.columns
        and not p["column"].startswith(("fs_","fc_"))]
fig, axes = plt.subplots(1, len(cats), figsize=(3*len(cats), 3.2))
for ax, c in zip(np.atleast_1d(axes), cats):
    df[c].astype(str).value_counts().plot(kind="bar", ax=ax, color=IND)
    ax.set_title(c, fontsize=9); ax.tick_params(axis="x", labelsize=8, rotation=30)
plt.suptitle("Categorical features", y=1.03); plt.tight_layout(); plt.show()
"""),
        md("## 5. Correlation structure (clustered) and dendrogram\\n"
           "The intelligence target (`IST_intelligence_total`, highlighted in red) is included in "
           "the matrix, so the clustering places it next to the self-report features it correlates "
           "with most. The ranked list below reads off, directly from the data, which features most "
           "strongly predict intelligence."),
        code("""
cols = tab_num + [target]
corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
try:
    g = sns.clustermap(corr, cmap="RdBu_r", center=0, figsize=(10.5,10.5), linewidths=.3,
                       cbar_pos=(0.02,0.83,0.03,0.12))
    g.fig.suptitle("Clustered Spearman correlation (self-report + intelligence target)", y=1.02)
    for tl in g.ax_heatmap.get_xticklabels() + g.ax_heatmap.get_yticklabels():
        if tl.get_text() == target:
            tl.set_color(RED); tl.set_fontweight("bold")
    plt.show()
except Exception as e:
    print("clustermap skipped:", e)
# Which self-report features most strongly predict intelligence, straight from the data?
tcorr = corr[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(tcorr.index[::-1], tcorr.values[::-1],
        color=[RED if v < 0 else GRN for v in tcorr.values[::-1]])
ax.axvline(0, color="#333")
ax.set(title="Self-report features ranked by |Spearman r| with intelligence (from data)",
       xlabel="Spearman r with IST_intelligence_total")
plt.tight_layout(); plt.show()
print("Spearman correlation with", target, "(strongest first):")
for f, r in tcorr.items():
    print(f"  {r:+.3f}  {f}")
"""),
        code("""
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
d = 1 - corr.abs().values; np.fill_diagonal(d,0); d=(d+d.T)/2
Z = linkage(squareform(d, checks=False), method="average")
fig, ax = plt.subplots(figsize=(12,4))
dendrogram(Z, labels=list(corr.columns), leaf_rotation=90, color_threshold=0.7, ax=ax)
ax.set_title("Feature clustering dendrogram (1 - |Spearman r|)"); plt.tight_layout(); plt.show()
"""),
        md("## 6. What tracks intelligence?"),
        code("""
assoc = pd.DataFrame(explore.get("target_associations", []))
assoc = assoc[~assoc["feature"].str.startswith(("fs_","fc_"))].sort_values("spearman_r")
fig, ax = plt.subplots(figsize=(8,6))
ax.barh(assoc["label"], assoc["spearman_r"], color=[RED if v<0 else GRN for v in assoc["spearman_r"]])
ax.axvline(0, color="#333"); ax.set(title="Spearman correlation with total intelligence", xlabel="rho")
plt.tight_layout(); plt.show()
"""),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
order = ["low","medium","high"]
sns.violinplot(data=df.assign(_t=pd.to_numeric(df[target],errors="coerce")),
               x="education_level", y="_t", order=order, ax=ax[0], hue="education_level",
               palette="viridis", legend=False)
ax[0].set(title="Intelligence by education level", xlabel="education", ylabel="IST total")
sns.violinplot(data=df.assign(_t=pd.to_numeric(df[target],errors="coerce")),
               x="sex", y="_t", ax=ax[1], hue="sex", palette="mako", legend=False)
ax[1].set(title="Intelligence by sex", xlabel="", ylabel="IST total")
plt.tight_layout(); plt.show()
"""),
        md("## 7. Top associates and a participant embedding"),
        code("""
top = assoc.reindex(assoc["spearman_r"].abs().sort_values(ascending=False).index)["feature"].head(4).tolist()
pair = df[top + [target]].apply(pd.to_numeric, errors="coerce").dropna()
g = sns.pairplot(pair, corner=True, plot_kws={"s":8,"alpha":0.4,"color":IND}, diag_kws={"color":IND})
g.fig.suptitle("Top IQ-associated features vs each other and the target", y=1.02); plt.show()
"""),
        code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = df[tab_num].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())
Z2 = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
fig, ax = plt.subplots(figsize=(7.5,6))
sc = ax.scatter(Z2[:,0], Z2[:,1], c=pd.to_numeric(df[target],errors="coerce"), cmap="viridis", s=14, alpha=0.8)
plt.colorbar(sc, label="total intelligence")
ax.set(title="Participants in self-report feature space (PCA), coloured by IQ",
       xlabel="PC1", ylabel="PC2")
plt.tight_layout(); plt.show()
"""),
        md("Personality and education carry the strongest (still weak) individual signals; "
           "no single feature separates high from low intelligence, which is why the engine is "
           "asked to recover a ranking from the whole multi-modal profile."),
    ]
    return nb


# =========================================================================== #
# Notebook 2: brain preprocessing
# =========================================================================== #
def nb_brain():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# AOMIC-ID1000 - Brain Preprocessing and Visualisation

Two brain modalities feed the highest tiers, both nested under a single **Brain**
domain: **Morphometry** (high-resolution FreeSurfer output - per-region Desikan-Killiany
cortical thickness, surface area and gray-matter volume, plus subcortical and global
volumes) and **Connectomics** (Schaefer-100 / Yeo-7 functional connectivity from
movie-watching fMRI). This notebook visualises both and their relationship to
intelligence. The morphometry branch alone contributes 228 leaf features.
"""),
        code(PRELUDE + """
from nilearn import plotting, datasets
part = pd.read_csv(ROOT / "dataset" / "participants.tsv", sep="\\t", na_values=["n/a",""])
target = pd.to_numeric(part.set_index("participant_id")["IST_intelligence_total"], errors="coerce")
morph = pd.read_csv(ROOT / "brain" / "freesurfer" / "morphometry_features.csv", index_col=0)
conn = pd.read_csv(ROOT / "brain" / "connectome" / "connectome_features.csv", index_col=0)
fc_dir = ROOT / "brain" / "connectome" / "network_fc"
print("morphometry", morph.shape, "| connectome", conn.shape)
"""),
        md("## 1. Subcortical volumes: distribution and hemispheric symmetry"),
        code("""
sub = [c for c in morph.columns if c.startswith("fs_vol_")]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
m = morph[sub].melt(var_name="s", value_name="v"); m["s"]=m["s"].str.replace("fs_vol_","",regex=False)
sns.boxplot(data=m, x="s", y="v", ax=ax[0], color=IND); ax[0].tick_params(axis="x", rotation=60, labelsize=8)
ax[0].set(title="Subcortical volumes", ylabel="mm^3", xlabel="")
structs = sorted(set(c.replace("fs_vol_lh_","").replace("fs_vol_rh_","") for c in sub))
for s in structs:
    l, r = f"fs_vol_lh_{s}", f"fs_vol_rh_{s}"
    if l in morph and r in morph:
        ax[1].scatter(morph[l], morph[r], s=12, alpha=0.6, label=s)
lims=[morph[sub].min().min(), morph[sub].max().max()]
ax[1].plot(lims, lims, color=RED, ls="--"); ax[1].set(title="Left vs right volume", xlabel="left mm^3", ylabel="right mm^3")
ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
"""),
        md("## 2. Head-size scaling and per-region cortical thickness map"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
if "fs_etiv" in morph:
    for c,col in zip(["fs_total_gray_vol","fs_cerebral_wm_vol","fs_subcort_gray_vol"],[GRN,ORG,BLU]):
        if c in morph: ax[0].scatter(morph["fs_etiv"], morph[c], s=12, alpha=0.6, label=c, color=col)
    ax[0].set(title="Tissue volumes scale with intracranial volume", xlabel="eTIV mm^3", ylabel="volume mm^3"); ax[0].legend(fontsize=8)
thk = [c for c in morph.columns if c.startswith("fs_thk_")]
z = (morph[thk]-morph[thk].mean())/morph[thk].std()
sns.heatmap(z.T, cmap="RdBu_r", center=0, ax=ax[1], cbar_kws={"shrink":.6}, xticklabels=False, yticklabels=False)
ax[1].set(title=f"Cortical thickness by region (z, {len(thk)} regions x subjects)", xlabel="subjects (reference cohort)", ylabel="Desikan-Killiany regions")
plt.tight_layout(); plt.show()
"""),
        md("## 3. Morphometry correlation structure and links to intelligence"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":[1.3,1]})
sns.heatmap(morph.corr(), cmap="RdBu_r", center=0, square=True, cbar_kws={"shrink":.6}, ax=ax[0], xticklabels=False, yticklabels=False)
ax[0].set_title(f"Morphometry feature correlations ({morph.shape[1]} features)")
tgt = target.reindex(morph.index)
rows=[]
for c in morph.columns:
    mm = morph[c].notna() & tgt.notna()
    if mm.sum()>30: rows.append((c, float(np.corrcoef(morph[c][mm], tgt[mm])[0,1])))
rho = pd.DataFrame(rows, columns=["f","r"]).sort_values("r")
rho2 = rho.reindex(rho["r"].abs().sort_values(ascending=False).index).head(14).sort_values("r")
ax[1].barh(rho2["f"], rho2["r"], color=[RED if v<0 else GRN for v in rho2["r"]])
ax[1].axvline(0,color="#333"); ax[1].set_title("Top morphometry-IQ correlations"); ax[1].tick_params(labelsize=7)
plt.tight_layout(); plt.show()
"""),
        code("""
glob = [c for c in ["fs_etiv","fs_total_gray_vol","fs_mean_thickness_lh"] if c in morph]
fig, ax = plt.subplots(1, len(glob), figsize=(5*len(glob), 4))
for a, c in zip(np.atleast_1d(ax), glob):
    mm = morph[c].notna() & tgt.notna()
    a.scatter(morph[c][mm], tgt[mm], s=14, alpha=0.6, color=IND)
    r = np.corrcoef(morph[c][mm], tgt[mm])[0,1]
    a.set(title=f"{c}\\nr={r:.2f}", xlabel=c, ylabel="IST total")
plt.tight_layout(); plt.show()
"""),
        code("""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Mp = morph.dropna()
if len(Mp) > 10:
    Z2 = PCA(2).fit_transform(StandardScaler().fit_transform(Mp))
    fig, axp = plt.subplots(figsize=(7,5.5))
    sc = axp.scatter(Z2[:,0], Z2[:,1], c=target.reindex(Mp.index), cmap="viridis", s=18)
    plt.colorbar(sc, label="IST total"); axp.set(title="Morphometry PCA, coloured by IQ", xlabel="PC1", ylabel="PC2")
    plt.tight_layout(); plt.show()
"""),
        md("## 4. Functional connectome: network matrices"),
        code("""
YEO7 = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
LBL = ["Visual","SomMot","DorsAttn","Sal/VentAttn","Limbic","Frontopar","Default"]
mats = {p.stem: np.load(p) for p in sorted(fc_dir.glob("*.npy"))}
group = np.nanmean(np.stack(list(mats.values())), axis=0)
fig = plt.figure(figsize=(6.5,5.5))
plotting.plot_matrix(group, labels=LBL, colorbar=True, vmin=-1, vmax=1, title="Group-average Yeo-7 network FC")
plt.show()
"""),
        code("""
ids = list(mats)[:6]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, pid in zip(axes.flat, ids):
    im = ax.imshow(mats[pid], vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title(pid, fontsize=9); ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(YEO7, rotation=90, fontsize=6); ax.set_yticklabels(YEO7, fontsize=6)
fig.colorbar(im, ax=axes, shrink=0.6, label="FC (r)")
plt.suptitle("Per-subject network FC (individual variation)", y=1.02); plt.show()
"""),
        code("""
within = [c for c in conn.columns if c.endswith("_within")]
between = [c for c in conn.columns if c not in within]
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].hist(conn[within].values.flatten(), bins=25, alpha=0.7, color=GRN, label="within-network")
ax[0].hist(conn[between].values.flatten(), bins=25, alpha=0.7, color=ORG, label="between-network")
ax[0].set(title="FC edge strengths", xlabel="Pearson r"); ax[0].legend()
tgtc = target.reindex(conn.index)
rows=[]
for c in conn.columns:
    mm = conn[c].notna() & tgtc.notna()
    if mm.sum()>15: rows.append((c, float(np.corrcoef(conn[c][mm], tgtc[mm])[0,1])))
rc = pd.DataFrame(rows, columns=["f","r"])
rc = rc.reindex(rc["r"].abs().sort_values(ascending=False).index).head(12).sort_values("r")
ax[1].barh(rc["f"], rc["r"], color=[RED if v<0 else GRN for v in rc["r"]])
ax[1].axvline(0,color="#333"); ax[1].set_title("Top connectome-IQ correlations"); ax[1].tick_params(labelsize=7)
plt.tight_layout(); plt.show()
"""),
        md("## 5. Parcel-level connectome and glass brain (one subject)\\n"
           "This optional view reconstructs the full 100x100 parcel FC from the raw movie-watching "
           "BOLD. The BOLD is not cached (only the reduced 7x7 network matrices are), so it runs only "
           "if the file is already present locally; otherwise it is skipped with a note. Re-extracting "
           "at a finer atlas (e.g. 17 Yeo sub-networks) is a config knob in `validation/common/connectome.py`."),
        code("""
repo = next(p for p in [ROOT,*ROOT.parents] if (p/"src"/"full_stack").is_dir())
sys.path.insert(0, str(repo))
from validation.common import connectome as C
sub = list(mats)[0]
cache = ROOT/"brain"/"_cache"/"connectome"
space = "space-MNI152NLin2009cAsym"
bold_cached = (cache / f"{sub}_task-moviewatching_{space}_desc-preproc_bold.nii.gz").exists()
if bold_cached:
    atlas = C.load_atlas(100,7,2); coords = plotting.find_parcellation_cut_coords(atlas.maps)
    paths = C.download_func(sub, "ds003097", cache)
    ts = C.parcel_timeseries(paths, atlas); pfc = np.corrcoef(ts.T)
    fig, ax = plt.subplots(figsize=(6.5,5.5))
    im = ax.imshow(pfc, vmin=-1, vmax=1, cmap="RdBu_r"); ax.set_title(f"Parcel-level FC 100x100 ({sub})")
    plt.colorbar(im, label="r"); plt.tight_layout(); plt.show()
    plotting.plot_connectome(pfc, coords, edge_threshold="99.3%", node_size=16,
                             title=f"Movie-watching FC glass brain ({sub}, top 0.7% edges)")
    plt.show()
else:
    print(f"Raw BOLD for {sub} not cached; skipping parcel-level reconstruction.")
    print("To enable: run validation.common.connectome.download_func for one subject, then re-run.")
"""),
        md("Naturalistic-viewing connectivity varies clearly between individuals; the engine "
           "receives the compact 28-value network summary, while these parcel-level views are "
           "for inspection. Morphometry and connectomics both nest under the single Brain domain."),
    ]
    return nb


# =========================================================================== #
# Notebook 3: ontology and results
# =========================================================================== #
def nb_ontology():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# AOMIC-ID1000 - Automated Ontology and Tiered Results

This notebook shows the automated data understanding that grounds the ontology, the
resulting ontology graph, how well the semantic ontology matches the data-driven
clusters, the complexity-tier ladder, and per-tier engine performance.

For hands-on exploration, open `../ontology/ontology_viewer.html` in a browser: an
interactive tree with expand/collapse, drag, and top-down / left-right / radial layouts.
"""),
        code(PRELUDE + """
import networkx as nx
onto = json.load(open(ROOT / "ontology" / "subclass_structure.json"))
explore = json.load(open(ROOT / "ontology" / "exploration_report.json"))
report = json.load(open(ROOT / "ontology" / "ontology_report.json"))
tiers = json.load(open(ROOT / "results" / "tiers_summary.json"))["tiers"]
print("domains:", len(onto["domains"]), "| ARI ontology vs clusters:",
      report.get("cluster_agreement",{}).get("adjusted_rand_index"))
"""),
        md("## 1. The ontology as a graph\\n"
           "The ontology is now arbitrary-depth. This view shows ROOT -> domain -> first-level "
           "child; the Brain domain expands into Morphometry and Connectomics, which themselves "
           "hold deeper region/network structure (node size = number of leaf features)."),
        code("""
depth = lambda node: 0 if not node.get("children") else 1+max(depth(c) for c in node["children"])
G = nx.DiGraph(); pos = {}; sizes=[]; colors=[]; labels={}
palette = sns.color_palette("tab10", len(onto["domains"]))
G.add_node("ROOT"); pos["ROOT"]=(0,0); sizes.append(1500); colors.append("#111827"); labels["ROOT"]="ROOT"
ndom = len(onto["domains"])
for di,d in enumerate(onto["domains"]):
    dy = (di-(ndom-1)/2)*4
    G.add_node(d["id"]); G.add_edge("ROOT", d["id"]); pos[d["id"]]=(1,dy)
    sizes.append(1000); colors.append(palette[di]); labels[d["id"]]=f"{d['label']}\\n({n_leaves(d)})"
    kids=d.get("children",[]); ns=len(kids)
    for si,s in enumerate(kids):
        sid=f"{d['id']}/{s['id']}"; sy=dy+(si-(ns-1)/2)*1.1
        G.add_node(sid); G.add_edge(d["id"], sid); pos[sid]=(2, sy)
        sizes.append(160+22*n_leaves(s)); colors.append(palette[di]); labels[sid]=f"{s['label']}\\n({n_leaves(s)})"
fig, ax = plt.subplots(figsize=(14, 9))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cbd5e1")
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.9)
for n,(x,y) in pos.items():
    ax.text(x, y, labels.get(n,n), fontsize=7.5, ha="center", va="center",
            color="white" if x<2 else "#111827", fontweight="bold" if x<2 else "normal")
maxdepth = max(depth(d) for d in onto["domains"])
ax.set_title(f"Master ontology: ROOT -> domain -> first-level child (max depth {maxdepth}; node size = #leaf features)"); ax.axis("off")
plt.tight_layout(); plt.show()
"""),
        md("## 2. Feature counts and data-driven clusters"),
        code("""
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
dom_counts = {d["label"]: n_leaves(d) for d in onto["domains"]}
ax[0].bar(range(len(dom_counts)), list(dom_counts.values()),
          color=sns.color_palette("tab10", len(dom_counts)))
ax[0].set_xticks(range(len(dom_counts))); ax[0].set_xticklabels(list(dom_counts), rotation=30, ha="right", fontsize=8)
ax[0].set(title="Features per ontology domain", ylabel="#features")
csz = sorted((len(v) for v in explore["auto_clusters"].values()), reverse=True)
ax[1].bar(range(len(csz)), csz, color=IND)
ari = report.get("cluster_agreement",{}).get("adjusted_rand_index")
ax[1].set(title=f"Data-driven cluster sizes (n={len(csz)}); ontology-vs-cluster ARI={ari}",
          xlabel="cluster", ylabel="#features")
plt.tight_layout(); plt.show()
"""),
        md("The adjusted Rand index compares the LLM's semantic subdomains against purely "
           "statistical clusters. Moderate agreement is expected and healthy: constructs like the "
           "Big Five are grouped by meaning even when they are not the tightest statistical cluster, "
           "while genuine redundancies below are grouped by both."),
        md("## 3. Redundancy detected by exploration"),
        code("""
rp = pd.DataFrame(explore["redundant_pairs"])
if len(rp):
    rp["pair"] = rp["a"]+"  ~  "+rp["b"]
    rp = rp.reindex(rp["spearman_r"].abs().sort_values(ascending=False).index).head(13)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.barh(rp["pair"], rp["spearman_r"], color=[RED if v<0 else GRN for v in rp["spearman_r"]])
    ax.axvline(0,color="#333"); ax.set(title="Near-redundant feature pairs (|Spearman r| >= 0.9)", xlabel="r")
    ax.tick_params(labelsize=7); plt.tight_layout(); plt.show()
"""),
        md("## 4. Complexity tier ladder and per-tier performance"),
        code("""
td = pd.DataFrame(tiers)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(td["tier"], td["n_features"], color=IND)
ax[0].set_xticklabels(td["tier"], rotation=45, ha="right", fontsize=8)
ax[0].set(title="Features per tier", ylabel="#features")
x = np.arange(len(td)); w=0.4
ax[1].bar(x-w/2, td["pearson_r"], w, label="Pearson r", color=IND)
ax[1].bar(x+w/2, td["spearman_rho"], w, label="Spearman rho", color=GRN)
ax[1].set_xticks(x); ax[1].set_xticklabels(td["tier"], rotation=45, ha="right", fontsize=8)
ax[1].set(title="Rank recovery of intelligence per tier", ylabel="correlation"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        code("""
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].bar(x, td["mae_iq15_equivalent"], color=ORG)
ax[0].set_xticks(x); ax[0].set_xticklabels(td["tier"], rotation=45, ha="right", fontsize=8)
ax[0].set(title="MAE on 100/15-equivalent scale", ylabel="IQ-equivalent points")
ax[1].bar(x, td["rank_mae_positions"], color=IND)
ax[1].set_xticks(x); ax[1].set_xticklabels(td["tier"], rotation=45, ha="right", fontsize=8)
ax[1].set(title="Mean absolute rank error", ylabel="rank positions")
ci = np.array(td["spearman_bootstrap_ci95"].tolist(), dtype=float)
rho = td["spearman_rho"].to_numpy(float)
yerr = np.vstack([rho-ci[:,0], ci[:,1]-rho])
ax[2].bar(x, rho, yerr=yerr, color=GRN, capsize=3)
ax[2].set_xticks(x); ax[2].set_xticklabels(td["tier"], rotation=45, ha="right", fontsize=8)
ax[2].set(title="Actual vs predicted rank recovery", ylabel="Spearman rho, bootstrap 95% CI")
plt.tight_layout(); plt.show()
"""),
        md("## 5. One participant as the engine sees it"),
        code("""
import glob
tdir = sorted(glob.glob(str(ROOT/"compass_inputs"/"T6_connectome"/"eval-*")))[0]
mm = json.load(open(Path(tdir)/"multimodal_data.json"))
rows=[]
for dom, tree in mm.items():          # arbitrary-depth: collect leaves recursively per domain
    leaves=[]; mm_leaves(tree, leaves)
    for leaf in leaves:
        if leaf.get("z_score") is not None:
            rows.append({"domain":dom, "feature":leaf["feature"], "z":leaf["z_score"]})
zz = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(13,5))
palette = dict(zip(zz["domain"].unique(), sns.color_palette("tab10", zz["domain"].nunique())))
ax.bar(range(len(zz)), zz["z"], color=[palette[d] for d in zz["domain"]], width=1.0)
ax.axhline(0,color="#333"); ax.axhline(1,color="#999",ls=":"); ax.axhline(-1,color="#999",ls=":")
ax.set(title=f"Deviation profile (z) across all {len(zz)} present leaves - {Path(tdir).name}", ylabel="z-score")
ax.set_xticks([])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=d) for d,c in palette.items()], fontsize=7, ncol=4)
plt.tight_layout(); plt.show()
"""),
        md("Every tier is a filtered projection of this one ontology, so the engine always "
           "receives a clean, non-redundant hierarchy at full depth; only the set of present leaf "
           "values changes. The full-tier profile above spans 279 leaves (256 brain + 23 self-report)."),
        md("## 6. Fresh full-engine run on the upgraded high-resolution structure (2 subjects)\\n"
           "The tier ladder above is the prior 100-subject run on the earlier feature structure "
           "(kept for reference). Below is a fresh demonstration of two full-tier (T6, 279-feature) "
           "engine runs on the new high-resolution ontology, blinded to the target."),
        code("""
fresh = json.load(open(ROOT/"results"/"full_engine_2subject"/"summary.json"))
preds = fresh["predictions"]
labels = [p["participant_id"] for p in preds]
xp = np.arange(len(preds)); w=0.38
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ax[0].bar(xp-w/2, [p["ground_truth"] for p in preds], w, label="ground truth", color=IND)
ax[0].bar(xp+w/2, [p["predicted"] for p in preds], w, label="predicted", color=GRN)
ax[0].set_xticks(xp); ax[0].set_xticklabels(labels)
ax[0].set(title="Native IST: predicted vs ground truth", ylabel="IST total"); ax[0].legend()
ax[1].bar(xp-w/2, [p["true_iq_equiv"] for p in preds], w, label="true IQ-equiv", color=IND)
ax[1].bar(xp+w/2, [p["pred_iq_equiv"] for p in preds], w, label="pred IQ-equiv", color=GRN)
ax[1].axhline(100,color="#999",ls=":"); ax[1].set_xticks(xp); ax[1].set_xticklabels(labels)
ax[1].set(title=f"IQ-equivalent (rank recovered={fresh['rank_recovered']})", ylabel="IQ (100/15 scale)"); ax[1].legend()
plt.tight_layout(); plt.show()
print("Interpretation:", fresh["interpretation"])
"""),
    ]
    return nb


def main():
    for name, builder in [("01_tabular_data_exploration", nb_tabular),
                          ("02_brain_preprocessing", nb_brain),
                          ("03_ontology_and_results", nb_ontology)]:
        nb = builder()
        print(f"[notebooks] executing {name} ...")
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, HERE / f"{name}.ipynb")
        print(f"[notebooks] wrote {name}.ipynb")


if __name__ == "__main__":
    main()
