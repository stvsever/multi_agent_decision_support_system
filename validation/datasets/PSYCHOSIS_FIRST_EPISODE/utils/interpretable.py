"""Interpretable deliverables built on top of the raw EEG feature matrix:

  1. non_eeg_features.csv       identifiers + covariates + every clinical target
  2. non_eeg_feature_dictionary.csv   provenance and description of each column
  3. eeg_features_zscores.csv   age/sex control-referenced z-scores per feature
  4. eeg_zscore_reference_parameters.csv  the reference regression coefficients

The z-scores answer "how far is this person from a same-age, same-sex control?"
Each feature is regressed on centred age and sex within the control group of its
own dataset; every subject's residual is divided by the control residual SD.
That keeps the two acquisitions on their own reference while making the numbers
directly interpretable as control-referenced standard deviations.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RAW_ROOT, RESULTS_ROOT, SUBJECT_ROOT
from .io import discover_records

DATASETS = ("ds003944", "ds003947")
PHENOTYPE_TABLES = ("bprs", "gafgas", "hollingshead", "matrics", "medication",
                    "sans", "saps", "sfs", "wasi")
IDENTIFIER_COLUMNS = ("recording_id", "dataset_id", "participant_id")

COVARIATE_NAMES = {
    "age": "covariate__demographics__age_years",
    "gender": "covariate__demographics__sex_source_reported",
    "race": "covariate__demographics__race_source_reported",
    "ethnicity": "covariate__demographics__ethnicity_source_reported",
    "SUBSES": "covariate__socioeconomic__participant_hollingshead_index",
    "MOMSES": "covariate__socioeconomic__mother_hollingshead_index",
    "DADSES": "covariate__socioeconomic__father_hollingshead_index",
    "PSES": "covariate__socioeconomic__parental_mean_hollingshead_index",
    "CPZ_at_scan": "covariate__medication__chlorpromazine_equivalent_at_eeg_scan",
}

TARGET_SPECIAL_NAMES = {
    "type": "target__psychosis__case_control_label",
    "BPRST19": "target__bprs__total_q01_to_q19",
    "BPRST18": "target__bprs__total_q01_to_q18",
    "bprs_motor_hyper": "target__bprs__q20_motor_hyperactivity",
    "ROLECURR": "target__functioning__gaf_role_current_0_to_10_higher_better",
    "ROLELOW": "target__functioning__gaf_role_lifetime_low_0_to_10_higher_better",
    "ROLEHIGH": "target__functioning__gaf_role_lifetime_high_0_to_10_higher_better",
    "SOCIALCURR": "target__functioning__gaf_social_current_0_to_10_higher_better",
    "SOCIALLOW": "target__functioning__gaf_social_lifetime_low_0_to_10_higher_better",
    "SOCIALHIGH": "target__functioning__gaf_social_lifetime_high_0_to_10_higher_better",
    "GAS": "target__functioning__global_assessment_scale_0_to_100_higher_better",
    "VOCAB_RS": "target__wasi__vocabulary_raw_score",
    "VOCAB_TS": "target__wasi__vocabulary_t_score",
    "MATRIX_RS": "target__wasi__matrix_reasoning_raw_score",
    "MATRIX_TS": "target__wasi__matrix_reasoning_t_score",
    "FULL2TS": "target__wasi__two_subtest_combined_t_score",
    "FULL2IQ": "target__wasi__two_subtest_full_iq_estimate",
    "FULL2PCT": "target__wasi__two_subtest_combined_percentile",
}

OPEN_TEXT_SFS_FIELDS = {"SFS_Q1_LIVEWHERE", "SFS_Q2_LIVEWHO", "SFS_Q3A_WKDAYGMT",
                        "SFS_Q3B_WKENDGMT", "SFS_Q15A", "SFS_Q15B", "SFS_Q15C"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(text: str, maximum: int = 92) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    normalized = normalized.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized[:maximum].rstrip("_")


def _description_tail(description: str) -> str:
    text = re.sub(r"^.*?\s+-\s+", "", description)
    text = re.sub(r"^Q\d+\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*section\)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _target_name(table: str, source_column: str, metadata: dict[str, Any]) -> str:
    if source_column in TARGET_SPECIAL_NAMES:
        return TARGET_SPECIAL_NAMES[source_column]
    description = str(metadata.get("Description", source_column))
    if table in {"bprs", "sans", "saps"}:
        match = re.search(r"Q(\d+)\s+(.+)$", description)
        if match:
            q = int(match.group(1))
            concept = _description_tail(f"Q{q} {match.group(2)}")
            return f"target__{table}__q{q:02d}_{_slug(concept, 72)}"
    if table == "sfs":
        original = _slug(source_column, 30)
        if ":" in description:
            concept = re.sub(r"^\s*\(\d+\)\s*", "", description.split(":", 1)[1])
        else:
            concept = description
        return f"target__social_functioning_scale__{original}_{_slug(concept, 68)}"
    if table in {"matrics", "wasi"}:
        return f"target__{table}__{_slug(source_column, 24)}_{_slug(description, 72)}"
    return f"target__{table}__{_slug(source_column, 28)}_{_slug(description, 72)}"


def _source_paths(dataset_id: str, table: str) -> tuple[Path, Path]:
    base = (RAW_ROOT / dataset_id / "participants" if table == "participants"
            else RAW_ROOT / dataset_id / "phenotype" / table)
    return base.with_suffix(".tsv"), base.with_suffix(".json")


def _dictionary(dataset_id: str, table: str) -> dict[str, dict[str, Any]]:
    return json.loads(_source_paths(dataset_id, table)[1].read_text(encoding="utf-8"))


def _rename_map(table: str, dictionary: dict[str, dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source_column, metadata in dictionary.items():
        if source_column == "participant_id":
            continue
        if source_column in COVARIATE_NAMES:
            mapping[source_column] = COVARIATE_NAMES[source_column]
        elif table == "participants" and source_column == "type":
            mapping[source_column] = TARGET_SPECIAL_NAMES[source_column]
        else:
            mapping[source_column] = _target_name(table, source_column, metadata)
    return mapping


def _identity_frame() -> pd.DataFrame:
    rows = [{"recording_id": r["recording_id"], "dataset_id": r["dataset_id"],
             "participant_id": r["participant_id"]} for r in discover_records()]
    return pd.DataFrame(rows)


def _load_dataset(dataset_id: str, identity: pd.DataFrame):
    source = identity.loc[identity["dataset_id"] == dataset_id,
                          list(IDENTIFIER_COLUMNS)].copy()
    dictionary_rows: list[dict[str, Any]] = []
    assembled = source
    for table in ("participants", *PHENOTYPE_TABLES):
        tsv_path, json_path = _source_paths(dataset_id, table)
        frame = pd.read_csv(tsv_path, sep="\t", na_values=["n/a", "N/A", "NA"])
        dictionary = _dictionary(dataset_id, table)
        mapping = _rename_map(table, _dictionary(DATASETS[0], table))
        assert set(frame.columns) == {"participant_id", *mapping.keys()}
        assembled = assembled.merge(frame.rename(columns=mapping), on="participant_id",
                                    how="left", validate="one_to_one")
        for source_column, output_column in mapping.items():
            metadata = dictionary[source_column]
            dictionary_rows.append({
                "column_name": output_column, "dataset_id": dataset_id,
                "role": "covariate" if output_column.startswith("covariate__") else "target",
                "domain": output_column.split("__")[1], "source_table": table,
                "source_column": source_column,
                "source_tsv": str(tsv_path.relative_to(RAW_ROOT.parents[1])),
                "source_json": str(json_path.relative_to(RAW_ROOT.parents[1])),
                "description": metadata.get("Description", ""),
                "units": metadata.get("Units", ""),
                "levels_json": json.dumps(metadata.get("Levels", {}), sort_keys=True),
                "sensitivity": ("potentially_identifying_open_text"
                                if table == "sfs" and source_column in OPEN_TEXT_SFS_FIELDS
                                else "standard_research_phenotype")})
    assembled["target__psychosis__case_control_binary"] = assembled[
        "target__psychosis__case_control_label"].map({"Control": 0, "Psychosis": 1}).astype("Int64")
    dictionary_rows.append({
        "column_name": "target__psychosis__case_control_binary", "dataset_id": dataset_id,
        "role": "target", "domain": "psychosis", "source_table": "participants",
        "source_column": "type", "source_tsv": "", "source_json": "",
        "description": "Binary case-control label: Control=0, Psychosis=1",
        "units": "binary", "levels_json": json.dumps({"0": "Control", "1": "First Episode Psychosis"}),
        "sensitivity": "standard_research_phenotype"})
    return assembled, dictionary_rows


def build_non_eeg_table():
    identity = _identity_frame()
    frames, dict_records = [], []
    for dataset_id in DATASETS:
        frame, records = _load_dataset(dataset_id, identity)
        frames.append(frame)
        dict_records.extend(records)
    first = frames[0].columns.tolist()
    assert all(f.columns.tolist() == first for f in frames[1:])
    table = pd.concat(frames, ignore_index=True)
    assert len(table) == 143 and table["recording_id"].is_unique

    covariates = [c for c in table.columns if c.startswith("covariate__")]
    targets = [c for c in table.columns if c.startswith("target__")]
    head = ["target__psychosis__case_control_label", "target__psychosis__case_control_binary"]
    targets = head + [t for t in targets if t not in head]
    table = table[[*IDENTIFIER_COLUMNS, *covariates, *targets]]

    dictionary = pd.DataFrame(dict_records)
    combined: list[dict[str, Any]] = []
    for column_name, records in dictionary.groupby("column_name", observed=True, sort=False):
        row = records.iloc[0].to_dict()
        row["dataset_coverage"] = "|".join(sorted(records["dataset_id"].unique()))
        row["source_tsv"] = "|".join(sorted(set(records["source_tsv"]) - {""}))
        row["source_json"] = "|".join(sorted(set(records["source_json"]) - {""}))
        row.pop("dataset_id", None)
        combined.append(row)
    dictionary = pd.DataFrame(combined)
    identity_desc = {
        "recording_id": "Dataset-prefixed recording id joining EEG and non-EEG tables",
        "dataset_id": "OpenNeuro dataset identifier",
        "participant_id": "Participant id within the source OpenNeuro dataset"}
    for name in IDENTIFIER_COLUMNS:
        dictionary = pd.concat([dictionary, pd.DataFrame([{
            "column_name": name, "dataset_coverage": "ds003944|ds003947",
            "role": "identifier", "domain": "identifier", "source_table": "participants",
            "source_column": name, "source_tsv": "", "source_json": "",
            "description": identity_desc[name], "units": "", "levels_json": "{}",
            "sensitivity": "research_identifier"}])], ignore_index=True)
    dictionary = (dictionary.set_index("column_name").reindex(table.columns)
                  .rename_axis("column_name").reset_index())
    dictionary["pandas_dtype"] = [str(table[c].dtype) for c in table.columns]
    dictionary["non_missing_count"] = [int(table[c].notna().sum()) for c in table.columns]
    dictionary["missing_count"] = [int(table[c].isna().sum()) for c in table.columns]
    dictionary["unique_non_missing"] = [int(table[c].nunique(dropna=True)) for c in table.columns]
    assert dictionary["column_name"].tolist() == table.columns.tolist()
    return table, dictionary


def _eligibility() -> pd.DataFrame:
    rows = []
    for record in discover_records():
        qc_path = SUBJECT_ROOT / record["recording_id"] / "preprocessing_qc.json"
        elig = False
        if qc_path.exists():
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            elig = qc.get("status") == "processed" and bool(qc.get("feature_eligible", False))
        rows.append({"recording_id": record["recording_id"], "feature_eligible": elig})
    return pd.DataFrame(rows)


def build_control_zscores(non_eeg: pd.DataFrame):
    eeg = pd.read_csv(RESULTS_ROOT / "eeg_features.csv")
    feature_names = eeg.columns.tolist()[1:]
    meta = non_eeg[["recording_id", "dataset_id",
                    "covariate__demographics__age_years",
                    "covariate__demographics__sex_source_reported",
                    "target__psychosis__case_control_label"]].merge(
        _eligibility(), on="recording_id", how="left", validate="one_to_one")
    analysis = eeg.merge(meta, on="recording_id", how="left", validate="one_to_one")

    z = pd.DataFrame(index=analysis.index, columns=feature_names, dtype=float)
    params: list[dict[str, Any]] = []
    for dataset_id in DATASETS:
        ds = analysis["dataset_id"].eq(dataset_id)
        ref_mask = (ds & analysis["target__psychosis__case_control_label"].eq("Control")
                    & analysis["feature_eligible"].astype(bool)
                    & analysis["covariate__demographics__age_years"].notna()
                    & analysis["covariate__demographics__sex_source_reported"].isin(["M", "F"]))
        ref = analysis.loc[ref_mask]
        age_center = float(ref["covariate__demographics__age_years"].mean())
        x_ref = np.column_stack([
            np.ones(len(ref)),
            ref["covariate__demographics__age_years"].to_numpy(float) - age_center,
            ref["covariate__demographics__sex_source_reported"].eq("F").to_numpy(float)])
        # Need a stable control reference; skip a dataset that has too few eligible
        # controls (only happens on partial/interim data, never on the full cohort).
        if len(ref) < 15 or np.linalg.matrix_rank(x_ref) != 3:
            continue

        score_mask = (ds & analysis["covariate__demographics__age_years"].notna()
                      & analysis["covariate__demographics__sex_source_reported"].isin(["M", "F"]))
        score = analysis.loc[score_mask]
        x_score = np.column_stack([
            np.ones(len(score)),
            score["covariate__demographics__age_years"].to_numpy(float) - age_center,
            score["covariate__demographics__sex_source_reported"].eq("F").to_numpy(float)])
        for name in feature_names:
            y = ref[name].to_numpy(float)
            ok = np.isfinite(y)
            if ok.sum() < 10:
                continue
            beta, _, rank, _ = np.linalg.lstsq(x_ref[ok], y[ok], rcond=None)
            resid = y[ok] - x_ref[ok] @ beta
            dof = ok.sum() - int(rank)
            sd = float(np.sqrt(np.sum(resid**2) / dof)) if dof > 0 else np.nan
            if not (np.isfinite(sd) and sd > 0):
                continue
            ys = score[name].to_numpy(float)
            std = (ys - x_score @ beta) / sd
            std[~np.isfinite(ys)] = np.nan
            z.loc[score.index, name] = std
            params.append({"dataset_id": dataset_id, "feature": name,
                           "reference_n": int(ok.sum()), "age_center_years": age_center,
                           "intercept_male_at_mean_age": float(beta[0]),
                           "age_slope_per_year": float(beta[1]),
                           "female_minus_male": float(beta[2]),
                           "reference_residual_sd": sd})
    z.columns = [f"Z_control__{n}" for n in feature_names]
    z_table = pd.concat([analysis[["recording_id"]].reset_index(drop=True),
                         z.reset_index(drop=True)], axis=1)
    return z_table, pd.DataFrame(params)


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    non_eeg, dictionary = build_non_eeg_table()
    non_eeg.to_csv(RESULTS_ROOT / "non_eeg_features.csv", index=False, na_rep="")
    dictionary.to_csv(RESULTS_ROOT / "non_eeg_feature_dictionary.csv", index=False, na_rep="")

    z_table, params = build_control_zscores(non_eeg)
    z_table.to_csv(RESULTS_ROOT / "eeg_features_zscores.csv", index=False, na_rep="")
    params.to_csv(RESULTS_ROOT / "eeg_zscore_reference_parameters.csv", index=False, na_rep="")

    covariates = [c for c in non_eeg.columns if c.startswith("covariate__")]
    targets = [c for c in non_eeg.columns if c.startswith("target__")]
    summary = {
        "recording_count": int(len(non_eeg)),
        "covariate_columns": len(covariates), "target_columns": len(targets),
        "non_eeg_columns": int(non_eeg.shape[1]),
        "eeg_zscore_columns": int(z_table.shape[1] - 1),
        "reference": "dataset-specific control group, centred age + sex, control residual SD",
        "outputs": {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
                    for p in [RESULTS_ROOT / "non_eeg_features.csv",
                              RESULTS_ROOT / "non_eeg_feature_dictionary.csv",
                              RESULTS_ROOT / "eeg_features_zscores.csv",
                              RESULTS_ROOT / "eeg_zscore_reference_parameters.csv"]},
    }
    (RESULTS_ROOT / "interpretable_tables_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
