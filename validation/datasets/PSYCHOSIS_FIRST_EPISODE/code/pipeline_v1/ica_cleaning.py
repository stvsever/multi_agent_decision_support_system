"""Stage 6: ICA-assisted ocular/cardiac artifact removal (extended Infomax)."""

import mne

RANDOM_STATE = 97


def fit_and_clean(raw_main: mne.io.Raw, raw_ica_fit: mne.io.Raw, bad_channels: list[str]) -> tuple[mne.io.Raw, dict]:
    """Fit ICA on the 1 Hz branch (excluding bad channels), apply to the main branch."""
    for branch in (raw_main, raw_ica_fit):
        present_bads = [ch for ch in bad_channels if ch in branch.ch_names]
        branch.info["bads"] = present_bads

    good_eeg_picks = mne.pick_types(raw_ica_fit.info, eeg=True, exclude="bads")
    good_eeg_only = raw_ica_fit.copy().pick(picks=good_eeg_picks)
    rank = mne.compute_rank(good_eeg_only, rank="info")
    n_components = min(rank.get("eeg", len(good_eeg_picks) - 1), len(good_eeg_picks) - 1)
    n_components = max(n_components, 2)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=RANDOM_STATE,
        max_iter="auto",
    )
    ica.fit(raw_ica_fit, picks=good_eeg_picks, verbose=False)

    exclude = set()
    n_eog = n_ecg = 0
    if "VEOG" in raw_ica_fit.ch_names:
        eog_idx, _ = ica.find_bads_eog(raw_ica_fit, ch_name="VEOG", verbose=False)
        exclude.update(eog_idx)
        n_eog = len(eog_idx)
    if "ECG" in raw_ica_fit.ch_names:
        ecg_idx, _ = ica.find_bads_ecg(raw_ica_fit, ch_name="ECG", verbose=False)
        exclude.update(ecg_idx)
        n_ecg = len(ecg_idx)
    ica.exclude = sorted(exclude)

    raw_clean = raw_main.copy()
    ica.apply(raw_clean, verbose=False)

    qc = {
        "n_ica_components_fit": n_components,
        "n_ica_components_removed_eog": n_eog,
        "n_ica_components_removed_ecg": n_ecg,
        "n_ica_components_removed_other": len(exclude) - n_eog - n_ecg,
    }
    return raw_clean, qc
