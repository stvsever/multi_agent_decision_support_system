"""QC record schema (spec section 9) - kept structurally separate from feature values."""

from dataclasses import dataclass, field, fields

QC_FIELDS = [
    "dataset_id",
    "source_participant_id_private",
    "recording_duration_s",
    "usable_duration_s",
    "usable_fraction",
    "n_eeg_channels_original",
    "n_bad_channels",
    "bad_channel_fraction",
    "n_interpolated_channels",
    "n_epochs_total",
    "n_epochs_retained",
    "bad_epoch_fraction",
    "n_ica_components_fit",
    "n_ica_components_removed_eog",
    "n_ica_components_removed_ecg",
    "n_ica_components_removed_other",
    "residual_line_noise_ratio",
    "high_frequency_muscle_ratio",
    "median_peak_to_peak_uv",
    "aperiodic_fit_r2_median",
    "aperiodic_fit_error_median",
    "alpha_peak_detected_scope_count",
    "microstate_global_explained_variance",
    "connectivity_node_coverage_count",
    "exclusion_status",
    "exclusion_reason",
]


@dataclass
class SubjectQC:
    """One subject's QC record. Unset fields default to None (missing, not zero)."""

    dataset_id: str
    source_participant_id_private: str
    recording_duration_s: float | None = None
    usable_duration_s: float | None = None
    usable_fraction: float | None = None
    n_eeg_channels_original: int | None = None
    n_bad_channels: int | None = None
    bad_channel_fraction: float | None = None
    n_interpolated_channels: int | None = None
    n_epochs_total: int | None = None
    n_epochs_retained: int | None = None
    bad_epoch_fraction: float | None = None
    n_ica_components_fit: int | None = None
    n_ica_components_removed_eog: int | None = None
    n_ica_components_removed_ecg: int | None = None
    n_ica_components_removed_other: int | None = None
    residual_line_noise_ratio: float | None = None
    high_frequency_muscle_ratio: float | None = None
    median_peak_to_peak_uv: float | None = None
    aperiodic_fit_r2_median: float | None = None
    aperiodic_fit_error_median: float | None = None
    alpha_peak_detected_scope_count: int | None = None
    microstate_global_explained_variance: float | None = None
    connectivity_node_coverage_count: int | None = None
    exclusion_status: str = "included"  # "included" | "excluded"
    exclusion_reason: str | None = None
    extra: dict = field(default_factory=dict)  # e.g. group_f_status, entropy epoch cap used

    def as_row(self) -> dict:
        row = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        row.update(self.extra)
        return row


# Fixed inclusion gates (spec section 7 "Stage 7: Final reference, interpolation, and epoching").
MIN_USABLE_DURATION_S = 120.0
MIN_RETAINED_EPOCHS = 30
MIN_CHANNEL_RETENTION_FRACTION = 0.80
MIN_GOOD_CHANNELS_PER_NODE = 2
MAX_BAD_EPOCH_FRACTION = 0.60
MAX_INTERPOLATED_CHANNEL_FRACTION = 0.20


def evaluate_inclusion_gates(qc: SubjectQC) -> tuple[bool, str | None]:
    """Apply the six fixed subject-inclusion gates. Returns (included, reason_if_excluded)."""
    if qc.usable_duration_s is not None and qc.usable_duration_s < MIN_USABLE_DURATION_S:
        return False, f"usable_duration_s {qc.usable_duration_s:.1f} < {MIN_USABLE_DURATION_S}"
    if qc.n_epochs_retained is not None and qc.n_epochs_retained < MIN_RETAINED_EPOCHS:
        return False, f"n_epochs_retained {qc.n_epochs_retained} < {MIN_RETAINED_EPOCHS}"
    if (
        qc.n_eeg_channels_original
        and qc.n_bad_channels is not None
        and (qc.n_eeg_channels_original - qc.n_bad_channels) / qc.n_eeg_channels_original
        < MIN_CHANNEL_RETENTION_FRACTION
    ):
        return False, "channel retention below 80%"
    if qc.bad_epoch_fraction is not None and qc.bad_epoch_fraction > MAX_BAD_EPOCH_FRACTION:
        return False, f"bad_epoch_fraction {qc.bad_epoch_fraction:.2f} > {MAX_BAD_EPOCH_FRACTION}"
    if (
        qc.n_interpolated_channels is not None
        and qc.n_eeg_channels_original
        and qc.n_interpolated_channels / qc.n_eeg_channels_original
        > MAX_INTERPOLATED_CHANNEL_FRACTION
    ):
        return False, "interpolated channel fraction above 20%"
    return True, None
