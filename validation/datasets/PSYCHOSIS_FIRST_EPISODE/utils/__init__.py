"""Resting-state EEG feature-extraction pipeline for the first-episode-psychosis
validation dataset (OpenNeuro ds003944 + ds003947).

Importable modules:
  config        frozen settings and shared paths
  montage       the shared 49-channel montage, regions, canonical microstate order
  io            dataset discovery and direct BrainVision loading
  preprocess    cleaning and harmonization to the shared montage
  features      the eight canonical feature families (A-H)
  interpretable non-EEG phenotype tables and control-referenced z-scores
  viz           cohort figures and the representative-subject dashboard
"""

from . import config, montage  # noqa: F401

__all__ = ["config", "montage", "io", "preprocess", "features", "interpretable", "viz"]
