#!/usr/bin/env python3
"""
Step 01 - bulk download real lesion-mask content for every subject.

The dataset ships as broken git-annex symlinks (content was never fetched).
This replaces each ``sub-XXX_lesion-mask.nii.gz`` with the real NIfTI content
from OpenNeuro's public S3 mirror. Idempotent: already-real files are skipped.
"""

import _bootstrap  # noqa: F401

import config
from validation.datasets.NUMERACY_STROKE.utils import lesion


def main() -> None:
    subjects = config.all_subjects()
    print(f"[01] Downloading lesion masks for {len(subjects)} subjects from "
          f"OpenNeuro/{config.ACCESSION} ...")

    results = lesion.download_many(subjects, config.ACCESSION, config.LESION_MASKS_DIR, workers=8)

    ok = [s for s, r in results.items() if r.get("lesion_mask")]
    failed = [s for s, r in results.items() if not r.get("lesion_mask")]

    print(f"[01] Downloaded/verified {len(ok)}/{len(subjects)} lesion masks.")
    if failed:
        print(f"[01] FAILED for {len(failed)} subjects: {sorted(failed)}")
        raise SystemExit(f"[01] {len(failed)} subjects missing real lesion-mask content.")


if __name__ == "__main__":
    main()
