#!/usr/bin/env python3
"""Standalone CLI of the evaluation library (no engine import needed).

    python run_evaluation.py evaluate --predictions results/ --annotations annotations.json \
        --group-by tier,predictor --out analysis/

Same arguments as ``python -m <package>.with_annotated_dataset evaluate``.
"""

from __future__ import annotations

import sys

try:
    from .core.evaluate import main
except ImportError:
    from core.evaluate import main


if __name__ == "__main__":
    sys.exit(main())
