"""``python -m <package>.with_annotated_dataset evaluate --predictions ... --annotations ... --out DIR``."""

import sys

from .core.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
