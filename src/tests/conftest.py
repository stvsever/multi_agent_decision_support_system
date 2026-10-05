"""Shared test setup: the suite never calls a provider, so it runs without real credentials.

A fresh clone has no .env; several tests build an OpenRouter client or a dashboard run and only
need a key to be present. A placeholder key is set when none is configured (a real key in the
environment or .env is left alone, and no test sends a request).
"""

import os

os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test-placeholder")
