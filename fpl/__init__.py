"""Fantasy Premier League expected-points prediction package.

Self-contained: nothing in here imports from the NBA pipeline (``scripts/``,
``rating_engine/``, ``api/``) and nothing there imports from here.
"""

__all__ = ["config", "scoring"]
