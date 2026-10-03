"""
scripts/run_full_pipeline.py

Legacy script-level wrapper for the canonical root pipeline runner.
Delegates directly to run_full_pipeline.py located at the repository root.
"""
import os
import sys

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import run_full_pipeline


def main():
    run_full_pipeline.main()


if __name__ == "__main__":
    main()
