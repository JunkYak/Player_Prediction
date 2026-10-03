"""
orchestrator.py

Root entry point for the NBA Player Prediction Daily Pipeline Orchestrator.
Delegates to scripts/orchestrator.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.orchestrator import main

if __name__ == "__main__":
    main()
