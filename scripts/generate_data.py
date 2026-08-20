#!/usr/bin/env python3
"""CLI: generate all synthetic raw datasets + ground-truth labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generation.generate_dataset import main

if __name__ == "__main__":
    main()
