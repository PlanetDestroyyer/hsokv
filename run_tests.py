#!/usr/bin/env python3
"""
Simple test runner for HSOKV
No pytest required - just run: python run_tests.py

This will test all key features including:
- Catastrophic forgetting prevention
- Dual memory system (STM + LTM)
- 3-stage lifecycle
- GPU acceleration (if available)
- CLIP embedder
"""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent / 'tests'))

from test_hsokv_comprehensive import run_all_tests

if __name__ == "__main__":
    print("Starting HSOKV Test Suite...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
