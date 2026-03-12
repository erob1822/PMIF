"""
pytest configuration – adds src/ to the Python path so tests can import pmif.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
