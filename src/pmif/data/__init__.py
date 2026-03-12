"""
data/__init__.py – exposes data utilities at the pmif.data namespace.
"""

from pmif.data.generator import SyntheticDataGenerator
from pmif.data.loader import DataLoader

__all__ = ["SyntheticDataGenerator", "DataLoader"]
