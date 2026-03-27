"""Common utilities, types, and protocols for the BioASQ pipeline.

This package contains shared abstractions, configuration loading, typings,
and I/O utilities used across both Phase A (retrieval) and Phase B (generation).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

PROJECT_DATA_DIR = PROJECT_ROOT / "data"
PROJECT_DATA_BASELINES_DIR = PROJECT_DATA_DIR / "baselines"
PROJECT_DATA_EMBEDDINGS_DIR = PROJECT_DATA_DIR / "embeddings"
