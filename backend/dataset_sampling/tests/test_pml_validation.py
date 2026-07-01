import pytest
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "backend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.schema import DatasetConfig, validate_gprmax_pml_profile


def test_2d_dataset_config_zeros_thin_axis_pml():
    cfg = DatasetConfig(num_samples=1, pml_cells=10, dimensionality="2D")

    assert cfg.gprmax_pml_cells() == (10, 10, 0, 10, 10, 0)


def test_pml_profile_rejects_pml_that_consumes_thin_z_axis():
    with pytest.raises(ValueError, match=r"2\*pml_cells >= n_axis"):
        validate_gprmax_pml_profile(
            (10, 10, 10, 10, 10, 10),
            nx=100,
            ny=100,
            nz=1,
        )
