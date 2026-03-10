import pytest

from connectx.types import Config
from connectx.utils import make_config


@pytest.fixture
def default_config() -> Config:
    """Classic Connect 4: 6 rows, 7 cols, k=4, two players."""
    return {"shape": (6, 7), "k": 4, "players": [1, 2]}


@pytest.fixture
def small_config() -> Config:
    """Small grid for fast tests."""
    return {"shape": (4, 5), "k": 3, "players": [1, 2]}


@pytest.fixture
def config_via_helper() -> Config:
    """Config built with make_config (standardized construction)."""
    return make_config((6, 7), 4, [1, 2])
