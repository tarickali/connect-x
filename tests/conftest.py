import pytest

from connectx.config import make_config
from connectx.types import Config


@pytest.fixture
def default_config() -> Config:
    """Classic Connect 4: 6 rows, 7 cols, k=4, two players."""
    return make_config((6, 7), 4, [1, 2])


@pytest.fixture
def small_config() -> Config:
    """Small grid for fast tests."""
    return make_config((4, 5), 3, [1, 2])


@pytest.fixture
def multiplayer_config() -> Config:
    """Three seats, to exercise the non-alternating paths."""
    return make_config((5, 6), 3, [1, 2, 3])


@pytest.fixture
def config_via_helper() -> Config:
    """Config built with make_config (standardized construction)."""
    return make_config((6, 7), 4, [1, 2])
