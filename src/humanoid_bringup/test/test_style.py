"""Check Python formatting and Google-style docstrings."""

from pathlib import Path

from ament_flake8.main import main_with_errors
from ament_pep257.main import main
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """Check formatting of the package sources."""
    root = Path(__file__).resolve().parents[1]
    rc, errors = main_with_errors(argv=[str(root)])
    assert rc == 0, '\n'.join(errors)


@pytest.mark.pep257
@pytest.mark.linter
def test_pep257():
    """Check Google-style docstrings throughout the package."""
    root = Path(__file__).resolve().parents[1]
    assert main(argv=['--convention', 'google', '--', str(root)]) == 0
