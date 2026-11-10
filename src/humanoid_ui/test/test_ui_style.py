# Copyright 2026 Bjarne
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Check the UI package's existing ROS Python lint conventions."""

from pathlib import Path

from ament_copyright.main import main as copyright_main
from ament_flake8.main import main_with_errors
from ament_pep257.main import main as pep257_main


ROOT = Path(__file__).resolve().parents[1]


def test_flake8():
    """Check Python formatting."""
    rc, errors = main_with_errors(argv=[str(ROOT)])
    assert rc == 0, '\n'.join(errors)


def test_docstrings():
    """Check Google-style docstrings."""
    assert pep257_main(argv=['--convention', 'google', '--', str(ROOT)]) == 0


def test_copyright():
    """Check copyright and license headers."""
    assert copyright_main(argv=[str(ROOT)]) == 0
