import subprocess
import sys

from sample_bond_ed import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "sample_bond_ed", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
