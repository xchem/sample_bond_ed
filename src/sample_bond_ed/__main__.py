from argparse import ArgumentParser

import gemmi

from . import __version__

__all__ = ["main"]


def sample_at_positions(positions: list[gemmi.Position]):
    ...


def sample_along_bond(
    xmap,
    bond,
    rate: int = 10,
) -> list[float]:

    ...


def main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(args)


# test with: python -m sample_bond_ed
if __name__ == "__main__":
    main()
