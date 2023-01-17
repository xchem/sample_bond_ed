import dataclasses
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, NewType

import fire
import gemmi
import numpy as np
import pandas as pd
import pydantic
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

# from . import __version__

__all__ = ["main"]

# Bond = NewType("Bond")
Plot = NewType("Plot", Figure)
Samples = list[float]
Position = NewType("Position", gemmi.Position)
Positions = NewType("Positions", list[Position])
Xmap = NewType("Xmap", gemmi.FloatGrid)
Structure = NewType("Structure", gemmi.Structure)
AtomID = NewType("AtomID", str)

PLOT_X = "Distance Along Bond"
PLOT_Y = "Electron Density"


@dataclasses.dataclass()
class Bond:
    atom_1: Position
    atom_2: Position


def sample_at_positions(xmap: Xmap, positions: Positions) -> Samples:
    return [xmap.tricubic_interpolation(pos) for pos in positions]


def get_sample_posisitons(bond: Bond, rate: int) -> Positions:
    pos_1 = bond.atom_1
    pos_2 = bond.atom_2
    sample_point_array = np.linspace(
        np.array(
            [pos_1.x, pos_1.y, pos_1.z],
        ),
        np.array([pos_2.x, pos_2.y, pos_2.z]),
        num=rate,
    )

    sample_points: Positions = [
        gemmi.Position(
            sample_point_array[j, 0],
            sample_point_array[j, 1],
            sample_point_array[j, 2],
        )
        for j in range(len(sample_point_array))
    ]

    return sample_points


def sample_along_bond(
    xmap: Xmap,
    bond: Bond,
    rate: int = 10,
) -> Samples:

    positions: Positions = get_sample_posisitons(bond, rate)
    samples: Samples = sample_at_positions(xmap, positions)

    return samples


def make_sample_dataframe(samples: Samples):
    records = []
    num_samples = len(samples)
    percents = np.linspace(0, 100, num=num_samples)
    for percent, sample in zip(percents, samples):
        records.append(
            {
                PLOT_X: f"{round(percent)}%",
                PLOT_Y: round(sample, 2),
            },
        )

    return pd.DataFrame(records)


def make_sample_plot(samples: Samples) -> Plot:
    df = make_sample_dataframe(samples)

    fig, ax = plt.subplots()

    sns.lineplot(data=df, x=PLOT_X, y=PLOT_Y, ax=ax)

    return fig


def save_plot(plot: Plot, path: Path):
    plot.savefig(str(path))


def read_structure(path: Path) -> Structure:
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    return structure


def read_map(path: Path) -> Xmap:
    return gemmi.read_ccp4_map(str(path), setup=True).grid


def pos_from_atom_id(atom_id: AtomID, structure: Structure) -> Position:
    # sel = gemmi.Selection(atom_id)

    # selected_tuple = sel.first(structure)
    # logger.debug(f"Selected tuple: {selected_tuple}")

    # atom = selected_tuple[1].atom

    chain_sel, residue_sel, atom_sel = atom_id.split("/")
    logger.debug(f"{chain_sel} : {residue_sel} : {atom_sel}")

    selected_atom = None
    for model in structure:
        for chain in model:
            if not chain.name == chain_sel:
                continue
            for residue in chain:
                if not residue.seqid.num == int(residue_sel):
                    continue
                for atom in residue:
                    # print(atom.name)
                    if atom.name == atom_sel:
                        selected_atom = atom

    if not selected_atom:
        raise Exception(f"No atom matching selection code: {atom_id}")

    return selected_atom.pos


def sample_bond_ed(
    structure_path: Path,
    xmap_path: Path,
    output_dir: Path,
    atom_1_id: AtomID,
    atom_2_id: AtomID,
):
    logger.info(f"Reading structure at path: {structure_path}")
    structure: Structure = read_structure(structure_path)
    logger.debug(f"Structure: {structure}")

    logger.info(f"Reading structure at path: {xmap_path}")
    xmap: Xmap = read_map(xmap_path)
    logger.debug(f"Xmap: {xmap}")

    bond: Bond = Bond(
        atom_1=pos_from_atom_id(atom_1_id, structure),
        atom_2=pos_from_atom_id(atom_2_id, structure),
    )

    logger.info("Determining samples...")
    samples: Samples = sample_along_bond(
        xmap,
        bond,
    )

    logger.info("Making plot...")
    plot: Plot = make_sample_plot(samples)

    logger.info(f"Saving plot in dir: {output_dir}")
    save_plot(plot, output_dir / "bond_ed_sampling.png")


# test with: python -m sample_bond_ed
if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--structure_path", type=Path)
    parser.add_argument("--xmap_path", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--atom_1_id",
        help="""
        A selector for a single atom of the form CHAIN/RESIDUE/NAME, 
        such as \"C/LIG/C1\"
        """,
    )
    parser.add_argument(
        "--atom_2_id",
        help="""
        A selector for a single atom of the form CHAIN/RESIDUE/NAME, 
        such as \"C/LIG/C1\"
        """,
    )
    args = parser.parse_args()

    structure_path: Path = args.structure_path
    xmap_path: Path = args.xmap_path
    output_dir: Path = args.output_dir
    atom_1_id: AtomID = args.atom_1_id
    atom_2_id: AtomID = args.atom_2_id

    logger.info(f"Structure path: {structure_path}")
    logger.info(f"Atom 1 ID: {atom_1_id}")
    logger.info(f"Atom 2 ID: {atom_2_id}")

    sample_bond_ed(
        structure_path,
        xmap_path,
        output_dir,
        atom_1_id,
        atom_2_id,
    )
