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


def sample_at_positions(xmap: Xmap, positions: list[Positions]) -> Samples:
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


def get_sample_centres(bond: Bond, num_sample_along_bond: int):
    pos_1 = bond.atom_1
    pos_2 = bond.atom_2
    sample_pos_array = np.linspace(
        np.array(
            [pos_1.x, pos_1.y, pos_1.z],
        ),
        np.array([pos_2.x, pos_2.y, pos_2.z]),
        num=num_sample_along_bond,
    )
    return [gemmi.Position(*sample_pos) for sample_pos in sample_pos_array]


def get_sample_positions_near_sample_centre(
    sample_centre: Position, num_samples_around_bond: int, radius: float
):
    from numpy.random import default_rng

    rng = default_rng()
    sample_centre_array = np.array(
        [
            sample_centre.x,
            sample_centre.y,
            sample_centre.z,
        ]
    )
    initial_sample_array = rng.uniform(
        low=sample_centre_array - radius,
        high=sample_centre_array + radius,
        size=(num_samples_around_bond, 3),
    )
    distances = np.linalg.norm(
        initial_sample_array - sample_centre_array, axis=1
    )
    sample_array = initial_sample_array[distances < radius]

    return [gemmi.Position(*sample) for sample in sample_array]

def get_sample_positions_near_samples(
    sample_centres: list[Position], num_samples_around_bond: int, radius: float
):
    from numpy.random import default_rng

    rng = default_rng()
    sample_centres_array = np.array(
        [
            [
                sample_centre.x,
                sample_centre.y,
                sample_centre.z,
            ]
            for sample_centre
            in sample_centres
        ]
    )
    min_coord = np.min(sample_centres_array, axis=0)
    max_coord = np.max(sample_centres_array, axis=0)

    initial_sample_array = rng.uniform(
        low=min_coord - radius,
        high=max_coord + radius,
        size=(num_samples_around_bond, 3),
    )
    min_distances = []
    for coord in initial_sample_array:
        distances = np.linalg.norm(
            coord.reshape(-1,3) - sample_centres_array.reshape(-1,3), axis=1
        )
        min_dist = np.min(distances)
        min_distances.append(min_dist)
    sample_array = initial_sample_array[np.array(min_distances) < radius]

    return [gemmi.Position(*sample) for sample in sample_array]


def sample_along_bond_radius(
    xmap,
    bond: Bond,
    num_sample_along_bond=10,
    num_samples_around_bond=100,
    radius=0.5,
):
    samples_along_bond: Samples = []
    for sample_centre in get_sample_centres(bond, num_sample_along_bond):
        sample_positions = get_sample_positions_near_sample_centre(
            sample_centre, num_samples_around_bond, radius
        )
        samples: Samples = sample_at_positions(xmap, sample_positions)
        mean_sample = float(np.mean(samples))
        samples_along_bond.append(mean_sample)

    return samples_along_bond


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


def make_sample_dataframe_compare(samples_1: Samples, samples_2: Samples):
    records = []
    num_samples = len(samples_1)
    percents = np.linspace(0, 100, num=num_samples)
    for percent, sample_1 in zip(percents, samples_1):
        records.append(
            {
                PLOT_X: f"{round(percent)}%",
                PLOT_Y: round(sample_1, 2),
                "Conformer": "R",
            },
        )

    for percent, sample_2 in zip(percents, samples_2):
        records.append(
            {
                PLOT_X: f"{round(percent)}%",
                PLOT_Y: round(sample_2, 2),
                "Conformer": "L",
            },
        )

    return pd.DataFrame(records)


def make_sample_plot(samples: Samples) -> Plot:
    df = make_sample_dataframe(samples)

    fig, ax = plt.subplots()

    sns.lineplot(
        data=df,
        x=PLOT_X,
        y=PLOT_Y,
        ax=ax,
    )

    return fig


def make_sample_plot_compare(samples_1: Samples, samples_2: Samples) -> Plot:
    df = make_sample_dataframe_compare(samples_1, samples_2)

    fig, ax = plt.subplots()

    sns.lineplot(data=df, x=PLOT_X, y=PLOT_Y, ax=ax, hue="Conformer")

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


def sample_bond_ed_compare(
    structure_path_1: Path,
    structure_path_2: Path,
    xmap_path: Path,
    output_dir: Path,
    atom_1_id: AtomID,
    atom_2_id: AtomID,
):
    logger.info(f"Reading structure at path: {structure_path_1}")
    structure_1: Structure = read_structure(structure_path_1)
    logger.debug(f"Structure: {structure_1}")

    logger.info(f"Reading structure at path: {structure_path_2}")
    structure_2: Structure = read_structure(structure_path_2)
    logger.debug(f"Structure: {structure_2}")

    logger.info(f"Reading structure at path: {xmap_path}")
    xmap: Xmap = read_map(xmap_path)
    logger.debug(f"Xmap: {xmap}")

    bond_1: Bond = Bond(
        atom_2=pos_from_atom_id(atom_2_id, structure_1),
        atom_1=pos_from_atom_id(atom_1_id, structure_1),
    )

    logger.info("Determining samples...")
    samples_1: Samples = sample_along_bond_radius(
        xmap,
        bond_1,
    )

    bond_2: Bond = Bond(
        atom_1=pos_from_atom_id(atom_1_id, structure_2),
        atom_2=pos_from_atom_id(atom_2_id, structure_2),
    )

    logger.info("Determining samples...")
    samples_2: Samples = sample_along_bond_radius(
        xmap,
        bond_2,
    )

    logger.info("Making plot...")
    plot: Plot = make_sample_plot_compare(samples_1, samples_2)

    logger.info(f"Saving plot in dir: {output_dir}")
    save_plot(plot, output_dir / "bond_ed_sampling.png")


def sample_bond_ed_radius(
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

    bond_1: Bond = Bond(
        atom_1=pos_from_atom_id(atom_1_id, structure),
        atom_2=pos_from_atom_id(atom_2_id, structure),
    )

    logger.info("Determining samples...")
    samples_1: Samples = sample_along_bond_radius(
        xmap,
        bond_1,
    )

    bond_2: Bond = Bond(
        atom_1=pos_from_atom_id(atom_1_id, structure),
        atom_2=pos_from_atom_id(atom_2_id, structure),
    )

    logger.info("Determining samples...")
    samples_2: Samples = sample_along_bond_radius(
        xmap,
        bond_2,
    )

    logger.info("Making plot...")
    plot: Plot = make_sample_plot_compare(samples_1, samples_2)

    logger.info(f"Saving plot in dir: {output_dir}")
    save_plot(plot, output_dir / "bond_ed_sampling.png")

def get_predicted_xmap(model, xmap):

    model.cell = xmap.unit_cell
    model.spacegroup_hm = gemmi.find_spacegroup_by_name("P 1").hm
    dencalc = gemmi.DensityCalculatorE()
    dencalc.d_min = 1.35001  # *2
    dencalc.rate = 1.5
    dencalc.set_grid_cell_and_spacegroup(model)
    dencalc.put_model_density_on_grid(model[0])
    # dencalc.add_model_density_to_grid(optimized_structure[0])
    calc_grid = dencalc.grid

    return calc_grid

def get_corr(masked_event_map_vals, masked_calc_vals):
    event_map_mean = np.mean(masked_event_map_vals)
    calc_map_mean = np.mean(masked_calc_vals)
    delta_event_map = masked_event_map_vals - event_map_mean
    delta_calc_map = masked_calc_vals - calc_map_mean
    nominator = np.sum(delta_event_map * delta_calc_map)
    denominator = np.sqrt(
        np.sum(np.square(delta_event_map)) * np.sum(np.square(delta_calc_map))
    )

    corr = nominator / denominator

    return corr

def intergrate_along_bond(
    model,
    xmap,
    atom_1_id,
    atom_2_id,
    num_sample_along_bond=100
):
    bond: Bond = Bond(
        atom_1=pos_from_atom_id(atom_1_id, model),
        atom_2=pos_from_atom_id(atom_2_id, model),
    )
    sample_centers = get_sample_centres(bond, num_sample_along_bond)

    sample_positions = get_sample_positions_near_samples(sample_centers, 10000, 0.5)
    # print(sample_positions[0])

    # Get the predicted density
    # predicted_xmap = get_predicted_xmap(model, xmap)
    # print([xmap.nu, predicted_xmap.nu])

    samples: Samples = sample_at_positions(xmap, sample_positions)
    # calc_samples: Samples = sample_at_positions(predicted_xmap, sample_positions)

    print(f"Num Samples: {len(samples)}")

    sample_array = np.array(samples)
    # calc_sample_array = np.array(calc_samples)

    # corr = get_corr(sample_array, calc_sample_array)

    return np.mean(sample_array)
    # return corr

class CLI:
    def harold_data(self):

        harold_runs = {
            "8BW3_5S9F_PHIPA-x11637" : {
                "structure_path_1": "stero_final_files_incl_S/8BW3_5S9F_PHIPA-x11637-final_files/PHIPA-x11637_4.1_refmac8O.mmcif",
                "structure_path_2": "stero_final_files_incl_S/8BW3_5S9F_PHIPA-x11637-final_files/PHIPA-x11637_4.1_refmac8O_S.mmcif",
                "xmap_path": "stero_final_files_incl_S/8BW3_5S9F_PHIPA-x11637-final_files/PHIPA-x11637-event_1_1-BDC_0.4_map.native.ccp4",
                "atom_1_id": "A/1501/C2",
                "atom_2_id": "A/1501/C1"
            },
            "8BW3_5S9H_PHIPA-x12337": {
                "structure_path_1": "stero_final_files_incl_S/8BW3_5S9H_PHIPA-x12337-final_files/PHIPA-x12337_TRUE_2O.mmcif",
                "structure_path_2": "stero_final_files_incl_S/8BW3_5S9H_PHIPA-x12337-final_files/PHIPA-x12337_TRUE_2O_S.mmcif",
                "xmap_path": "stero_final_files_incl_S/8BW3_5S9H_PHIPA-x12337-final_files/PHIPA-x12337-event_1_1-BDC_0.35_map.native.ccp4",
                "atom_1_id": "A/1501/C2",
                "atom_2_id": "A/1501/C1"
            },
            "8BW4_5S9I_PHIPA-x12340": {
                "structure_path_1": "stero_final_files_incl_S/8BW4_5S9I_PHIPA-x12340-final_files/PHIPA-x12340_16.1_refmac10O.mmcif",
                "structure_path_2": "stero_final_files_incl_S/8BW4_5S9I_PHIPA-x12340-final_files/PHIPA-x12340_16.1_refmac10O_S.mmcif",
                "xmap_path": "stero_final_files_incl_S/8BW4_5S9I_PHIPA-x12340-final_files/PHIPA-x12340-event_1_1-BDC_0.51_map.native.ccp4",
                "atom_1_id": "A/1501/C2",
                "atom_2_id": "A/1501/C1"
            }
        }

        # Get the intergrated ED around
        records = []
        for dataset, dataset_info in harold_runs.items():
            xmap = read_map(Path(dataset_info['xmap_path']))

            # Get the normal model
            model_r = read_structure(Path(dataset_info['structure_path_1']))
            density = intergrate_along_bond(
                model_r,
                xmap,
                dataset_info['atom_1_id'],
                dataset_info['atom_2_id']
            )
            records.append(
                {
                    "Dataset": dataset,
                    "Chirality": "R",
                    "Density": density

                }
            )

            # Get the s model
            model_s = read_structure(Path(dataset_info['structure_path_2']))
            density = intergrate_along_bond(
                model_s,
                xmap,
                dataset_info['atom_1_id'],
                dataset_info['atom_2_id']
            )
            records.append(
                {
                    "Dataset": dataset,
                    "Chirality": "S",
                    "Density": density

                }
            )

        # Generate table
        df = pd.DataFrame(records)
        print(df)

        # Generate csv

        # Plot swarm




# test with: python -m sample_bond_ed
if __name__ == "__main__":
    fire.Fire(CLI)
    # parser = ArgumentParser()
    # # parser.add_argument("--version", action="version", version=__version__)
    # parser.add_argument("--structure_path_1", type=Path)
    # parser.add_argument("--structure_path_2", type=Path)
    # parser.add_argument("--xmap_path", type=Path)
    # parser.add_argument("--output_dir", type=Path)
    # parser.add_argument(
    #     "--atom_1_id",
    #     help="""
    #     A selector for a single atom of the form CHAIN/RESIDUE/NAME,
    #     such as \"C/LIG/C1\"
    #     """,
    # )
    # parser.add_argument(
    #     "--atom_2_id",
    #     help="""
    #     A selector for a single atom of the form CHAIN/RESIDUE/NAME,
    #     such as \"C/LIG/C1\"
    #     """,
    # )
    # args = parser.parse_args()
    #
    # structure_path_1: Path = args.structure_path_1
    # structure_path_2: Path = args.structure_path_2
    # xmap_path: Path = args.xmap_path
    # output_dir: Path = args.output_dir
    # atom_1_id: AtomID = args.atom_1_id
    # atom_2_id: AtomID = args.atom_2_id
    #
    # logger.info(f"Structure path: {structure_path_1}")
    # logger.info(f"Structure path: {structure_path_2}")
    #
    # logger.info(f"Atom 1 ID: {atom_1_id}")
    # logger.info(f"Atom 2 ID: {atom_2_id}")
    #
    # sample_bond_ed_compare(
    #     structure_path_1,
    #     structure_path_2,
    #     xmap_path,
    #     output_dir,
    #     atom_1_id,
    #     atom_2_id,
    # )
