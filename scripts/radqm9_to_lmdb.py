import argparse
import lmdb
from io import StringIO
from ase.io import read
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import mdsim.common.utils as utils

EV_TO_KCAL_MOL = 23.06052


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.

    """

    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        device="cpu",
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_edges = r_edges
        self.device = device

    def convert(
        self,
        natoms,
        positions,
        atomic_numbers,
        lengths=None,
        angles=None,
        energy=None,
        forces=None,
        cell=None,
        charge=None,
        spin=None,
        partial_charges=None,
        partial_spin=None,
        dipole=None,
        resp_dipole=None,
        mol_id=None,
    ):
        """Convert a batch of atomic stucture to a batch of graphs.

        Args:
            natoms: (B), sum(natoms) == N
            positions: (B*N, 3)
            atomic_numbers: (B*N)
            lengths: (B, 3) lattice lengths [lx, ly, lz]
            angles: (B, 3) lattice angles [ax, ay, az]
            forces: (B*N, 3)
            energy: (B)
            cell: (B, 3, 3) lattice vectors
            charge: (B)
            spin: (B)
            partial_charges: (B*N)
            partial_spin: (B*N)
            dipole: (B, 3)
            resp_dipole: (B, 3)
            mol_id: (B)


        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        natoms = torch.from_numpy(natoms).to(self.device).long()
        positions = torch.from_numpy(positions).to(self.device).float()
        atomic_numbers = torch.from_numpy(atomic_numbers).to(self.device).long()
        if cell is None:
            lengths = torch.from_numpy(lengths).to(self.device).float()
            angles = torch.from_numpy(angles).to(self.device).float()
            cells = utils.lattice_params_to_matrix_torch(lengths, angles).float()
        else:
            cells = torch.from_numpy(cell).to(self.device).float()

        data = Data(
            cell=cells,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
        )

        # optionally include other properties
        if self.r_edges:
            edge_index, cell_offsets, edge_distances, _ = utils.radius_graph_pbc(
                data, self.radius, self.max_neigh
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if energy is not None:
            energy = torch.from_numpy(energy).to(self.device).float()
            data.y = energy
        if forces is not None:
            forces = torch.from_numpy(forces).to(self.device).float()
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances

        if dipole is not None:
            dipole = torch.from_numpy(dipole).to(self.device).float()
            data.dipole = dipole

        if resp_dipole is not None:
            resp_dipole = torch.from_numpy(resp_dipole).to(self.device).float()
            data.resp_dipole = resp_dipole

        if partial_charges is not None:
            partial_charges = torch.from_numpy(partial_charges).to(self.device).float()
            data.partial_charges = partial_charges

        if partial_spin is not None:
            partial_spin = torch.from_numpy(partial_spin).to(self.device).float()
            data.partial_spin = partial_spin

        if mol_id is not None:
            data.mol_id = mol_id
        # print(charge, spin)
        # print(type(charge), type(spin))
        if charge is not None:
            # convert np.int64 to torch tensor
            charge = torch.tensor(charge, device=self.device).int()
            data.charge = charge

        if spin is not None:
            # convert np.int64 to torch tensor
            spin = torch.tensor(spin, device=self.device).int()
            data.spin = spin

        fixed_idx = torch.zeros(natoms).float()
        data.fixed = fixed_idx

        return data.cpu()


def stream_xyz(file_xyz):
    lines_temp = []
    with open(file_xyz, "r") as file:
        for line in file:
            if line.strip().isnumeric() and len(lines_temp) > 0:
                lines_concat = "".join(lines_temp)
                f = StringIO(lines_concat)
                atoms = read(f, format="extxyz")
                yield atoms
                lines_temp = [line]
            else:
                lines_temp.append(line)


def convert_to_lmdb(
    dataset_name="radqm9_test",
    ref_file="/home/santiagovargas/dev/aimnet2/data/xyz/radqm9_65_10_25_sp_vacuum_full_data_20240807_val.xyz",
    db_path="/home/santiagovargas/dev/EScAIP/dev/scripts/data/radqm9_val_full/",
    split_tf=True,
):
    # split_tf = True
    # dataset_name = "radqm9_test"
    # ref_file = "/home/santiagovargas/dev/aimnet2/data/xyz/radqm9_65_10_25_sp_vacuum_full_data_20240807_val.xyz"
    # db_path = "/home/santiagovargas/dev/EScAIP/dev/scripts/data/radqm9_val_full/"

    save_path = Path(db_path) / dataset_name
    save_path.mkdir(parents=True, exist_ok=True)

    dataset_size, dataset_size_atoms = 0, 0
    energy_list = []
    force_list = []
    spin_list = []
    charge_list = []
    positions_list = []
    dipole_list = []
    resp_dipole_list = []
    resp_partial_charges_list = []
    mulliken_partial_charges_list = []
    atomic_number_list = []
    mol_id_list = []

    for atoms in stream_xyz(ref_file):
        positions = atoms.get_positions()
        n_points = positions.shape[0]
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_numbers = atomic_numbers.astype(np.int64)

        info_dict = atoms.info
        # print(info_dict)
        spin = int(info_dict["spin"])
        charge = int(info_dict["charge"])
        total_energy = float(info_dict["relative_energy"])
        # ref_energy = info_dict["REF_energy"]
        mol_id = str(info_dict["mol_id"])
        # position_type = info_dict["position_type"]
        dipole = info_dict["dipole_moment"]
        resp_dipole = info_dict["resp_dipole_moment"]
        # print(atoms.arrays["REF_forces"])
        forces = np.array(atoms.arrays["REF_forces"])

        # partial charges
        resp_partial_charges = atoms.arrays["resp_partial_charges"]
        mulliken_partial_charges = atoms.arrays["mulliken_partial_charges"]

        dataset_size += 1
        dataset_size_atoms += n_points
        # try:
        energy_list.append(total_energy)
        force_list.append(forces)
        spin_list.append(spin)
        atomic_number_list.append(atomic_numbers)
        charge_list.append(charge)
        positions_list.append(positions)
        dipole_list.append(dipole)
        resp_dipole_list.append(resp_dipole)
        resp_partial_charges_list.append(resp_partial_charges)
        mulliken_partial_charges_list.append(mulliken_partial_charges)
        mol_id_list.append(mol_id)
        # except:
        #    # throw error

        # if dataset_size == 100:
        #    break
    print("spin set", set(spin_list))
    print("charge set", set(charge_list))

    energy_list = np.array(energy_list).reshape(
        -1, 1
    )  # Reshape energy into 2D array ADDED TODO
    # force_list = np.array(force_list)
    lengths = np.ones(3)[None, :] * 30.0  # where does this come from?
    angles = np.ones(3)[None, :] * 90.0  # where does this come from?

    # n_size = dataset_size
    print(f"dataset size: {dataset_size}")

    if split_tf:
        train_val_index = np.linspace(0, dataset_size - 1, dataset_size, dtype=int)
        # print(train_val_index)
        test = np.setdiff1d(np.arange(dataset_size), train_val_index)
        np.random.shuffle(test)

        test = test.tolist()
        train, val = train_test_split(
            train_val_index, train_size=0.95, test_size=0.05, random_state=42
        )

        ranges = [train, val, test]

        e_train = energy_list[train]
        f_train = [force_list[i] for i in train]
        spin_train = [spin_list[i] for i in train]
        charge_train = [charge_list[i] for i in train]

        norm_stats = {
            "e_mean": e_train.mean(),
            "e_std": e_train.std(),
            "f_mean": np.concatenate(f_train).mean(),
            "f_std": np.concatenate(f_train).std(),
            "spin_mean": np.array(spin_train).mean(),
            "spin_std": np.array(spin_train).std(),
            "charge_mean": np.array(charge_train).mean(),
            "charge_std": np.array(charge_train).std(),
        }
        print(norm_stats)

        np.save(save_path / "metadata", norm_stats)
        # save identical to npz
        np.savez(save_path / "data.npz", norm_stats)

        a2g = AtomsToGraphs(
            max_neigh=1000,
            radius=6,
            r_energy=True,
            r_forces=True,
            r_distances=False,
            r_edges=False,
            device="cpu",
        )

        for spidx, split in enumerate(["train", "val", "test"]):
            print(f"processing split {split}.")
            if len(ranges[spidx]) == 0:
                continue
            save_path = Path(db_path) / dataset_name / split
            save_path.mkdir(parents=True, exist_ok=True)

            db = lmdb.open(
                str(save_path / "data.lmdb"),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )

            for i, idx in enumerate(tqdm(ranges[spidx])):
                natoms = np.array([positions_list[idx].shape[0]] * 1, dtype=np.int64)
                # print("converting")
                # print(natoms, positions_list[idx].shape, atomic_number_list[idx].shape, force_list[idx].shape)

                data = a2g.convert(
                    natoms,
                    positions_list[idx],
                    atomic_number_list[idx],
                    lengths,
                    angles,
                    energy_list[idx],
                    force_list[idx],
                    charge=charge_list[idx],
                    spin=spin_list[idx],
                    mol_id=mol_id_list[idx],
                    # partial_charges=resp_partial_charges_list[idx],
                    # partial_spin=mulliken_partial_charges_list[idx],
                    dipole=dipole_list[idx],
                    resp_dipole=resp_dipole_list[idx],
                )
                # print("converted")
                data.sid = 0
                data.fid = idx

                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pkl.dumps(data, protocol=-1))
                txn.commit()

            # Save count of objects in lmdb.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pkl.dumps(i, protocol=-1))
            txn.commit()

            db.sync()
            db.close()

    else:
        norm_stats = {
            "e_mean": float(energy_list.mean()),
            "e_std": float(energy_list.std()),
            "f_mean": np.concatenate(force_list).mean(),
            "f_std": np.concatenate(force_list).std(),
        }
        print(norm_stats)
        np.save(save_path / "metadata", norm_stats)
        np.savez(save_path / "data.npz", norm_stats)

        a2g = AtomsToGraphs(
            max_neigh=1000,
            radius=6,
            r_energy=True,
            r_forces=True,
            r_distances=False,
            r_edges=False,
            device="cpu",
        )

        save_path = Path(db_path) / dataset_name
        save_path.mkdir(parents=True, exist_ok=True)

        db = lmdb.open(
            str(save_path / "data.lmdb"),
            map_size=1099511627776 * 2,
            subdir=False,
            meminit=False,
            map_async=True,
        )

        for i in tqdm(range(dataset_size)):
            natoms = np.array([positions_list[i].shape[0]] * 1, dtype=np.int64)
            data = a2g.convert(
                natoms,
                positions_list[i],
                atomic_number_list[i],
                lengths,
                angles,
                energy_list[i],
                force_list[i],
                charge=charge_list[i],
                spin=spin_list[i],
                mol_id=mol_id_list[i],
                # partial_charges=resp_partial_charges_list[i],
                # partial_spin=mulliken_partial_charges_list[i],
                dipole=dipole_list[i],
                resp_dipole=resp_dipole_list[i],
            )
            data.sid = 0
            data.fid = i

            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), pkl.dumps(data, protocol=-1))
            txn.commit()

        # Save count of objects in lmdb.
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pkl.dumps(i, protocol=-1))
        txn.commit()

        db.sync()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="radqm9_test")
    parser.add_argument(
        "--ref_file",
        type=str,
        default="/home/santiagovargas/dev/aimnet2/data/xyz/radqm9_65_10_25_sp_vacuum_full_data_20240807_val.xyz",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="/home/santiagovargas/dev/EScAIP/dev/scripts/data/radqm9_val_full/",
    )
    parser.add_argument("--split_tf", type=bool, default=True)
    args = parser.parse_args()

    convert_to_lmdb(
        str(args.dataset_name),
        str(args.ref_file),
        str(args.db_path),
        bool(args.split_tf),
    )
