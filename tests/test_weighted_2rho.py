"""Cross-check species-weighted 2rho descriptors and the LAMMPS export."""

import os
import subprocess

import ase
from ase.io import write
import numpy as np
import pytest

from lammps_plugin.scripts.generate_uf3_lammps_pots import write_uf3_lammps_pot_files
from uf3.data.composition import ChemicalSystem
from uf3.forcefield.calculator import UFCalculator
from uf3.regression.least_squares import WeightedLinearModel, arrange_coefficients
from uf3.representation.bspline import BSplineBasis
from uf3.representation.process import BasisFeaturizer


def make_model(unit_weights=False):
    system = ChemicalSystem(["C", "Co", "W"], degree=2)
    weights = {("density", center, neighbor): (1.0 if unit_weights else weight)
               for center in system.element_list
               for neighbor, weight in (("C", 0.5), ("Co", 1.0), ("W", 1.7))}
    config = BSplineBasis(
        system,
        r_min_map={pair: 0.5 for pair in system.interactions_map[2]},
        r_max_map={pair: 5.0 for pair in system.interactions_map[2]},
        resolution_map={pair: 8 for pair in system.interactions_map[2]},
        knot_strategy="lammps",
        density_modulated_2b_enabled=True,
        density_modulated_2b_alphas=[0.5],
        density_modulated_2b_r_min_map={pair: 0.1 for pair in system.interactions_map[2]},
        density_modulated_2b_r_max_map={pair: 5.0 for pair in system.interactions_map[2]},
        density_modulated_2b_rho_bulk_map={key: 0.6 for key in weights},
        density_modulated_2b_weight_map=weights,
    )
    model = WeightedLinearModel(bspline_config=config)
    model.coefficients = np.zeros(sum(config.partition_sizes))
    _, offsets = config.get_interaction_partitions()
    for key in config.density_modulated_2b_interactions:
        model.coefficients[offsets[key] + 3] = 0.7
        model.coefficients[offsets[key] + 4] = -0.2
    return model


def make_atoms():
    return ase.Atoms(
        symbols=["C", "Co", "W", "Co", "W", "Co"],
        positions=[[1.0, 1.0, 1.0], [2.7, 1.1, 1.0],
                   [1.2, 3.0, 1.1], [3.5, 3.4, 1.3],
                   [1.2, 3.0, 5.5], [2.0, 3.4, 6.5]],
        cell=[12.0, 12.0, 12.0], pbc=True,
    )


def test_weighted_2rho_round_trip_and_forces(tmp_path):
    model = make_model()
    model_path = tmp_path / "weighted.json"
    model.to_json(str(model_path))
    model = WeightedLinearModel.from_json(str(model_path))
    assert model.bspline_config.density_modulated_2b_weight_map[
        ("density", "Co", "W")] == 1.7

    atoms = make_atoms()
    calc = UFCalculator(model)
    atoms.calc = calc
    force = atoms.get_forces()[2, 1]
    step = 1e-6
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[2, 1] += step
    minus.positions[2, 1] -= step
    plus.calc = minus.calc = calc
    finite_difference = -(plus.get_potential_energy() - minus.get_potential_energy()) / (2 * step)
    np.testing.assert_allclose(force, finite_difference, atol=1e-6)


def test_weighted_2rho_training_features():
    model = make_model()
    atoms = make_atoms()
    atoms.calc = UFCalculator(model)
    config = model.bspline_config
    featurizer = BasisFeaturizer(config)
    solutions = arrange_coefficients(model.coefficients, config)
    coefficients = np.concatenate([
        solutions[key]
        for key in config.density_modulated_2b_interactions
    ])
    energy_features = featurizer.featurize_energy_density_modulated_2b(atoms)
    force_features = featurizer.featurize_force_density_modulated_2b(atoms)
    np.testing.assert_allclose(energy_features @ coefficients,
                               atoms.get_potential_energy(), atol=1e-8)
    np.testing.assert_allclose(np.tensordot(force_features, coefficients, axes=([2], [0])),
                               atoms.get_forces(), atol=1e-8)
    unit_model = make_model(unit_weights=True)
    unit_atoms = make_atoms()
    unit_atoms.calc = UFCalculator(unit_model)
    assert not np.isclose(atoms.get_potential_energy(), unit_atoms.get_potential_energy())


@pytest.mark.parametrize("legacy_header", [False, True])
def test_weighted_2rho_lammps_parity(tmp_path, legacy_header):
    lmp = os.environ.get("UF3_LAMMPS_BIN")
    if not lmp:
        pytest.skip("Set UF3_LAMMPS_BIN to run the local LAMMPS parity test")
    model = make_model(unit_weights=legacy_header)
    atoms = make_atoms()
    atoms.calc = UFCalculator(model)
    expected_energy = atoms.get_potential_energy()
    expected_forces = atoms.get_forces()

    pot_path = tmp_path / "weighted.uf3"
    write_uf3_lammps_pot_files(
        model.bspline_config.chemical_system, model, "nk", str(tmp_path),
        pot_path.name, "UF3_test", "metal", emit_density_modulated_2b=True,
    )
    lines = pot_path.read_text().splitlines()
    assert sum(line.startswith("2RHO ") for line in lines) == 9
    for index, line in enumerate(lines):
        if line.startswith("2RHO "):
            assert len(lines[index + 3].split()) == 8
            if legacy_header:
                fields = lines[index + 3].split()
                lines[index + 3] = " ".join(fields[:5] + fields[6:])
    if legacy_header:
        pot_path.write_text("\n".join(lines) + "\n")

    write(tmp_path / "atoms.data", atoms, format="lammps-data",
          specorder=["C", "Co", "W"], atom_style="atomic")
    (tmp_path / "in.parity").write_text(
        "units metal\natom_style atomic\nboundary p p p\n"
        "read_data atoms.data\n"
        "mass 1 12.011\nmass 2 58.933\nmass 3 183.84\n"
        "pair_style uf3 2\n"
        "pair_coeff * * weighted.uf3 C Co W\n"
        "thermo_style custom step pe\nthermo 1\nrun 0\n"
        "variable energy equal pe\nprint \"UF3_TEST_PE ${energy}\"\n"
        "write_dump all custom forces.dump id type fx fy fz\n"
    )
    command = [lmp, "-in", "in.parity", "-log", "none"]
    ranks = int(os.environ.get("UF3_LAMMPS_RANKS", "1"))
    if ranks > 1:
        command = ["mpirun", "-np", str(ranks), *command]
    result = subprocess.run(command,
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = next(line for line in result.stdout.splitlines()
                  if line.startswith("UF3_TEST_PE "))
    actual_energy = float(marker.split()[1])
    dump_lines = (tmp_path / "forces.dump").read_text().splitlines()
    start = dump_lines.index("ITEM: ATOMS id type fx fy fz") + 1
    actual_forces = np.array([list(map(float, line.split()[2:]))
                              for line in sorted(dump_lines[start:start + len(atoms)],
                                                 key=lambda row: int(row.split()[0]))])
    np.testing.assert_allclose(actual_energy, expected_energy, atol=1e-6)
    np.testing.assert_allclose(actual_forces, expected_forces, atol=1e-6)
