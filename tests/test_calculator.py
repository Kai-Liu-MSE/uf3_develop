import numpy as np
import ase
import os
import uf3
from uf3.data import composition
from uf3.representation import bspline
from uf3.regression import least_squares
from uf3.forcefield import calculator


class TestCalculator:
    def test_unary_dimer(self):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(chemical_system,
                                              r_min_map={('W', 'W'): 2.0},
                                              r_max_map={('W', 'W'): 6.0},
                                              resolution_map={('W', 'W'): 20},
                                              knot_strategy='lammps')
        model = least_squares.WeightedLinearModel(
            bspline_config=bspline_config)
        pair = bspline_config.interactions_map[2][0]
        x = np.linspace(bspline_config.r_min_map[pair],
                        bspline_config.r_max_map[pair],
                        1000)
        y = 4 * 0.87 * ((2.5 / x)**12 - (2.5 / x)**6)
        knot_sequence = bspline_config.knots_map[pair]
        coefficient_vector = bspline.fit_spline_1d(x,
                                                   y,
                                                   knot_sequence)
        coefficient_vector = np.insert(coefficient_vector, 0, 0)
        model.coefficients = coefficient_vector
        calc = calculator.UFCalculator(model)
        assert len(calc.solutions) == 2
        assert len(calc.pair_potentials) == 1
        geom = ase.Atoms('W2',
                          positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
                          pbc=False,
                          cell=None)
        energy = calc.get_potential_energy(geom)
        assert np.isclose(energy, -1.21578)
        geom.calc = calc
        forces = geom.get_forces()
        assert np.allclose(forces, [[-3.96244881, -3.96244881, -3.96244881],
                                    [3.96244881, 3.96244881, 3.96244881]])
        geom.set_pbc([True, True, True])
        geom.set_cell([[3, 0, 0], [3, 5, 0], [0, 0, 3]])
        assert np.isclose(geom.get_potential_energy(), -15.33335)
        forces = geom.get_forces()
        assert np.allclose(forces, [[0, -17.3656864, 0], [0, 17.3656864, 0]])

    def test_density_unary_dimer(self):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(
            chemical_system,
            density_enabled=True,
            density_r_min_map={('W', 'W'): 0.1},
            density_r_max_map={('W', 'W'): 6.0},
            density_power=1.0)
        model = least_squares.WeightedLinearModel(bspline_config=bspline_config)
        model.coefficients = np.zeros(sum(bspline_config.partition_sizes))
        density_key = bspline_config.density_interactions[0]
        offset = bspline_config.get_interaction_partitions()[1][density_key]
        model.coefficients[offset] = 2.0
        calc = calculator.UFCalculator(model)
        geom = ase.Atoms('W2',
                         positions=[[0, 0, 0], [2.0, 0, 0]],
                         pbc=False,
                         cell=None)
        geom.calc = calc
        # Ordered center-neighbor density counts both atoms: 2 * (1 / r).
        assert np.isclose(geom.get_potential_energy(), 2.0)
        assert np.allclose(geom.get_forces(), [[-1.0, 0.0, 0.0],
                                               [1.0, 0.0, 0.0]])

    def test_density_modulated_2b_force_consistency(self, tmp_path):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(
            chemical_system,
            r_min_map={('W', 'W'): 0.5},
            r_max_map={('W', 'W'): 5.0},
            resolution_map={('W', 'W'): 6},
            density_modulated_2b_enabled=True,
            density_modulated_2b_alphas=[0.5],
            density_modulated_2b_r_min_map={('W', 'W'): 0.1},
            density_modulated_2b_r_max_map={('W', 'W'): 5.0},
            density_modulated_2b_power=1.0,
            density_modulated_2b_rho_bulk_map={('W', 'W'): 0.5})
        model = least_squares.WeightedLinearModel(bspline_config=bspline_config)
        model.coefficients = np.zeros(sum(bspline_config.partition_sizes))
        key = bspline_config.density_modulated_2b_interactions[0]
        offset = bspline_config.get_interaction_partitions()[1][key]
        model.coefficients[offset + 3] = 1.0
        model_path = tmp_path / "model_2rho.json"
        model.to_json(str(model_path))
        model = least_squares.WeightedLinearModel.from_json(str(model_path))
        calc = calculator.UFCalculator(model)

        geom = ase.Atoms('W3',
                         positions=[[0.0, 0.0, 0.0],
                                    [2.0, 0.0, 0.0],
                                    [0.0, 2.3, 0.0]],
                         pbc=False,
                         cell=None)
        geom.calc = calc
        forces = geom.get_forces()
        step = 1e-6
        geom_plus = geom.copy()
        geom_minus = geom.copy()
        geom_plus.positions[1, 0] += step
        geom_minus.positions[1, 0] -= step
        geom_plus.calc = calc
        geom_minus.calc = calc
        finite_difference = -(
            geom_plus.get_potential_energy() -
            geom_minus.get_potential_energy()) / (2 * step)
        assert np.isclose(forces[1, 0], finite_difference)

    def test_density_modulated_2b_group_freezing(self):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(
            chemical_system,
            r_min_map={('W', 'W'): 0.5},
            r_max_map={('W', 'W'): 5.0},
            resolution_map={('W', 'W'): 6},
            density_modulated_2b_enabled=True,
            density_modulated_2b_r_min_map={('W', 'W'): 0.1},
            density_modulated_2b_r_max_map={('W', 'W'): 5.0},
            density_modulated_2b_rho_bulk_map={('W', 'W'): 0.5})
        model = least_squares.WeightedLinearModel(bspline_config=bspline_config)
        idx_2rho = bspline_config.get_feature_group_indices("2rho")
        idx_base = bspline_config.get_feature_group_indices(["1b", "2b"])
        model.freeze_feature_groups(["1b", "2b"])
        assert set(idx_base).issubset(set(model.col_idx))
        assert not set(idx_2rho).issubset(set(model.col_idx))

    def test_dipole_modulated_2b_force_consistency(self, tmp_path):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(
            chemical_system,
            r_min_map={('W', 'W'): 0.5},
            r_max_map={('W', 'W'): 5.0},
            resolution_map={('W', 'W'): 6},
            dipole_modulated_2b_enabled=True,
            dipole_modulated_2b_r_min_map={('W', 'W'): 0.1},
            dipole_modulated_2b_r_max_map={('W', 'W'): 5.0},
            dipole_modulated_2b_power=1.0,
            dipole_modulated_2b_eps=1e-8)
        model = least_squares.WeightedLinearModel(bspline_config=bspline_config)
        model.coefficients = np.zeros(sum(bspline_config.partition_sizes))
        key = bspline_config.dipole_modulated_2b_interactions[0]
        offset = bspline_config.get_interaction_partitions()[1][key]
        model.coefficients[offset + 3] = 1.0
        model_path = tmp_path / "model_2D.json"
        model.to_json(str(model_path))
        model = least_squares.WeightedLinearModel.from_json(str(model_path))
        calc = calculator.UFCalculator(model)

        geom = ase.Atoms('W3',
                         positions=[[0.0, 0.0, 0.0],
                                    [2.0, 0.1, 0.0],
                                    [0.3, 2.3, 0.2]],
                         pbc=False,
                         cell=None)
        geom.calc = calc
        forces = geom.get_forces()
        step = 1e-6
        geom_plus = geom.copy()
        geom_minus = geom.copy()
        geom_plus.positions[1, 0] += step
        geom_minus.positions[1, 0] -= step
        geom_plus.calc = calc
        geom_minus.calc = calc
        finite_difference = -(
            geom_plus.get_potential_energy() -
            geom_minus.get_potential_energy()) / (2 * step)
        assert np.isclose(forces[1, 0], finite_difference, rtol=1e-5, atol=1e-7)

    def test_dipole_modulated_2b_group_freezing(self):
        element_list = ['W']
        chemical_system = composition.ChemicalSystem(element_list)
        bspline_config = bspline.BSplineBasis(
            chemical_system,
            r_min_map={('W', 'W'): 0.5},
            r_max_map={('W', 'W'): 5.0},
            resolution_map={('W', 'W'): 6},
            dipole_modulated_2b_enabled=True,
            dipole_modulated_2b_r_min_map={('W', 'W'): 0.1},
            dipole_modulated_2b_r_max_map={('W', 'W'): 5.0})
        model = least_squares.WeightedLinearModel(bspline_config=bspline_config)
        idx_2d = bspline_config.get_feature_group_indices("2D")
        idx_base = bspline_config.get_feature_group_indices(["1b", "2b"])
        model.freeze_feature_groups(["1b", "2b"])
        assert set(idx_base).issubset(set(model.col_idx))
        assert not set(idx_2d).issubset(set(model.col_idx))

    def test_unary_trimer(self):
        geom = ase.Atoms("W3",
                         positions=[[0, 0, 0], [2, 0, 0], [0, 3, 0]],
                         pbc=False,
                         cell=None)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_unary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, -18.79979353611411)
        forces = geom.get_forces()
        assert np.allclose(forces, [[-12.26367499,   0.15140673,   0.        ],
                                    [ 12.05608935,   0.31137845,   0.        ],
                                    [  0.20758563,  -0.46278518,   0.        ]]
                           )

    def test_unary_pbc(self):
        geom = ase.Atoms("W8",
                         positions=[[ 0.00,  0.00,  0.00], [ 2.89,  0.12, -0.04],
                                    [-0.32,  2.71, -0.11], [ 2.65,  2.81,  0.37],
                                    [ 0.00,  0.00,  3.00], [ 2.64,  0.00,  3.00],
                                    [-0.08,  2.94,  3.16], [ 2.53,  2.87,  3.23]],
                         pbc=True,
                         cell=np.eye(3)*2.74*2)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_unary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, -76.358888229785)
        forces = geom.get_forces()
        assert np.allclose(forces, [[ 1.36696442, -0.46307   ,  1.78573347],
                                    [ 0.20112587,  0.17014795,  1.22172728],
                                    [-0.66043959, -1.08374173,  6.78845939],
                                    [-1.30913745,  0.36888897,  1.48182124],
                                    [-0.33315563,  1.28359885, -1.56572912],
                                    [ 0.01504262,  0.06574851, -2.38044283],
                                    [ 0.25436762,  0.2491558 , -7.48063062],
                                    [ 0.46523214, -0.59072835,  0.14906119]]
                           )

    def test_binary(self):
        geom = ase.Atoms("NeXe", positions=[[0, 0, 0], [3.1, 0, 0]], pbc=False)
        pkg_directory = os.path.dirname(os.path.dirname(uf3.__file__))
        data_directory = os.path.join(pkg_directory, "tests/data")
        model_file = os.path.join(data_directory, "precalculated_ref",
                                  "model_binary.json")
        model = least_squares.WeightedLinearModel.from_json(model_file)
        calc = calculator.UFCalculator(model)
        geom.calc = calc
        energy = geom.get_potential_energy()
        assert np.isclose(energy, 0.3464031387757268)
        forces = geom.get_forces()
        assert np.allclose(forces, [[-0.28138023,  0.        ,  0.        ],
                                    [ 0.28138023,  0.        ,  0.        ]]
                           )
    
