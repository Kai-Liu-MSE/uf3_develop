from uf3.regression import least_squares
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from uf3.data import composition
import numpy as np
import os, sys, argparse
from datetime import datetime


def _trim_values(model, body_order):
    """Return the scalar trim counts expected by the LAMMPS .uf3 format."""
    leading = model.bspline_config.leading_trim
    trailing = model.bspline_config.trailing_trim
    return int(leading[body_order]), int(trailing[body_order])


def _unsupported_nonzero_terms(model, tolerance=1e-14):
    """Return enabled descriptor terms that the current LAMMPS writer omits."""
    config = model.bspline_config
    term_specs = (
        ("density", "density_enabled", "density_interactions"),
        ("dipole", "dipole_enabled", "dipole_interactions"),
        ("shell_density", "shell_density_enabled", "shell_density_interactions"),
        ("density_moment", "density_moment_enabled", "density_moment_interactions"),
        ("local_composition", "local_composition_enabled", "local_composition_interactions"),
        ("density_modulated_2b", "density_modulated_2b_enabled",
         "density_modulated_2b_interactions"),
        ("density_modulated_3b", "density_modulated_3b_enabled",
         "density_modulated_3b_interactions"),
        ("dipole_modulated_2b", "dipole_modulated_2b_enabled",
         "dipole_modulated_2b_interactions"),
        ("bond_dipole_modulated_2b", "bond_dipole_modulated_2b_enabled",
         "bond_dipole_modulated_2b_interactions"),
        ("tensor_modulated_2b", "tensor_modulated_2b_enabled",
         "tensor_modulated_2b_interactions"),
        ("bond_tensor_modulated_2b", "bond_tensor_modulated_2b_enabled",
         "bond_tensor_modulated_2b_interactions"),
    )
    solutions = least_squares.arrange_coefficients(model.coefficients, config)
    unsupported = []
    for term_name, enabled_attr, interactions_attr in term_specs:
        if not getattr(config, enabled_attr, False):
            continue
        interactions = getattr(config, interactions_attr, ())
        if any(np.any(np.abs(solutions[interaction]) > tolerance)
               for interaction in interactions):
            unsupported.append(term_name)
    return unsupported

def main():
    parser = argparse.ArgumentParser(description="Generate UF3 LAMMPS potential file")

    parser.add_argument('-a', '--author', required=True, help="Author Name Seperated by '_'")
    parser.add_argument('-u', '--units', required=True, help="LAMMPS Units")
    parser.add_argument('-m', '--model', required=True, help="UF3 Model JSON file")
    default_directory = "."
    parser.add_argument('-d', '--directory', default=default_directory,
                        help="Directory path (default: current directory)")
    parser.add_argument('-o', '--output', default=None,
                        help="Output UF3 LAMMPS potential filename "
                             "(default: concatenated element symbols + .uf3)")
    default_knots_spacing_type = "nk"
    parser.add_argument('-k', '--knots_spacing_type', default=default_knots_spacing_type,
                        help="Knot spacing type, uk (uniform spacing) or \
                                nk (non-uniform spacing)\n\
                                (default: nk (non-uniform))")
    parser.add_argument('--emit-2rho-schema', action='store_true',
                        help="Deprecated compatibility flag; nonzero 2rho terms "
                             "are emitted by default.")

    args = parser.parse_args()

    model = least_squares.WeightedLinearModel.from_json(args.model)
    unsupported_terms = _unsupported_nonzero_terms(model)
    emittable_terms = {"density", "dipole", "density_modulated_2b"}
    blocked_terms = [term for term in unsupported_terms if term not in emittable_terms]
    if blocked_terms:
        raise ValueError(
            "The current LAMMPS .uf3 exporter only supports ordinary UF3 "
            "2B/3B terms. This model has nonzero unsupported term(s): "
            f"{', '.join(blocked_terms)}. Refusing to write an incomplete "
            "potential file."
        )
    knots_spacing_type = args.knots_spacing_type

    if (knots_spacing_type != "uk") and (knots_spacing_type != "nk"):
        raise ValueError(f"Supplied knot spacing type {knots_spacing_type} is not\n\
                 a valid choice. Only uk or nk are valid types")

    model_elements = set(model.bspline_config.chemical_system.element_list)
    pot_dir = args.directory

    chemical_sys = model.bspline_config.chemical_system

    uf3_lammps_pot_name = args.output or "".join(chemical_sys.element_list)+".uf3"

    write_uf3_lammps_pot_files(chemical_sys=chemical_sys, model=model,
                                knots_spacing_type = knots_spacing_type,
                                pot_dir=pot_dir,
                                uf3_lammps_pot_name = uf3_lammps_pot_name,
                                author = args.author,
                                lammps_units = args.units,
                                emit_density=("density" in unsupported_terms),
                                emit_dipole=("dipole" in unsupported_terms),
                                emit_density_modulated_2b=(
                                    "density_modulated_2b" in unsupported_terms))

    lines = "pair_style\tuf3 %i %i"%(model.bspline_config.degree,len(chemical_sys.element_list))
    lines += "\n"
    lines += "pair_coeff\t* * "+pot_dir+"/"+uf3_lammps_pot_name+ " "
    lines += " ".join(chemical_sys.element_list)

    print("\n\n***Add the following line to the lammps input script***\n\n")
    print(lines)
    print("\n\nCitation meta-data has been left blank. Please enter appropriate"\
            "i citation for the generated UF3 LAMMPS potential\n\n")

def write_uf3_lammps_pot_files(chemical_sys,
                                model,
                                knots_spacing_type,
                                pot_dir,
                                uf3_lammps_pot_name,
                                author,
                                lammps_units,
                                emit_density=False,
                                emit_dipole=False,
                                emit_density_modulated_2b=False):
    """Returns list

    Creates and writes UF3 lammps potential files. Takes UF3 composition object,
    UF3 model, knots_spacing_type, name of potential directory, name of uf3
    lammps pot file to be generateas, author and lammps units input. Will
    overwrite the files if files with the same exists
    """

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(pot_dir):
        os.mkdir(pot_dir)
    files = {}

    for interaction in chemical_sys.interactions_map[2]:
        key = '_'.join(interaction)
        files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
        files[key] += f"AUTHOR: {author} CITATION:\n"

        files[key] += f"2B {interaction[0]} {interaction[1]}"
        leading_trim, trailing_trim = _trim_values(model, 2)
        files[key] += f" {leading_trim} {trailing_trim}"
        if knots_spacing_type == "uk":
            files[key] += " uk\n"
        elif knots_spacing_type == "nk":
            files[key] += " nk\n"
        else:
            raise ValueError(f"Supplied knot spacing type {knots_spacing_type}\n\
                                is not a valid choice. Only uk or nk are valid types")

        files[key] += str(model.bspline_config.r_max_map[interaction]) + " " + \
                str(len(model.bspline_config.knots_map[interaction]))+"\n"

        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.bspline_config.knots_map[interaction]]) + "\n"

        files[key] += str(model.bspline_config.get_interaction_partitions()[0][interaction]) \
                + "\n"

        start_index = model.bspline_config.get_interaction_partitions()[1][interaction]
        length = model.bspline_config.get_interaction_partitions()[0][interaction]
        files[key] += " ".join(['{:.17g}'.format(v) for v in \
                model.coefficients[start_index:start_index + length]]) + "\n"

        files[key] += "#\n"

    if 3 in model.bspline_config.interactions_map:
        for interaction in model.bspline_config.interactions_map[3]:
            key = '_'.join(interaction)
            files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
            files[key] += f"AUTHOR: {author} CITATION:\n"

            files[key] += f"3B {interaction[0]} {interaction[1]} {interaction[2]}"
            leading_trim, trailing_trim = _trim_values(model, 3)
            files[key] += f" {leading_trim} {trailing_trim}"
            if knots_spacing_type == "uk":
                files[key] += " uk\n"
            elif knots_spacing_type == "nk":
                files[key] += " nk\n"
            else:
                raise ValueError(f"Supplied knot spacing type {knots_spacing_type}\n\
                                    is not a valid choice. Only uk or nk are valid types")

            files[key] += str(model.bspline_config.r_max_map[interaction][2]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][1]) \
                    + " " + str(model.bspline_config.r_max_map[interaction][0]) \
                    + " "

            files[key] += str(len(model.bspline_config.knots_map[interaction][2])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][1])) \
                    + " " + str(len(model.bspline_config.knots_map[interaction][0])) \
                    + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][2]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][1]]) + "\n"

            files[key] += " ".join(['{:.17g}'.format(v) for v in \
                    model.bspline_config.knots_map[interaction][0]]) + "\n"

            solutions = least_squares.arrange_coefficients(model.coefficients, \
                    model.bspline_config)

            decompressed = model.bspline_config.decompress_3B( \
                    solutions[(interaction[0], interaction[1],interaction[2])], \
                    (interaction[0], interaction[1],interaction[2]))

            files[key] += str(decompressed.shape[0]) + " " \
                    + str(decompressed.shape[1]) + " " \
                    + str(decompressed.shape[2]) + "\n"

            for i in range(decompressed.shape[0]):
                for j in range(decompressed.shape[1]):
                    files[key] += ' '.join(map(str, decompressed[i,j]))
                    files[key] += "\n"

            files[key] += "#\n"

    solutions = least_squares.arrange_coefficients(model.coefficients,
                                                   model.bspline_config)

    if emit_density:
        config = model.bspline_config
        if not config.density_enabled:
            raise ValueError("DENSITY emission requested for a model without density.")
        for interaction in config.density_interactions:
            coefficients = np.asarray(solutions[interaction])
            if len(coefficients) != 1:
                raise ValueError(f"DENSITY interaction {interaction} must have one coefficient.")
            _, center, neighbor = interaction
            key = "density_" + "_".join(str(value) for value in interaction[1:])
            files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
            files[key] += f"AUTHOR: {author} CITATION:\n"
            files[key] += f"DENSITY {center} {neighbor}\n"
            files[key] += (
                f"{config.density_r_min_map[interaction]:.17g} "
                f"{config.density_r_max_map[interaction]:.17g} "
                f"{config.density_power_map[interaction]:.17g} "
                f"{coefficients[0]:.17g}\n"
            )
            files[key] += "#\n"

    if emit_dipole:
        config = model.bspline_config
        if not config.dipole_enabled:
            raise ValueError("DIPOLE emission requested for a model without dipole.")
        for interaction in config.dipole_interactions:
            coefficients = np.asarray(solutions[interaction])
            if len(coefficients) != 1:
                raise ValueError(f"DIPOLE interaction {interaction} must have one coefficient.")
            _, center, neighbor = interaction
            key = "dipole_" + "_".join(str(value) for value in interaction[1:])
            files[key] = f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
            files[key] += f"AUTHOR: {author} CITATION:\n"
            files[key] += f"DIPOLE {center} {neighbor}\n"
            files[key] += (
                f"{config.dipole_r_min_map[interaction]:.17g} "
                f"{config.dipole_r_max_map[interaction]:.17g} "
                f"{config.dipole_power_map[interaction]:.17g} "
                f"{coefficients[0]:.17g}\n"
            )
            files[key] += "#\n"

    if emit_density_modulated_2b:
        config = model.bspline_config
        if not config.density_modulated_2b_enabled:
            raise ValueError("2RHO emission requested for a model without 2rho.")
        for interaction in config.density_modulated_2b_interactions:
            pair = config.density_modulated_2b_pair_map[interaction]
            alpha = config.density_modulated_2b_alpha_map[interaction]
            leading_trim, trailing_trim = _trim_values(model, 2)
            knots = config.knots_map[pair]
            coefficients = np.asarray(solutions[interaction])
            orientations = [pair] if pair[0] == pair[1] else [pair, pair[::-1]]
            for center, neighbor in orientations:
                density_interaction = ("density", center, neighbor)
                if density_interaction not in config.density_modulated_2b_r_max_map:
                    raise ValueError(
                        f"Missing density metadata for {density_interaction}."
                    )
                key = f"2rho_{alpha:g}_{center}_{neighbor}"
                lines = [
                    f"#UF3 POT UNITS: {lammps_units} DATE: {current_datetime} "
                    f"AUTHOR: {author} CITATION:",
                    f"2RHO {center} {neighbor} {alpha:.17g} "
                    f"{leading_trim} {trailing_trim} {knots_spacing_type}",
                    f"{config.r_max_map[pair]:.17g} {len(knots)}",
                    " ".join(f"{value:.17g}" for value in knots),
                    " ".join([
                        f"{config.density_modulated_2b_r_min_map[density_interaction]:.17g}",
                        f"{config.density_modulated_2b_cutoff_on_map[density_interaction]:.17g}",
                        f"{config.density_modulated_2b_r_max_map[density_interaction]:.17g}",
                        f"{config.density_modulated_2b_power_map[density_interaction]:.17g}",
                        f"{config.density_modulated_2b_rho_bulk_map[density_interaction]:.17g}",
                        f"{config.density_modulated_2b_weight_map[density_interaction]:.17g}",
                        config.density_modulated_2b_cutoff_mode,
                        config.density_modulated_2b_modulation,
                    ]),
                    str(len(coefficients)),
                    " ".join(f"{value:.17g}" for value in coefficients),
                    "#",
                ]
                files[key] = "\n".join(lines) + "\n"

    with open(pot_dir +"/"+ uf3_lammps_pot_name, "w") as fp:
        for k, v in files.items():
            fp.write(v)

if __name__ == "__main__":
    main()
