"""
Density-style feature helpers.

The density term is an optional linear feature block.  For each ordered
``(center element, neighbor element)`` interaction, it sums inverse-distance
weights from every center atom to matching neighbors inside a cutoff.
"""

from typing import Dict, List, Tuple

import numpy as np
import ase
from ase import symbols as ase_symbols

from uf3.representation import angles
from uf3.representation import distances


DensityInteraction = Tuple


def get_density_interactions(element_list: List[str]) -> List[DensityInteraction]:
    """Return ordered center-neighbor density interaction keys."""
    return [("density", str(center), str(neighbor))
            for center in element_list
            for neighbor in element_list]


def get_dipole_interactions(element_list: List[str]) -> List[DensityInteraction]:
    """Return ordered center-neighbor dipole-magnitude interaction keys."""
    return [("dipole", str(center), str(neighbor))
            for center in element_list
            for neighbor in element_list]


def get_shell_density_interactions(element_list: List[str],
                                   n_shells: int
                                   ) -> List[DensityInteraction]:
    """Return ordered center-neighbor shell-density interaction keys."""
    return [("shell_density", int(shell), str(center), str(neighbor))
            for shell in range(n_shells)
            for center in element_list
            for neighbor in element_list]


def get_density_moment_interactions(element_list: List[str],
                                    moments: List[float]
                                    ) -> List[DensityInteraction]:
    """Return ordered local density-moment interaction keys."""
    return [("density_moment", float(moment), str(center), str(neighbor))
            for moment in moments
            for center in element_list
            for neighbor in element_list]


def get_composition_interactions(element_list: List[str]
                                 ) -> List[DensityInteraction]:
    """Return ordered normalized local-composition interaction keys."""
    return [("local_composition", str(center), str(neighbor))
            for center in element_list
            for neighbor in element_list]


def get_density_modulated_pair_interactions(
        pair_interactions: List[Tuple[str, str]],
        alphas: List[float],
        ) -> List[DensityInteraction]:
    """Return density-modulated two-body interaction keys."""
    interactions = []
    for alpha in alphas:
        alpha_label = _format_alpha_label(alpha)
        for pair in pair_interactions:
            interactions.append(("density_modulated_2b", alpha_label, *pair))
    return interactions


def get_density_modulated_trio_interactions(
        trio_interactions: List[Tuple[str, str, str]],
        alphas: List[float],
        ) -> List[DensityInteraction]:
    """Return density-modulated three-body interaction keys."""
    interactions = []
    for alpha in alphas:
        alpha_label = _format_alpha_label(alpha)
        for trio in trio_interactions:
            interactions.append(("density_modulated_3b", alpha_label, *trio))
    return interactions


def get_dipole_modulated_pair_interactions(
        pair_interactions: List[Tuple[str, str]],
        ) -> List[DensityInteraction]:
    """Return dipole-modulated two-body interaction keys."""
    return [("dipole_modulated_2b", *pair) for pair in pair_interactions]


def _numbers_from_interaction(interaction: DensityInteraction) -> Tuple[int, int]:
    center, neighbor = interaction[-2:]
    center_z, neighbor_z = ase_symbols.symbols2numbers([center, neighbor])
    return center_z, neighbor_z


def _pair_density_by_center(distance_matrix: np.ndarray,
                            geo_composition: np.ndarray,
                            sup_composition: np.ndarray,
                            center_z: int,
                            neighbor_z: int,
                            r_min: float,
                            r_max: float,
                            power: float,
                            n_atoms: int,
                            cutoff_on: float | None = None,
                            cutoff_mode: str = "hard"):
    """Return pair indices and local per-center density for one interaction."""
    center_mask = geo_composition[:, None] == center_z
    neighbor_mask = sup_composition[None, :] == neighbor_z
    cut_mask = (distance_matrix > max(r_min, 0.0)) & (distance_matrix < r_max)
    mask = center_mask & neighbor_mask & cut_mask
    if not np.any(mask):
        return None, None, None
    i_where, j_where = np.where(mask)
    rij = distance_matrix[i_where, j_where]
    envelope, _ = cutoff_envelope(rij, cutoff_on, r_max, cutoff_mode)
    rho = np.zeros(n_atoms)
    np.add.at(rho, i_where, rij ** (-power) * envelope)
    return i_where, j_where, rho


def _format_alpha_label(alpha: float) -> str:
    return f"a{float(alpha):g}".replace("-", "m").replace(".", "p")


def _density_modulation(x: np.ndarray,
                        alpha: float,
                        derivative: bool = False,
                        eps: float = 1e-12,
                        mode: str = "conservative") -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), eps)
    mode = str(mode)
    if mode in {"conservative", "quadratic_reference"}:
        if derivative:
            return alpha * (x ** (alpha - 1.0) - 1.0)
        return x ** alpha - 1.0 - alpha * (x - 1.0)
    if mode in {"xalpha_minus_1", "power_minus_one", "production"}:
        if derivative:
            return alpha * x ** (alpha - 1.0)
        return x ** alpha - 1.0
    if mode in {"xalpha", "power", "full", "density_full"}:
        if derivative:
            return alpha * x ** (alpha - 1.0)
        return x ** alpha
    raise ValueError(f"Unknown density modulation mode: {mode}")


def cutoff_envelope(r: np.ndarray,
                    r_on: float | None,
                    r_cut: float,
                    mode: str = "hard") -> Tuple[np.ndarray, np.ndarray]:
    """Return smooth cutoff envelope and d(envelope)/dr.

    ``hard`` preserves the historical behavior.  ``cosine`` is C1 smooth at
    both ends of the switching interval.  ``quintic`` is C2 smooth and is the
    preferred low-cost polynomial taper for density features.
    """
    r = np.asarray(r, dtype=float)
    mode = str(mode or "hard").lower()
    if mode in {"hard", "none", "step"}:
        return np.ones_like(r), np.zeros_like(r)
    if r_on is None:
        raise ValueError("Smooth density cutoff requires r_on.")
    r_on = float(r_on)
    r_cut = float(r_cut)
    if not r_on < r_cut:
        raise ValueError(
            f"Smooth density cutoff needs r_on < r_cut; got {r_on} >= {r_cut}."
        )

    envelope = np.ones_like(r)
    derivative = np.zeros_like(r)
    switch = (r > r_on) & (r < r_cut)
    beyond = r >= r_cut
    envelope[beyond] = 0.0
    if not np.any(switch):
        return envelope, derivative

    width = r_cut - r_on
    t = (r[switch] - r_on) / width
    if mode in {"cosine", "c1"}:
        envelope[switch] = 0.5 * (1.0 + np.cos(np.pi * t))
        derivative[switch] = -0.5 * np.pi * np.sin(np.pi * t) / width
    elif mode in {"quintic", "smoothstep5", "c2", "legendre"}:
        # Quintic smoothstep: 1 - 10 t^3 + 15 t^4 - 6 t^5.
        # This is a compact polynomial taper with zero first and second
        # derivatives at both endpoints.
        envelope[switch] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        derivative[switch] = (
            -30.0 * t ** 2 + 60.0 * t ** 3 - 30.0 * t ** 4
        ) / width
    else:
        raise ValueError(f"Unknown density cutoff mode: {mode}")
    return envelope, derivative


def local_density_by_interaction(
        geom: ase.Atoms,
        density_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        cutoff_mode: str = "hard",
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-atom density and per-atom bulk-density normalization."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())

    rho = np.zeros(n_atoms)
    rho_bulk = np.zeros(n_atoms)
    for interaction in density_interactions:
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        cutoff_on = None
        if cutoff_on_map is not None:
            cutoff_on = cutoff_on_map.get(interaction, None)
        result = _pair_density_by_center(
            distance_matrix,
            geo_composition,
            sup_composition,
            center_z,
            neighbor_z,
            r_min_map[interaction],
            r_max_map[interaction],
            power_map[interaction],
            n_atoms,
            cutoff_on=cutoff_on,
            cutoff_mode=cutoff_mode,
        )
        center_rows = geo_composition == center_z
        if result[0] is not None:
            _, _, rho_part = result
            rho += rho_part
        rho_bulk[center_rows] += 1.0
    return rho, rho_bulk


def _bulk_density_by_atom(geom: ase.Atoms,
                          density_interactions: List[DensityInteraction],
                          rho_bulk_map: Dict[DensityInteraction, float],
                          eps: float = 1e-12) -> np.ndarray:
    geo_composition = np.array(geom.get_atomic_numbers())
    rho_bulk = np.zeros(len(geom))
    for interaction in density_interactions:
        center_z, _ = _numbers_from_interaction(interaction)
        center_rows = geo_composition == center_z
        rho_bulk[center_rows] += float(rho_bulk_map[interaction])
    return np.maximum(rho_bulk, eps)


def density_modulation_state(
        geom: ase.Atoms,
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        rho_bulk_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        density_cutoff_mode: str = "hard",
        supercell: ase.Atoms = None,
        eps: float = 1e-12,
        ) -> Dict[str, np.ndarray]:
    """Precompute local density, bulk normalization, and normalized density."""
    rho, _ = local_density_by_interaction(
        geom,
        density_interactions,
        density_r_min_map,
        density_r_max_map,
        density_power_map,
        supercell=supercell,
        cutoff_on_map=density_cutoff_on_map,
        cutoff_mode=density_cutoff_mode)
    rho_bulk = _bulk_density_by_atom(geom, density_interactions, rho_bulk_map, eps)
    return {
        "rho": rho,
        "rho_bulk": rho_bulk,
        "x": rho / rho_bulk,
    }


def _eps_by_atom(geom: ase.Atoms,
                 density_interactions: List[DensityInteraction],
                 eps_map: Dict[DensityInteraction, float],
                 default_eps: float = 1e-8) -> np.ndarray:
    """Return per-center eps for normalized local asymmetry."""
    geo_composition = np.array(geom.get_atomic_numbers())
    eps_values = np.full(len(geom), float(default_eps))
    for interaction in density_interactions:
        center_z, _ = _numbers_from_interaction(interaction)
        center_rows = geo_composition == center_z
        eps_values[center_rows] = np.maximum(
            eps_values[center_rows],
            float(eps_map.get(interaction, default_eps)))
    return eps_values


def local_asymmetry_by_interaction(
        geom: ase.Atoms,
        dipole_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        eps_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        default_eps: float = 1e-8,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rho, dipole vector, and normalized asymmetry per atom."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()

    rho = np.zeros(n_atoms)
    p_vec = np.zeros((n_atoms, 3))
    for interaction in dipole_interactions:
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        r_min = max(r_min_map[interaction], 0.0)
        r_max = r_max_map[interaction]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = center_mask & neighbor_mask & cut_mask
        if not np.any(mask):
            continue
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        delta = sup_positions[j_where] - geo_positions[i_where]
        unit = delta / rij[:, None]
        weight = rij ** (-power_map[interaction])
        np.add.at(rho, i_where, weight)
        np.add.at(p_vec, i_where, weight[:, None] * unit)

    eps_values = _eps_by_atom(
        geom, dipole_interactions, eps_map, default_eps=default_eps)
    denom = rho + eps_values
    p2 = np.einsum("ij,ij->i", p_vec, p_vec)
    a2 = p2 / (denom ** 2)
    return rho, p_vec, a2, eps_values


def _weighted_basis_sum(points: np.ndarray,
                        weights: np.ndarray,
                        basis_functions: List,
                        n_lead: int = 0,
                        n_trail: int = 0,
                        derivative: bool = False) -> np.ndarray:
    n_splines = len(basis_functions)
    features = np.zeros(n_splines)
    nu = 1 if derivative else 0
    for idx in range(n_lead, n_splines - n_trail):
        values = basis_functions[idx](points, nu=nu)
        values[np.isnan(values)] = 0.0
        features[idx] = np.sum(values * weights)
    return features


def density_modulated_pair_values(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        pair_map: Dict[DensityInteraction, Tuple[str, str]],
        alpha_map: Dict[DensityInteraction, float],
        pair_r_min_map: Dict[Tuple[str, str], float],
        pair_r_max_map: Dict[Tuple[str, str], float],
        basis_functions: Dict[Tuple[str, str], List],
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        rho_bulk_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        density_cutoff_mode: str = "hard",
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-12,
        modulation: str = "conservative",
        density_state: Dict[str, np.ndarray] | None = None,
        ) -> np.ndarray:
    """Generate density-modulated two-body energy features."""
    if supercell is None:
        supercell = geom
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    if density_state is None:
        density_state = density_modulation_state(
            geom,
            density_interactions,
            density_r_min_map,
            density_r_max_map,
            density_power_map,
            rho_bulk_map,
            density_cutoff_on_map,
            density_cutoff_mode,
            supercell=supercell,
            eps=eps)
    x = density_state["x"]

    blocks = []
    for interaction in modulated_interactions:
        pair = pair_map[interaction]
        pair_numbers = ase_symbols.symbols2numbers(pair)
        comp_mask = distances.mask_matrix_by_pair_interaction(
            pair_numbers, geo_composition, sup_composition)
        r_min = max(pair_r_min_map[pair], 0.0)
        r_max = pair_r_max_map[pair]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = comp_mask & cut_mask
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        alpha = alpha_map[interaction]
        weights = _density_modulation(
            x[i_where], alpha, eps=eps, mode=modulation)
        block = _weighted_basis_sum(rij,
                                    weights,
                                    basis_functions[pair],
                                    n_lead=n_lead,
                                    n_trail=n_trail)
        blocks.append(block)
    return np.concatenate(blocks) if blocks else np.array([])


def density_modulated_pair_force_features(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        pair_map: Dict[DensityInteraction, Tuple[str, str]],
        alpha_map: Dict[DensityInteraction, float],
        pair_r_min_map: Dict[Tuple[str, str], float],
        pair_r_max_map: Dict[Tuple[str, str], float],
        basis_functions: Dict[Tuple[str, str], List],
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        rho_bulk_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        density_cutoff_mode: str = "hard",
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-12,
        modulation: str = "conservative",
        density_state: Dict[str, np.ndarray] | None = None,
        ) -> np.ndarray:
    """Generate ``-d/dR`` force features for density-modulated pairs."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()

    if density_state is None:
        density_state = density_modulation_state(
            geom,
            density_interactions,
            density_r_min_map,
            density_r_max_map,
            density_power_map,
            rho_bulk_map,
            density_cutoff_on_map,
            density_cutoff_mode,
            supercell=supercell,
            eps=eps)
    rho_bulk = density_state["rho_bulk"]
    x = density_state["x"]
    density_pairs = _density_derivative_pairs(
        geom,
        supercell,
        density_interactions,
        density_r_min_map,
        density_r_max_map,
        density_power_map,
        density_cutoff_on_map,
        density_cutoff_mode,
        distance_matrix=distance_matrix,
        coords=sup_positions)

    feature_blocks = []
    for interaction in modulated_interactions:
        pair = pair_map[interaction]
        pair_numbers = ase_symbols.symbols2numbers(pair)
        comp_mask = distances.mask_matrix_by_pair_interaction(
            pair_numbers, geo_composition, sup_composition)
        r_min = max(pair_r_min_map[pair], 0.0)
        r_max = pair_r_max_map[pair]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = comp_mask & cut_mask
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        n_splines = len(basis_functions[pair])
        block = np.zeros((n_atoms, 3, n_splines))
        if len(rij) == 0:
            feature_blocks.append(block)
            continue

        alpha = alpha_map[interaction]
        psi = _density_modulation(
            x[i_where], alpha, eps=eps, mode=modulation)
        psi_prime = _density_modulation(
            x, alpha, derivative=True, eps=eps, mode=modulation) / rho_bulk
        drij_dR = distances.compute_direction_cosines(
            sup_positions, distance_matrix, i_where, j_where, n_atoms)

        for spline_idx in range(n_lead, n_splines - n_trail):
            values = basis_functions[pair][spline_idx](rij, nu=0)
            derivs = basis_functions[pair][spline_idx](rij, nu=1)
            values[np.isnan(values)] = 0.0
            derivs[np.isnan(derivs)] = 0.0

            direct_weights = psi * derivs
            block[:, :, spline_idx] -= np.sum(
                drij_dR * direct_weights[None, None, :], axis=-1)

            g = np.zeros(n_atoms)
            np.add.at(g, i_where, psi_prime[i_where] * values)
            for d_i, _, d_drij_dR, chain_base in density_pairs:
                chain_weights = g[d_i] * chain_base
                block[:, :, spline_idx] += np.sum(
                    d_drij_dR * chain_weights[None, None, :], axis=-1)

        feature_blocks.append(block)
    return np.concatenate(feature_blocks, axis=2) if feature_blocks else \
        np.zeros((n_atoms, 3, 0))


def _density_derivative_pairs(
        geom: ase.Atoms,
        supercell: ase.Atoms,
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None,
        density_cutoff_mode: str,
        distance_matrix: np.ndarray | None = None,
        coords: np.ndarray | None = None,
        ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Precompute pairwise ``-d rho / dR`` factors for density chain rules."""
    n_atoms = len(geom)
    if distance_matrix is None:
        distance_matrix = distances.get_distance_matrix(geom, supercell)
    if coords is None:
        coords = supercell.get_positions()
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())

    pairs = []
    for density_interaction in density_interactions:
        center_z, neighbor_z = _numbers_from_interaction(density_interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        d_r_min = max(density_r_min_map[density_interaction], 0.0)
        d_r_max = density_r_max_map[density_interaction]
        d_cut = ((distance_matrix > d_r_min) &
                 (distance_matrix < d_r_max))
        d_mask = center_mask & neighbor_mask & d_cut
        if not np.any(d_mask):
            continue
        d_i, d_j = np.where(d_mask)
        d_rij = distance_matrix[d_i, d_j]
        d_drij_dR = distances.compute_direction_cosines(
            coords, distance_matrix, d_i, d_j, n_atoms)
        power = density_power_map[density_interaction]
        cutoff_on = None
        if density_cutoff_on_map is not None:
            cutoff_on = density_cutoff_on_map.get(density_interaction, None)
        envelope, envelope_deriv = cutoff_envelope(
            d_rij, cutoff_on, d_r_max, density_cutoff_mode)
        chain_base = (
            power * d_rij ** (-power - 1.0) * envelope
            - d_rij ** (-power) * envelope_deriv
        )
        pairs.append((d_i, d_j, d_drij_dR, chain_base))
    return pairs


def density_modulated_trio_energy_grids(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        trio_map: Dict[DensityInteraction, Tuple[str, str, str]],
        alpha_map: Dict[DensityInteraction, float],
        knot_sets_map: Dict[Tuple[str, str, str], List[np.ndarray]],
        basis_functions_map: Dict[Tuple[str, str, str], List],
        trio_hashes: List,
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        rho_bulk_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        density_cutoff_mode: str = "hard",
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-12,
        modulation: str = "conservative",
        density_state: Dict[str, np.ndarray] | None = None,
        ) -> List[np.ndarray]:
    """Generate density-modulated three-body energy grids."""
    if supercell is None:
        supercell = geom
    n_interactions = len(modulated_interactions)
    trio_list = [trio_map[interaction] for interaction in modulated_interactions]
    knot_sets = [knot_sets_map[trio] for trio in trio_list]
    basis_functions = [basis_functions_map[trio] for trio in trio_list]
    sup_comp = supercell.get_atomic_numbers()
    L, M, N = angles.coefficient_counts_from_knots(knot_sets)
    grids = [np.zeros((L[i], M[i], N[i])) for i in range(n_interactions)]

    if density_state is None:
        density_state = density_modulation_state(
            geom,
            density_interactions,
            density_r_min_map,
            density_r_max_map,
            density_power_map,
            rho_bulk_map,
            density_cutoff_on_map,
            density_cutoff_mode,
            supercell=supercell,
            eps=eps)
    x = density_state["x"]

    dist_matrix, i_where, j_where = angles.identify_ij(
        geom, knot_sets, supercell)
    if len(i_where) == 0:
        return grids
    i_values, i_groups = angles.group_idx_by_center(i_where, j_where)

    hashes = np.asarray(trio_hashes)
    for i_group, i_value in zip(i_groups, i_values):
        if i_value >= len(geom):
            continue
        triplet_batch = angles.generate_triplets(
            i_value, i_group, sup_comp, hashes, dist_matrix, knot_sets,
            len(geom))
        for interaction_idx, interaction_data in enumerate(triplet_batch):
            if interaction_data is None:
                continue
            r_l, r_m, r_n, _ = interaction_data
            tuples_3b, idx_lmn = angles.evaluate_triplet_distances(
                r_l,
                r_m,
                r_n,
                basis_functions[interaction_idx],
                knot_sets[interaction_idx],
                n_lead=n_lead,
                n_trail=n_trail)
            alpha = alpha_map[modulated_interactions[interaction_idx]]
            weight = float(_density_modulation(
                x[i_value], alpha, eps=eps, mode=modulation))
            grids[interaction_idx] += weight * angles.arrange_3b(
                tuples_3b,
                idx_lmn,
                L[interaction_idx],
                M[interaction_idx],
                N[interaction_idx])
    return grids


def density_modulated_trio_force_grids(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        trio_map: Dict[DensityInteraction, Tuple[str, str, str]],
        alpha_map: Dict[DensityInteraction, float],
        knot_sets_map: Dict[Tuple[str, str, str], List[np.ndarray]],
        basis_functions_map: Dict[Tuple[str, str, str], List],
        trio_hashes: List,
        density_interactions: List[DensityInteraction],
        density_r_min_map: Dict[DensityInteraction, float],
        density_r_max_map: Dict[DensityInteraction, float],
        density_power_map: Dict[DensityInteraction, float],
        rho_bulk_map: Dict[DensityInteraction, float],
        density_cutoff_on_map: Dict[DensityInteraction, float] | None = None,
        density_cutoff_mode: str = "hard",
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-12,
        modulation: str = "conservative",
        density_state: Dict[str, np.ndarray] | None = None,
        ) -> List[List[List[np.ndarray]]]:
    """Generate ``-d/dR`` grids for density-modulated three-body features."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    n_interactions = len(modulated_interactions)
    trio_list = [trio_map[interaction] for interaction in modulated_interactions]
    knot_sets = [knot_sets_map[trio] for trio in trio_list]
    basis_functions = [basis_functions_map[trio] for trio in trio_list]
    sup_comp = supercell.get_atomic_numbers()
    L, M, N = angles.coefficient_counts_from_knots(knot_sets)
    force_grids = [[[np.zeros((L[i], M[i], N[i])),
                     np.zeros((L[i], M[i], N[i])),
                     np.zeros((L[i], M[i], N[i]))]
                    for _ in range(n_atoms)]
                   for i in range(n_interactions)]

    if density_state is None:
        density_state = density_modulation_state(
            geom,
            density_interactions,
            density_r_min_map,
            density_r_max_map,
            density_power_map,
            rho_bulk_map,
            density_cutoff_on_map,
            density_cutoff_mode,
            supercell=supercell,
            eps=eps)
    rho_bulk = density_state["rho_bulk"]
    x = density_state["x"]

    coords, matrix, x_where, y_where = angles.identify_ij(
        geom, knot_sets, supercell, square=True)
    if len(x_where) == 0:
        return force_grids
    density_pairs = _density_derivative_pairs(
        geom,
        supercell,
        density_interactions,
        density_r_min_map,
        density_r_max_map,
        density_power_map,
        density_cutoff_on_map,
        density_cutoff_mode,
        distance_matrix=matrix[:n_atoms, :],
        coords=coords)
    i_values, i_groups = angles.group_idx_by_center(x_where, y_where)

    hashes = np.asarray(trio_hashes)
    for i_group, i_value in zip(i_groups, i_values):
        if i_value >= n_atoms:
            continue
        triplet_batch = angles.generate_triplets(
            i_value, i_group, sup_comp, hashes, matrix, knot_sets, n_atoms)
        for interaction_idx, interaction_data in enumerate(triplet_batch):
            if interaction_data is None:
                continue
            r_l, r_m, r_n, ituples = interaction_data
            tuples_3b, idx_lmn = angles.evaluate_triplet_derivatives(
                r_l,
                r_m,
                r_n,
                basis_functions[interaction_idx],
                knot_sets[interaction_idx],
                n_lead=n_lead,
                n_trail=n_trail)
            drij_dr = distances.compute_direction_cosines(
                coords, matrix, ituples[:, 0], ituples[:, 1], n_atoms)
            drik_dr = distances.compute_direction_cosines(
                coords, matrix, ituples[:, 0], ituples[:, 2], n_atoms)
            drjk_dr = distances.compute_direction_cosines(
                coords, matrix, ituples[:, 1], ituples[:, 2], n_atoms)
            direct_grids = angles.arrange_deriv_3b(
                tuples_3b,
                idx_lmn,
                drij_dr,
                drik_dr,
                drjk_dr,
                L[interaction_idx],
                M[interaction_idx],
                N[interaction_idx])
            interaction = modulated_interactions[interaction_idx]
            alpha = alpha_map[interaction]
            psi = float(_density_modulation(
                x[i_value], alpha, eps=eps, mode=modulation))
            psi_prime = float(_density_modulation(
                x[i_value], alpha, derivative=True,
                eps=eps, mode=modulation) / rho_bulk[i_value])

            value_tuples = tuples_3b[:, :3]
            value_grid = angles.arrange_3b(
                value_tuples,
                idx_lmn,
                L[interaction_idx],
                M[interaction_idx],
                N[interaction_idx])
            for a in range(n_atoms):
                for c in range(3):
                    force_grids[interaction_idx][a][c] -= (
                        psi * direct_grids[a][c])

            if psi_prime == 0.0 or not np.any(value_grid):
                continue
            density_grid = psi_prime * value_grid
            for d_i, _, d_drij_dR, chain_base in density_pairs:
                center_mask = d_i == i_value
                if not np.any(center_mask):
                    continue
                weights = chain_base[center_mask]
                derivs = d_drij_dR[:, :, center_mask]
                prefactor = np.sum(derivs * weights[None, None, :], axis=2)
                for a in range(n_atoms):
                    for c in range(3):
                        if prefactor[a, c] != 0.0:
                            force_grids[interaction_idx][a][c] += (
                                prefactor[a, c] * density_grid)
    return force_grids


def dipole_modulated_pair_values(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        pair_map: Dict[DensityInteraction, Tuple[str, str]],
        pair_r_min_map: Dict[Tuple[str, str], float],
        pair_r_max_map: Dict[Tuple[str, str], float],
        basis_functions: Dict[Tuple[str, str], List],
        dipole_interactions: List[DensityInteraction],
        dipole_r_min_map: Dict[DensityInteraction, float],
        dipole_r_max_map: Dict[DensityInteraction, float],
        dipole_power_map: Dict[DensityInteraction, float],
        dipole_eps_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-8,
        ) -> np.ndarray:
    """Generate dipole-modulated two-body energy features."""
    if supercell is None:
        supercell = geom
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    _, _, a2, _ = local_asymmetry_by_interaction(
        geom,
        dipole_interactions,
        dipole_r_min_map,
        dipole_r_max_map,
        dipole_power_map,
        dipole_eps_map,
        supercell=supercell,
        default_eps=eps)

    blocks = []
    for interaction in modulated_interactions:
        pair = pair_map[interaction]
        pair_numbers = ase_symbols.symbols2numbers(pair)
        comp_mask = distances.mask_matrix_by_pair_interaction(
            pair_numbers, geo_composition, sup_composition)
        r_min = max(pair_r_min_map[pair], 0.0)
        r_max = pair_r_max_map[pair]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = comp_mask & cut_mask
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        block = _weighted_basis_sum(rij,
                                    a2[i_where],
                                    basis_functions[pair],
                                    n_lead=n_lead,
                                    n_trail=n_trail)
        blocks.append(block)
    return np.concatenate(blocks) if blocks else np.array([])


def dipole_modulated_pair_force_features(
        geom: ase.Atoms,
        modulated_interactions: List[DensityInteraction],
        pair_map: Dict[DensityInteraction, Tuple[str, str]],
        pair_r_min_map: Dict[Tuple[str, str], float],
        pair_r_max_map: Dict[Tuple[str, str], float],
        basis_functions: Dict[Tuple[str, str], List],
        dipole_interactions: List[DensityInteraction],
        dipole_r_min_map: Dict[DensityInteraction, float],
        dipole_r_max_map: Dict[DensityInteraction, float],
        dipole_power_map: Dict[DensityInteraction, float],
        dipole_eps_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        n_lead: int = 0,
        n_trail: int = 0,
        eps: float = 1e-8,
        ) -> np.ndarray:
    """Generate ``-d/dR`` force features for dipole-modulated pairs."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    geo_positions = geom.get_positions()
    sup_positions = supercell.get_positions()
    rho, p_vec, a2, eps_values = local_asymmetry_by_interaction(
        geom,
        dipole_interactions,
        dipole_r_min_map,
        dipole_r_max_map,
        dipole_power_map,
        dipole_eps_map,
        supercell=supercell,
        default_eps=eps)
    denom = rho + eps_values
    p2 = np.einsum("ij,ij->i", p_vec, p_vec)

    feature_blocks = []
    eye = np.eye(3)
    for interaction in modulated_interactions:
        pair = pair_map[interaction]
        pair_numbers = ase_symbols.symbols2numbers(pair)
        comp_mask = distances.mask_matrix_by_pair_interaction(
            pair_numbers, geo_composition, sup_composition)
        r_min = max(pair_r_min_map[pair], 0.0)
        r_max = pair_r_max_map[pair]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = comp_mask & cut_mask
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        n_splines = len(basis_functions[pair])
        block = np.zeros((n_atoms, 3, n_splines))
        if len(rij) == 0:
            feature_blocks.append(block)
            continue

        delta = sup_positions[j_where] - geo_positions[i_where]
        unit = delta / rij[:, None]
        for spline_idx in range(n_lead, n_splines - n_trail):
            values = basis_functions[pair][spline_idx](rij, nu=0)
            derivs = basis_functions[pair][spline_idx](rij, nu=1)
            values[np.isnan(values)] = 0.0
            derivs[np.isnan(derivs)] = 0.0

            direct_vecs = a2[i_where, None] * derivs[:, None] * unit
            np.add.at(block[:, :, spline_idx], i_where, direct_vecs)
            real_j = j_where < n_atoms
            if np.any(real_j):
                np.add.at(block[:, :, spline_idx],
                          j_where[real_j],
                          -direct_vecs[real_j])

            g = np.zeros(n_atoms)
            np.add.at(g, i_where, values)
            for dipole_interaction in dipole_interactions:
                center_z, neighbor_z = _numbers_from_interaction(
                    dipole_interaction)
                center_mask = geo_composition[:, None] == center_z
                neighbor_mask = sup_composition[None, :] == neighbor_z
                d_r_min = max(dipole_r_min_map[dipole_interaction], 0.0)
                d_r_max = dipole_r_max_map[dipole_interaction]
                d_cut = ((distance_matrix > d_r_min) &
                         (distance_matrix < d_r_max))
                d_mask = center_mask & neighbor_mask & d_cut
                if not np.any(d_mask):
                    continue
                d_i, d_j = np.where(d_mask)
                d_rij = distance_matrix[d_i, d_j]
                d_delta = sup_positions[d_j] - geo_positions[d_i]
                d_unit = d_delta / d_rij[:, None]
                power = dipole_power_map[dipole_interaction]
                weight = d_rij ** (-power)
                weight_prime = -power * d_rij ** (-power - 1.0)
                projection = (eye[None, :, :] -
                              d_unit[:, :, None] * d_unit[:, None, :])
                jacobian = (
                    weight_prime[:, None, None] *
                    d_unit[:, :, None] * d_unit[:, None, :] +
                    (weight / d_rij)[:, None, None] * projection
                )
                grad_p = 2.0 * np.einsum(
                    "nij,ni->nj", jacobian, p_vec[d_i]) / (
                    denom[d_i] ** 2)[:, None]
                grad_rho = (
                    2.0 * p2[d_i] / (denom[d_i] ** 3) *
                    weight_prime)[:, None] * d_unit
                grad_a2 = grad_p - grad_rho
                chain_vecs = g[d_i, None] * grad_a2
                np.add.at(block[:, :, spline_idx], d_i, chain_vecs)
                real_d_j = d_j < n_atoms
                if np.any(real_d_j):
                    np.add.at(block[:, :, spline_idx],
                              d_j[real_d_j],
                              -chain_vecs[real_d_j])

        feature_blocks.append(block)
    return np.concatenate(feature_blocks, axis=2) if feature_blocks else \
        np.zeros((n_atoms, 3, 0))


def density_values_by_interaction(geom: ase.Atoms,
                                  density_interactions: List[DensityInteraction],
                                  r_min_map: Dict[DensityInteraction, float],
                                  r_max_map: Dict[DensityInteraction, float],
                                  power_map: Dict[DensityInteraction, float],
                                  supercell: ase.Atoms = None,
                                  ) -> np.ndarray:
    """
    Compute summed density features for one configuration.

    Returns:
        vector: one scalar per density interaction.
    """
    if supercell is None:
        supercell = geom
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())

    values = np.zeros(len(density_interactions))
    for idx, interaction in enumerate(density_interactions):
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        r_min = max(r_min_map[interaction], 0.0)
        r_max = r_max_map[interaction]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = center_mask & neighbor_mask & cut_mask
        if np.any(mask):
            values[idx] = np.sum(distance_matrix[mask] ** (-power_map[interaction]))
    return values


def density_force_features(geom: ase.Atoms,
                           density_interactions: List[DensityInteraction],
                           r_min_map: Dict[DensityInteraction, float],
                           r_max_map: Dict[DensityInteraction, float],
                           power_map: Dict[DensityInteraction, float],
                           supercell: ase.Atoms = None,
                           ) -> np.ndarray:
    """
    Compute force features for density terms.

    The returned array has shape ``(n_atoms, 3, n_density_features)`` and
    contains ``-d rho / dR`` so a linear coefficient contributes directly to
    forces through ``F = X c``.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()

    features = np.zeros((n_atoms, 3, len(density_interactions)))
    for idx, interaction in enumerate(density_interactions):
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        r_min = max(r_min_map[interaction], 0.0)
        r_max = r_max_map[interaction]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = center_mask & neighbor_mask & cut_mask
        if not np.any(mask):
            continue
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        drij_dR = distances.compute_direction_cosines(sup_positions,
                                                      distance_matrix,
                                                      i_where,
                                                      j_where,
                                                      n_atoms)
        power = power_map[interaction]
        # rho = r^-p, so -d rho/dR = p * r^(-p-1) * dr/dR.
        weights = power * rij ** (-power - 1.0)
        features[:, :, idx] = np.sum(drij_dR * weights[None, None, :], axis=-1)
    return features


def dipole_values_by_interaction(geom: ase.Atoms,
                                 dipole_interactions: List[DensityInteraction],
                                 r_min_map: Dict[DensityInteraction, float],
                                 r_max_map: Dict[DensityInteraction, float],
                                 power_map: Dict[DensityInteraction, float],
                                 supercell: ase.Atoms = None,
                                 ) -> np.ndarray:
    """
    Compute local dipole-magnitude features for one configuration.

    For each ordered ``("dipole", center, neighbor)`` interaction, this
    returns ``sum_i |sum_j r_ij^-p rhat_ij|^2`` over center atoms ``i``.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()

    values = np.zeros(len(dipole_interactions))
    for idx, interaction in enumerate(dipole_interactions):
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        r_min = max(r_min_map[interaction], 0.0)
        r_max = r_max_map[interaction]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = center_mask & neighbor_mask & cut_mask
        if not np.any(mask):
            continue
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        delta = sup_positions[j_where] - geo_positions[i_where]
        weighted_vectors = delta * rij[:, None] ** (-power_map[interaction] - 1.0)
        dipoles = np.zeros((n_atoms, 3))
        np.add.at(dipoles, i_where, weighted_vectors)
        center_rows = geo_composition == center_z
        values[idx] = np.sum(dipoles[center_rows] ** 2)
    return values


def dipole_force_features(geom: ase.Atoms,
                          dipole_interactions: List[DensityInteraction],
                          r_min_map: Dict[DensityInteraction, float],
                          r_max_map: Dict[DensityInteraction, float],
                          power_map: Dict[DensityInteraction, float],
                          supercell: ase.Atoms = None,
                          ) -> np.ndarray:
    """
    Compute force features for dipole-magnitude terms.

    The returned array has shape ``(n_atoms, 3, n_dipole_features)`` and
    contains ``-dQ/dR`` for ``Q=sum_i |sum_j r_ij^-p rhat_ij|^2``.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()
    geo_positions = geom.get_positions()

    features = np.zeros((n_atoms, 3, len(dipole_interactions)))
    for idx, interaction in enumerate(dipole_interactions):
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        center_mask = geo_composition[:, None] == center_z
        neighbor_mask = sup_composition[None, :] == neighbor_z
        r_min = max(r_min_map[interaction], 0.0)
        r_max = r_max_map[interaction]
        cut_mask = (distance_matrix > r_min) & (distance_matrix < r_max)
        mask = center_mask & neighbor_mask & cut_mask
        if not np.any(mask):
            continue
        i_where, j_where = np.where(mask)
        rij = distance_matrix[i_where, j_where]
        power = power_map[interaction]
        delta = sup_positions[j_where] - geo_positions[i_where]
        weighted_vectors = delta * rij[:, None] ** (-power - 1.0)
        dipoles = np.zeros((n_atoms, 3))
        np.add.at(dipoles, i_where, weighted_vectors)

        s_identity = rij ** (-power - 1.0)
        s_outer = -(power + 1.0) * rij ** (-power - 3.0)
        for center_idx, neighbor_idx, delta_vec, s_i, s_o in zip(
                i_where, j_where, delta, s_identity, s_outer):
            local_dipole = dipoles[center_idx]
            jacobian_dot = s_i * local_dipole + s_o * delta_vec * np.dot(
                delta_vec, local_dipole)
            contribution = 2.0 * jacobian_dot
            features[center_idx, :, idx] += contribution
            if neighbor_idx < n_atoms:
                features[neighbor_idx, :, idx] -= contribution
    return features


def density_moment_values_by_interaction(
        geom: ase.Atoms,
        moment_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        ) -> np.ndarray:
    """
    Compute local density moments.

    For each ordered ``("density_moment", m, center, neighbor)`` feature,
    this returns ``sum_i rho_i(neighbor)^m`` over center atoms.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())

    values = np.zeros(len(moment_interactions))
    for idx, interaction in enumerate(moment_interactions):
        moment = float(interaction[1])
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        result = _pair_density_by_center(
            distance_matrix,
            geo_composition,
            sup_composition,
            center_z,
            neighbor_z,
            r_min_map[interaction],
            r_max_map[interaction],
            power_map[interaction],
            n_atoms,
        )
        if result[0] is None:
            continue
        _, _, rho = result
        center_rows = geo_composition == center_z
        values[idx] = np.sum(rho[center_rows] ** moment)
    return values


def density_moment_force_features(
        geom: ase.Atoms,
        moment_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        ) -> np.ndarray:
    """Return ``-d/dR`` features for local density moments."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()

    features = np.zeros((n_atoms, 3, len(moment_interactions)))
    for idx, interaction in enumerate(moment_interactions):
        moment = float(interaction[1])
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        result = _pair_density_by_center(
            distance_matrix,
            geo_composition,
            sup_composition,
            center_z,
            neighbor_z,
            r_min_map[interaction],
            r_max_map[interaction],
            power_map[interaction],
            n_atoms,
        )
        if result[0] is None:
            continue
        i_where, j_where, rho = result
        rij = distance_matrix[i_where, j_where]
        drij_dR = distances.compute_direction_cosines(sup_positions,
                                                      distance_matrix,
                                                      i_where,
                                                      j_where,
                                                      n_atoms)
        power = power_map[interaction]
        prefactor = moment * rho[i_where] ** (moment - 1.0)
        weights = prefactor * power * rij ** (-power - 1.0)
        features[:, :, idx] = np.sum(drij_dR * weights[None, None, :],
                                     axis=-1)
    return features


def local_composition_values_by_interaction(
        geom: ase.Atoms,
        composition_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        eps: float = 1e-12,
        ) -> np.ndarray:
    """
    Compute normalized local composition features.

    For each center atom, the neighbor-element density is divided by the
    total neighbor density before summing over centers of the requested type.
    """
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    neighbor_numbers = sorted(set(sup_composition.tolist()))

    values = np.zeros(len(composition_interactions))
    for idx, interaction in enumerate(composition_interactions):
        center_z, neighbor_z = _numbers_from_interaction(interaction)
        r_min = r_min_map[interaction]
        r_max = r_max_map[interaction]
        power = power_map[interaction]
        center_rows = geo_composition == center_z
        rho_target = np.zeros(n_atoms)
        rho_total = np.zeros(n_atoms)
        for z in neighbor_numbers:
            result = _pair_density_by_center(
                distance_matrix, geo_composition, sup_composition,
                center_z, z, r_min, r_max, power, n_atoms)
            if result[0] is None:
                continue
            _, _, rho = result
            rho_total += rho
            if z == neighbor_z:
                rho_target += rho
        valid = center_rows & (rho_total > eps)
        if np.any(valid):
            values[idx] = np.sum(rho_target[valid] / rho_total[valid])
    return values


def local_composition_force_features(
        geom: ase.Atoms,
        composition_interactions: List[DensityInteraction],
        r_min_map: Dict[DensityInteraction, float],
        r_max_map: Dict[DensityInteraction, float],
        power_map: Dict[DensityInteraction, float],
        supercell: ase.Atoms = None,
        eps: float = 1e-12,
        ) -> np.ndarray:
    """Return ``-d/dR`` features for normalized local composition."""
    if supercell is None:
        supercell = geom
    n_atoms = len(geom)
    distance_matrix = distances.get_distance_matrix(geom, supercell)
    geo_composition = np.array(geom.get_atomic_numbers())
    sup_composition = np.array(supercell.get_atomic_numbers())
    sup_positions = supercell.get_positions()
    neighbor_numbers = sorted(set(sup_composition.tolist()))

    features = np.zeros((n_atoms, 3, len(composition_interactions)))
    for idx, interaction in enumerate(composition_interactions):
        center_z, target_z = _numbers_from_interaction(interaction)
        r_min = r_min_map[interaction]
        r_max = r_max_map[interaction]
        power = power_map[interaction]

        pair_data = {}
        rho_by_z = {}
        rho_total = np.zeros(n_atoms)
        for z in neighbor_numbers:
            result = _pair_density_by_center(
                distance_matrix, geo_composition, sup_composition,
                center_z, z, r_min, r_max, power, n_atoms)
            if result[0] is None:
                continue
            i_where, j_where, rho = result
            pair_data[z] = (i_where, j_where)
            rho_by_z[z] = rho
            rho_total += rho
        if not pair_data:
            continue
        rho_target = rho_by_z.get(target_z, np.zeros(n_atoms))
        for z, (i_where, j_where) in pair_data.items():
            valid = rho_total[i_where] > eps
            if not np.any(valid):
                continue
            i_valid = i_where[valid]
            j_valid = j_where[valid]
            rij = distance_matrix[i_valid, j_valid]
            drij_dR = distances.compute_direction_cosines(
                sup_positions, distance_matrix, i_valid, j_valid, n_atoms)
            numerator = ((1.0 if z == target_z else 0.0) *
                         rho_total[i_valid] - rho_target[i_valid])
            quotient_derivative = numerator / (rho_total[i_valid] ** 2)
            weights = quotient_derivative * power * rij ** (-power - 1.0)
            features[:, :, idx] += np.sum(drij_dR * weights[None, None, :],
                                          axis=-1)
    return features
