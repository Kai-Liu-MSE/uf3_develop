#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train UF2,3 potential for Al–Mg–Cu using 90/10 split from an extended XYZ.

Outputs:
- df_features_AlMgCu.h5
- model_uf23_AlMgCu.json
- train_idx.txt, test_idx.txt
"""

import os
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from uf3.data import io
from uf3.data import composition
from uf3.representation import bspline, process
from uf3.regression import least_squares
from uf3.forcefield import calculator


def parse_args():
    p = argparse.ArgumentParser(description="Train UF2,3 potential for Co")
    p.add_argument("--xyz", type=str, default="output_all.xyz",
                   help="Input extended XYZ (must contain energy=... and Properties...:force:R:3)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for 90/10 split")
    p.add_argument("--train-frac", type=float, default=0.9,
                   help="Training fraction (default 0.9 for 90/10 split)")
    p.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 1)),
                   help="Worker processes for featurization")
    p.add_argument("--features", type=str, default="df_features_Co.h5",
                   help="Output HDF5 for features")
    p.add_argument("--model", type=str, default="model_uf23_Co.json",
                   help="Output JSON model")
    p.add_argument("--energy-key", type=str, default="energy")
    p.add_argument("--force-key",  type=str, default="force") 
    return p.parse_args()



def main():
    args = parse_args()

    # ---------------- 1) System & degree ----------------
    element_list = ["Co"]
    degree = 3  # UF2,3
    chemical_system = composition.ChemicalSystem(element_list=element_list, degree=degree)

    # ---------------- 2) Hyperparameters (Co; inner cutoff ~1.5 Å) ----------------
    r_min_map = {}
    r_max_map = {}
    resolution_map = {}

    pairs = [
        ("Co", "Co"),
    ]

    pair_rmin = 1.5
    pair_rmax = {
        ("Co", "Co"): 5.5,
    }
    pair_res = 8

    for a, b in pairs:
        r_min_map[(a, b)] = pair_rmin
        r_max_map[(a, b)] = pair_rmax[(a, b)]
        resolution_map[(a, b)] = pair_res

    leading_trim  = {2: 0, 3: 0}
    trailing_trim = {2: 3, 3: 3}

    # 3-body short cutoffs (tighter than 2-body to control feature size)
    pair_rmax_3b_short = {
        ("Co", "Co"): 3.5,
    }

    def _pair_key(x, y):
        return (x, y) if (x, y) in pair_rmax_3b_short else (y, x)

    elems = element_list
    for i in elems:
        for j in elems:
            for k in elems:
                key3 = (i, j, k)
                s1 = pair_rmax_3b_short[_pair_key(i, j)]
                s2 = pair_rmax_3b_short[_pair_key(j, k)]
                s = min(s1, s2)  # enforce equal short edges for stability
                r_min_map[key3] = [pair_rmin, pair_rmin, pair_rmin]
                r_max_map[key3] = [s, s, 2.0 * s]
                resolution_map[key3] = [4, 4, 8]

    # ---------------- 3) Build B-spline basis ----------------
    bspline_config = bspline.BSplineBasis(
        chemical_system,
        r_min_map=r_min_map,
        r_max_map=r_max_map,
        resolution_map=resolution_map,
        leading_trim=leading_trim,
        trailing_trim=trailing_trim
    )

    # ---------------- 4) Load data ----------------
    data_coordinator = io.DataCoordinator(energy_key=args.energy_key,
                                          force_key=args.force_key)
    data_coordinator.dataframe_from_trajectory(args.xyz, prefix="dft",
                                               energy_key=args.energy_key,
                                               force_key=args.force_key)
    df_data = data_coordinator.consolidate()



    # Drop configs with <3 atoms (cannot support 3-body)
    mask = df_data["geometry"].apply(lambda geom: len(geom) >= 3)
    n_drop = (~mask).sum()
    if n_drop > 0:
        print(f"[INFO] drop {n_drop} configs with < 3 atoms")
        np.savetxt(
            "dropped_lt3atoms.txt",
            df_data.index.to_numpy()[~mask].astype(str),
            fmt="%s"
        )
    df_data = df_data[mask].copy()


    # ---------------- 5) 90/10 split ----------------
    rng = np.random.default_rng(args.seed)
    all_keys = df_data.index.to_numpy()
    perm = rng.permutation(all_keys)

    n_train = int(args.train_frac * len(perm))
    train_keys = perm[:n_train]
    test_keys  = perm[n_train:]

    np.savetxt("train_idx.txt", train_keys, fmt="%s")
    np.savetxt("test_idx.txt",  test_keys,  fmt="%s")
    print(f"[INFO] Split: train {len(train_keys)}, test {len(test_keys)}")

    # ---------------- 6) Featurize to HDF5 ----------------
    representation = process.BasisFeaturizer(bspline_config)
    print(f"[INFO] Featurizing with {args.cores} workers → {args.features}")
    with ProcessPoolExecutor(max_workers=args.cores) as client:
        representation.batched_to_hdf(
            args.features,
            df_data,
            client,
            n_jobs=args.cores,
            batch_size=50,
            progress="bar",
            table_template="features_{}"
        )

    # ---------------- 7) Regularization & model ----------------
    regularizer = bspline_config.get_regularization_matrix(
        ridge_1b=0.0,
        ridge_2b=0.0,
        ridge_3b=1e-8,
        curvature_2b=1e-8,
        curvature_3b=0.0
    )
    model = least_squares.WeightedLinearModel(bspline_config, regularizer=regularizer)

    # ---------------- 8) Fit (energies + force) on train ----------------
    print("[INFO] Training model (UF2,3) on training split ...")
    model.fit_from_file(
        args.features,
        train_keys,
        weight=0.8,
        batch_size=500,
        energy_key="energy",
        progress="bar"
    )

    # ---------------- 9) Evaluate on test split ----------------
    print("[INFO] Evaluating on test split ...")
    y_e, p_e, y_f, p_f, rmse_e, rmse_f = model.batched_predict(args.features, keys=test_keys)
    print(f"[RESULT] Test RMSE (energy): {rmse_e:.6e}")
    print(f"[RESULT] Test RMSE (force): {rmse_f:.6e}")

    # ---------------- 10) Export model ----------------
    model.to_json(args.model)
    print(f"[INFO] Saved UF2,3 model → {args.model}")

    # ---------------- 11) Sanity check on a single geometry ----------------
    calc = calculator.UFCalculator(model)
    geom = df_data.iloc[0]["geometry"].copy()
    geom.set_calculator(calc)
    e = geom.get_potential_energy()
    fmax = np.max(np.abs(geom.get_forces()))
    print(f"[CHECK] Energy of first geom: {e:.6f} ; max |F|: {fmax:.6e}")


if __name__ == "__main__":
    main()
