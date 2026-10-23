# Local ML-UF3 LAMMPS source patch

`lammps_abfdbecc_ml_uf3_extensions.patch` captures the tested local
`pair_uf3.cpp` and `pair_uf3.h` changes against LAMMPS commit
`abfdbecc4956dd7017d9458b11001cb14e1feb20` (29 Aug 2024 update 4).
It includes the existing local ML-UF3 extensions as well as the weighted
`2rho` implementation. It is not an upstream LAMMPS patch or a separately
released LAMMPS fork.

Apply it only to a clean checkout at that exact base commit:

```bash
git checkout abfdbecc4956dd7017d9458b11001cb14e1feb20
git apply --check path/to/lammps_abfdbecc_ml_uf3_extensions.patch
git apply path/to/lammps_abfdbecc_ml_uf3_extensions.patch
```

Configure an out-of-source CMake build with `PKG_ML-UF3=ON` and the target
machine's MPI toolchain, then run the Python/LAMMPS parity tests in
`tests/test_weighted_2rho.py`. This patch is a reproducible source snapshot,
not evidence of AION build or production validation.
