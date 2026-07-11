UF3 Density Development Fork
============================

|Tests|


\Xie, S.R., Rupp, M. & Hennig, R.G., "Ultra-fast interpretable machine-learning potentials", npj Comput Mater 9, 162 (2023). https://doi.org/10.1038/s41524-023-01092-7

.. _https://doi.org/10.1038/s41524-023-01092-7: https://doi.org/10.1038/s41524-023-01092-7

All-atom dynamics simulations have become an indispensable quantitative
tool in physics, chemistry, and materials science, but large systems and
long simulation times remain challenging due to the trade-off between
computational efficiency and predictive accuracy. The UF3 framework is
built to address this challenge by combining effective two- and
three-body potentials in a cubic B-spline basis with regularized linear
regression to obtain machine-learning potentials that are physically
interpretable, sufficiently accurate for applications, and as fast as
the fastest traditional empirical potentials.

This repository is a development fork of UF3 for testing density-aware
extensions to the original two-body and three-body B-spline force-field
form. It keeps the original UF3 package structure while adding
environment-dependent descriptors and calculators used in our current
surface-energy studies.

For the upstream UF3 project and documentation, see:

- Documentation: https://uf3.readthedocs.io/
- Upstream repository: https://github.com/uf3/uf3

Development Scope
-----------------

The main additions in this fork are density-modulated interaction terms.
The current practical focus is the ``2rho`` term, which modulates the
two-body B-spline contribution by local atomic density channels. The
repository also contains experimental support for density-modulated
three-body terms used for ablation tests.

The most relevant files are:

- ``uf3/representation/density.py``: local density evaluation, density
  channels, and smooth density cutoff functions.
- ``uf3/representation/bspline.py``: two-body, three-body, ``2rho``,
  and experimental ``3rho`` B-spline feature construction.
- ``uf3/representation/process.py``: featurization pipeline integration.
- ``uf3/regression/least_squares.py``: fitting, feature scaling, and
  regularization support for the extended feature sets.
- ``uf3/forcefield/calculator.py``: ASE-compatible calculator support
  for evaluating trained extended UF3 models.

At present, the extended density terms are intended to be used through
the Python/ASE calculator path. The LAMMPS plugin included here is kept
for compatibility with the original UF3 workflow and should not be
assumed to support the new density-modulated terms.

Setup
-----

.. Recommended: Install UF3 in a new conda environment:

.. .. code:: bash

..    conda create -n uf3_env python=3.8
..    conda activate uf3_env

UF3 can be obtained by cloning the repository and installing it:

..
   1. Download and install automatically from PyPI (recommended):
   pip install uf3
   Download and install manually from GitHub:

.. code:: bash

   git clone git@github.com:Kai-Liu-MSE/uf3_develop.git
   cd uf3_develop
   pip install .

Getting Started
---------------

For original UF3 usage patterns, please refer to the upstream
documentation and tests. For density-modulated terms, the development
notes in this repository summarize the design choices and current test
protocols:

- ``uf3_density_modulated_terms_notes.md``
- ``uf3_2rho_term_formula.md``
- ``uf3_2rho_as_true_mlip_term.md``
- ``uf3_new_term_ablation_protocol.md``

This fork intentionally does not include local training datasets,
large feature files, model checkpoints, or surface-energy calculation
outputs.

Optional Dependencies
---------------------

Elastic constants:

::

   pip install setuptools_scm
   pip install uf3[elastic_constants]

Phonon spectra:

::

   pip install uf3[phonon_spectra]

LAMMPS interface:

::

   conda install numpy==1.20.3 --force-reinstall
   conda install -c conda-forge lammps --no-update-deps

The environment variable ``$ASE_LAMMPSRUN_COMMAND`` must also be set to use the LAMMPS interface within python. See the `ASE documentation <https://wiki.fysik.dtu.dk/ase/ase/calculators/lammpsrun.html>`_ for details.

Dependencies
------------

-  We rely on ase to handle parsing outputs from atomistic codes like
   LAMMPS, VASP, and CP2K.
-  We use Pandas to keep track of atomic configurations and their
   energies/forces as well as organizing data for featurization and
   training.
-  B-spline evaluations use scipy, numba, and ndsplines.
-  PyTables is used for reading/writing HDF5 files.
-  Matplotlib is used for plotting.
-  We use sklearn for regression.
-  We use tqdm for progress bars.
-  We use plotly for interactive plots.
-  We use PyYaml for configuration files.
-  We use numpy for array operations.


.. |Tests| image:: https://github.com/Kai-Liu-MSE/uf3_develop/workflows/Tests/badge.svg
   :target: https://github.com/Kai-Liu-MSE/uf3_develop/actions
