This document describes how to generate UF3 lammps potential from an UF3 model file(JSON).

Usuage-

.. code:: bash

   python generate_uf3_lammps_pots.py [-h] -a AUTHOR -u UNITS -m MODEL [-d DIRECTORY] [-k KNOTS_SPACING_TYPE]

   usage: generate_uf3_lammps_pots.py [-h] -a AUTHOR -u UNITS -m MODEL [-d DIRECTORY] [-k KNOTS_SPACING_TYPE]
   
   Generate UF3 LAMMPS potential file
   
   optional arguments:
        -h, --help            show this help message and exit
        -a AUTHOR, --author AUTHOR
                              Author Name Seperated by '_'
        -u UNITS, --units UNITS
                              LAMMPS Units
        -m MODEL, --model MODEL
                              UF3 Model JSON file
        -d DIRECTORY, --directory DIRECTORY
                              Directory path (default: current directory)
        -k KNOTS_SPACING_TYPE, --knots_spacing_type KNOTS_SPACING_TYPE
                              Knot spacing type, uk (uniform spacing) or nk (non-uniform spacing) (default: nk (non-uniform))

Will create :code:`directory` if it does not exists and will write UF3 lammps potential files in it. 


It will also additionally print lines that should be added to the lammps input file for using UF3 lammps potential files.
Citation meta-data will be left blank. Please enter appropriate citation for the generated UF3 LAMMPS potential manually.

Local weighted 2rho extension
=============================

The local UF3 exporter writes one :code:`2RHO` block per center-neighbor
orientation. A mixed pair therefore has two blocks with the same spline
coefficients but potentially different density metadata. The density line is

.. code:: text

   r_min r_on r_cut power rho_bulk weight quintic conservative

For center atom :code:`i`, the density sums :code:`weight(center, neighbor) *
r**(-power) * cutoff(r)` over its neighbors. :code:`rho_bulk` is supplied
separately for each center-neighbor orientation; it is not automatically
rescaled by the weight. An older seven-field density line without :code:`weight`
remains readable and uses a weight of 1. This extension requires the matching
local ML-UF3 LAMMPS source; it is not part of the upstream pair-style format.
