
Template Alignment Benchmark
============================
|Python 3.11+|

This repositery contains the companion code for the article *"Functional Templates in fMRI: Building Accurate and Interpretable Group-Level Decoders" (in submission)*.

Install
-------

This benchmark relies on `benchopt`_ and `fmralign`_. It can be installed using the following commands:

.. code-block::

   $ pip install -U benchopt git+https://github.com/fmralign/fmralign.git

Visualization is handled separetly using the scripts in the ``plots/`` directory. By default, benchopt will try to generate a figure. To avoid any error, make sure to pass the ``--no-plot`` option to ``benchopt run``.
Options can be passed to ``benchopt run --no-plot``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

   $ benchopt run --no-plot -d simulated -s procrustes

Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.


Configuration
-------------

Edit the ``benchmark_utils/conf.py`` file to suit your own config. We recommend running the complete benchmark on a Slurm cluster.

.. |Python 3.11+| image:: https://img.shields.io/badge/python-3.11%2B-blue
   :target: https://www.python.org/downloads/release/python-31115/

.. _benchopt: https://benchopt.github.io/stable/

.. _fmralign: https://fmralign.github.io/fmralign/
