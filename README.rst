
Template Alignment Benchmark
============================
|Python 3.11+|

This repositery contains the companion code for the article *"Functional Templates in fMRI: Building Accurate and Interpretable Group-Level Decoders"*. 

Install
-------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt git+https://github.com/fmralign/fmralign.git

Apart from the problem, options can be passed to ``benchopt run --no-plot``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

   $ benchopt run --no-plot -d simulated -s procrustes


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Python 3.11+| image:: https://img.shields.io/badge/python-3.11%2B-blue
   :target: https://www.python.org/downloads/release/python-31115/
