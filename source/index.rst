.. Godot Stat Math documentation master file, created by
   sphinx-quickstart on Mon Jun 23 00:36:19 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Godot Stat Math Documentation
=============================

**Godot Stat Math** is a comprehensive Godot 4 addon providing statistical functions for game developers, exposed via the global ``StatMath`` autoload singleton. It offers distributions, statistical analysis, special functions, and advanced sampling methods designed for practical game development use.

🚀 **Quick Start**
------------------

The StatMath singleton provides access to all functionality through organized modules:

.. code-block:: gdscript

   # Generate random numbers from distributions
   var normal_val = StatMath.Distributions.randf_normal(0.0, 1.0)
   var poisson_val = StatMath.Distributions.randi_poisson(3.5)
   
   # Calculate statistics
   var data = [95.5, 87.2, 92.1, 88.8, 90.0]
   var avg = StatMath.BasicStats.mean(data)
   var std_dev = StatMath.BasicStats.standard_deviation(data)
   
   # Use distribution functions
   var cdf_value = StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)
   var quantile = StatMath.PpfFunctions.normal_ppf(0.975, 0.0, 1.0)

📚 **Table of Contents**
------------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start
   seeding_rng

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   modules/basic_stats
   modules/distributions
   modules/cdf_functions
   modules/ppf_functions
   modules/pmf_pdf_functions
   modules/error_functions
   modules/helper_functions
   modules/sampling_gen

.. toctree::
   :maxdepth: 2
   :caption: Advanced Usage

   examples/game_analytics
   examples/procedural_generation
   examples/testing_randomness
   performance/optimization_guide
   performance/benchmarks

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
   mathematical_background
   testing_standards
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   architecture
   license

**Indices and Tables**
----------------------

* :ref:`genindex`
* :ref:`search`

**Key Features**
----------------

* 🎲 **Random Number Generation**: 20+ probability distributions (Normal, Poisson, Gamma, Beta, Weibull, etc.)
* 📊 **Statistical Analysis**: Mean, variance, quantiles, and robust statistics
* 🧮 **Distribution Functions**: CDF, PMF/PDF, and PPF (quantile) functions
* 🔬 **Special Functions**: Error functions, Gamma, Beta, incomplete functions
* 🎯 **Advanced Sampling**: Sobol, Halton, Latin Hypercube sampling
* 🎮 **Game-Focused**: Optimized for game development workflows
* 🔄 **Reproducible**: Comprehensive seeding system for deterministic results

**Compatibility**: Godot 4.0+

