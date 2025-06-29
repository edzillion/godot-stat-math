# Godot Stat Math

[![Release](https://img.shields.io/github/v/release/edzillion/godot-stat-math?sort=semver)](https://github.com/edzillion/godot-stat-math/releases)
[![Build Dev](https://github.com/edzillion/godot-stat-math/actions/workflows/build-develop.yaml/badge.svg)](https://github.com/edzillion/godot-stat-math/actions/workflows/build-develop.yaml)
[![Unit Tests](https://img.shields.io/endpoint?url=https://edzillion.github.io/godot-stat-math/_static/badges/unit_tests.json)](https://edzillion.github.io/godot-stat-math/unit_testing.html)
[![Performance Tests](https://img.shields.io/endpoint?url=https://edzillion.github.io/godot-stat-math/_static/badges/performance_tests.json)](https://edzillion.github.io/godot-stat-math/performance.html)
![Godot 4.x](https://img.shields.io/badge/Godot-4.x-blue?logo=godot-engine)

**Godot Stat Math** is a Godot 4 addon providing common statistical functions for game developers, exposed via the global `StatMath` autoload singleton. It is designed for practical, game-oriented use—if you need scientific-grade accuracy, consider a dedicated scientific library.

> **⚠️ Work in Progress:**  
> This addon is under active development. Comments, suggestions, and contributions are very welcome! Please open an issue or pull request if you have feedback or ideas.

## Features

- **Random variate generation** for 17+ distributions:
  - Discrete: Bernoulli, Binomial, Poisson, Geometric, Negative Binomial
  - Continuous: Normal, Log-normal, Exponential, Gamma, Beta, Weibull, Pareto, Cauchy, Triangular, Chi-squared, Student's t, F-distribution
  - Game-specific: Pseudo (capture mechanics), Siege (warfare mechanics), Erlang
- **Distribution functions**: CDF, PMF, PDF, and PPF (quantile) functions for statistical analysis
- **Special mathematical functions**: Error functions, Gamma, Beta, incomplete variants, combinatorial functions
- **Comprehensive statistical analysis**: Mean, median, variance, standard deviation, percentiles, quantiles, median absolute deviation, sample vs population statistics, and summary statistics
- **Advanced sampling methods**: Sobol sequences, Halton sequences, Latin Hypercube sampling, reservoir sampling, Fisher-Yates shuffling
- **Mathematical utilities**: Array preprocessing, combinatorial calculations, logarithmic functions for numerical stability
- All functions and constants are accessible via the `StatMath` singleton

## Quick Start

```gdscript
# Generate random numbers from distributions
var normal_val: float = StatMath.Distributions.randf_normal(0.0, 1.0)
var mean_val: float = StatMath.BasicStats.mean([1.0, 2.0, 3.0, 4.0, 5.0])
var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)

# Compute CDFs and quantiles
var cdf_value: float = StatMath.CdfFunctions.normal_cdf(1.96, 0.0, 1.0)
var quantile: float = StatMath.PpfFunctions.normal_ppf(0.975, 0.0, 1.0)

# Statistical analysis
var data = [95.5, 87.2, 92.1, 88.8, 90.0]
var summary: Dictionary = StatMath.BasicStats.summary_statistics(data)
```

## Documentation

📚 **Complete API Documentation**: [https://edzillion.github.io/godot-stat-math/](https://edzillion.github.io/godot-stat-math/)

The full documentation includes:
- **Installation Guide** - Step-by-step setup instructions
- **API Reference** - Complete function documentation for all 8 modules
- **Usage Examples** - Practical examples for game development
- **Testing Standards** - Information about our comprehensive test suite
- **Performance Benchmarks** - Performance analysis and optimization details

*Documentation is built with Sphinx from the `/docs` directory and automatically updated with each release.*

## Reproducible Results (Seeding the RNG)

`Godot Stat Math` provides a robust system for controlling the random number generation (RNG) to ensure reproducible results, which is essential for debugging, testing, and consistent behavior in game mechanics.

There are two main ways to control seeding:

1.  **Global Project Seed (`godot_stat_math_seed`):**
    *   On startup, `StatMath` looks for a project setting named `godot_stat_math_seed`.
    *   If this integer setting exists in your `project.godot` file, `StatMath` will use its value to seed its global RNG.
    *   Example `project.godot` entry:
        ```ini
        [application]
        config/name="My Game"
        # ... other settings ...
        godot_stat_math_seed=12345
        ```
    *   If the setting is not found, or is not an integer, `StatMath` will initialize its RNG with a default seed (0, which typically means Godot's RNG will pick a time-based random seed). A message will be printed to the console indicating the seed used.
    *   This method is convenient for setting a consistent seed across your entire project for all runs.

2.  **Runtime Seeding (`StatMath.set_global_seed()`):**
    *   You can change the seed of the global `StatMath` RNG at any point during runtime by calling:
        ```gdscript
        StatMath.set_global_seed(new_seed_value)
        ```
    *   This will re-initialize the global RNG with `new_seed_value`. All subsequent calls to `StatMath` functions that use random numbers (without an explicit per-call seed) will be based on this new seed.
    *   This is useful for specific scenarios where you want to ensure a particular sequence of random events is reproducible from a certain point in your game logic.

3.  **Per-Call Seeding (for `SamplingGen.generate_samples()`):**
    *   The `StatMath.SamplingGen.generate_samples()` function accepts an optional `sample_seed` parameter (defaulting to -1).
    *   When a `sample_seed` other than -1 is provided, it creates a *local* `RandomNumberGenerator` instance, seeded with the given value. This local RNG is used only for that specific call.
    *   This ensures that the output of that particular sampling operation is deterministic based on the provided seed, without affecting the global `StatMath` RNG state.
    *   If `sample_seed = -1` (the default) is used, the function will use the global `StatMath` RNG (controlled by `godot_stat_math_seed` or `StatMath.set_global_seed()`).

**How it Works for Determinism:**

By controlling the seed, you control the sequence of pseudo-random numbers generated. If you start with the same seed, and perform the exact same sequence of operations that consume random numbers, you will always get the same results. This is invaluable for:

*   **Debugging:** If a bug appears due to a specific random outcome, you can reproduce it by using the same seed.
*   **Testing:** Ensures tests that rely on random data behave consistently.
*   **Gameplay:** Can be used to create "daily challenges" with the same layout/events for all players, or to allow players to share seeds for specific game setups.

## Installation

1. Download the latest release from the [Releases page](https://github.com/edzillion/godot-stat-math/releases)
2. Extract the ZIP file to your project's `addons/` folder
3. Enable "Godot Stat Math" in Project Settings → Plugins
4. The `StatMath` singleton will be automatically available in your project

For detailed installation instructions, see the [Installation Guide](https://edzillion.github.io/godot-stat-math/installation.html).

## License

Unlicense (public domain, see LICENSE)