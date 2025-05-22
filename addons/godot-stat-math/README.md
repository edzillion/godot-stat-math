# Godot Stat Math

![Godot 4.x](https://img.shields.io/badge/Godot-4.x-blue?logo=godot-engine)

**Godot Stat Math** is a Godot 4 addon providing common statistical functions for game developers, exposed via the global `StatMath` autoload singleton. It is designed for practical, game-oriented use—if you need scientific-grade accuracy, consider a dedicated scientific library.

> **Work in Progress:**  
> This addon is under active development. Comments, suggestions, and contributions are very welcome! Please open an issue or pull request if you have feedback or ideas.

## Features

- Random variate generation for common distributions (Bernoulli, Binomial, Poisson, Normal, Exponential, etc.)
- CDF, PMF, and PPF functions for many distributions
- Special functions: error function, gamma, beta, and more
- All functions and constants are accessible via the `StatMath` singleton

## Example Usage

```gdscript
# Generate a random number from a normal distribution
var x: float = StatMath.Distributions.randf_normal(0.0, 1.0)

# Compute the CDF of the normal distribution
var p: float = StatMath.CdfFunctions.normal_cdf(x, 0.0, 1.0)

# Binomial coefficient
var k_val: float = StatMath.HelperFunctions.binomial_coefficient(10, 3)

# Error function
var erf_val: float = StatMath.ErrorFunctions.error_function(1.0)
```

## API Reference (Selected)

All modules are accessed as `StatMath.ModuleName.function_name(...)`.  
See the source for full documentation and comments.

### Distributions

- `randi_bernoulli(p: float) -> int`  
  Returns 1 with probability `p`, 0 otherwise.

- `randi_binomial(p: float, n: int) -> int`  
  Number of successes in `n` Bernoulli trials.

- `randf_normal(mu: float = 0.0, sigma: float = 1.0) -> float`  
  Random float from a normal (Gaussian) distribution.

### CDF Functions

- `normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float`  
  Cumulative probability for the normal distribution.

- `binomial_cdf(k: int, n: int, p: float) -> float`  
  Probability of ≤k successes in n binomial trials.

### Helper Functions

- `binomial_coefficient(n: int, r: int) -> float`  
  Number of ways to choose `r` from `n`.

- `gamma_function(z: float) -> float`  
  Gamma function Γ(z).

### Error Functions

- `error_function(x: float) -> float`  
  Computes erf(x).

- `error_function_inverse(y: float) -> float`  
  Inverse error function.

### Sampling (via StatMath.SamplingGen)

- `generate_samples_1d(ndraws: int, method: SamplingMethod, seed: int = -1) -> Array[float]`  
  Generates `ndraws` 1D samples (Array of floats) using the specified method. 
  If `seed` is not -1, a local RNG seeded with this value is used.

- `generate_samples_2d(ndraws: int, method: SamplingMethod, seed: int = -1) -> Array[Vector2]`  
  Generates `ndraws` 2D samples (Array of Vector2) using the specified method. 
  If `seed` is not -1, a local RNG seeded with this value is used.

## Reproducible Results (Seeding the RNG)

`Godot Stat Math` provides a robust system for controlling the random number generation (RNG) to ensure reproducible results, which is essential for debugging, testing, and consistent behavior in game mechanics.

There are two main ways to control seeding:

1.  **Global Project Seed (`monte_godot_seed`):**
    *   On startup, `StatMath` looks for a project setting named `monte_godot_seed`.
    *   If this integer setting exists in your `project.godot` file, `StatMath` will use its value to seed its global RNG.
    *   Example `project.godot` entry:
        ```gdscript
        [application]
        config/name="My Game"
        # ... other settings ...
        monte_godot_seed=12345
        ```
    *   If the setting is not found, or is not an integer, `StatMath` will initialize its RNG with a default seed (0, which typically means Godot's RNG will pick a time-based random seed). A message will be printed to the console indicating the seed used.
    *   This method is convenient for setting a consistent seed across your entire project for all runs.

2.  **Runtime Seeding (`StatMath.set_seed()`):**
    *   You can change the seed of the global `StatMath` RNG at any point during runtime by calling:
        ```gdscript
        StatMath.set_seed(new_seed_value)
        ```
    *   This will re-initialize the global RNG with `new_seed_value`. All subsequent calls to `StatMath` functions that use random numbers (without an explicit per-call seed) will be based on this new seed.
    *   This is useful for specific scenarios where you want to ensure a particular sequence of random events is reproducible from a certain point in your game logic.

3.  **Per-Call Seeding (for `SamplingGen.generate_samples_1d()` and `SamplingGen.generate_samples_2d()`):**
    *   The `StatMath.SamplingGen.generate_samples_1d()` and `StatMath.SamplingGen.generate_samples_2d()` functions accept an optional `seed` parameter (defaulting to -1).
    *   When a `seed` other than -1 is provided to these functions, it creates a *local* `RandomNumberGenerator` instance, seeded with the given value. This local RNG is used only for that specific call.
    *   This ensures that the output of that particular sampling operation is deterministic based on the provided seed, without affecting the global `StatMath` RNG state.
    *   If `seed = -1` (the default) is used, the functions will use the global `StatMath` RNG (controlled by `monte_godot_seed` or `StatMath.set_seed()`).

**How it Works for Determinism:**

By controlling the seed, you control the sequence of pseudo-random numbers generated. If you start with the same seed, and perform the exact same sequence of operations that consume random numbers, you will always get the same results. This is invaluable for:

*   **Debugging:** If a bug appears due to a specific random outcome, you can reproduce it by using the same seed.
*   **Testing:** Ensures tests that rely on random data behave consistently.
*   **Gameplay:** Can be used to create "daily challenges" with the same layout/events for all players, or to allow players to share seeds for specific game setups.

## Known Limitations / TODOs

- **Placeholder Functions:**
  - `StatMath.HelperFunctions.incomplete_beta(x, a, b)` is currently a placeholder and not implemented. It always returns `NAN` and should not be used for any calculations requiring accuracy.
  - `StatMath.HelperFunctions.lower_incomplete_gamma_regularized(a, z)` is also a placeholder and not fully verified. It may return unreliable or placeholder values.
- **General Reliability:**
  - This project is a work in progress. Some results, especially those relying on the above functions, may be unreliable or incorrect. Do not use this addon for critical or scientific/statistical applications requiring high accuracy at this time.

## Documentation

All functions are well-commented in the source code.  
For full details, see the scripts in `addons/godot-stat-math/core/`.

## License

Unlicense (public domain, see LICENSE)