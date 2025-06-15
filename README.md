# Godot Stat Math

![Godot 4.x](https://img.shields.io/badge/Godot-4.x-blue?logo=godot-engine)

**Godot Stat Math** is a Godot 4 addon providing common statistical functions for game developers, exposed via the global `StatMath` autoload singleton. It is designed for practical, game-oriented use—if you need scientific-grade accuracy, consider a dedicated scientific library.

> **Work in Progress:**  
> This addon is under active development. Comments, suggestions, and contributions are very welcome! Please open an issue or pull request if you have feedback or ideas.

## Features

- Random variate generation for common distributions (Bernoulli, Binomial, Poisson, Normal, Exponential, Gamma, Beta, Weibull, Pareto, Cauchy, Triangular, etc.)
- CDF, PMF, and PPF functions for many distributions
- Special functions: error function, gamma, beta, incomplete beta, incomplete gamma, and more
- Basic statistical analysis functions (mean, median, variance, standard deviation, etc.)
- Advanced sampling methods (Sobol, Halton, Latin Hypercube)
- All functions and constants are accessible via the `StatMath` singleton

## Example Usage

```python
# Generate random numbers from various distributions
var normal_val: float = StatMath.Distributions.randf_normal(0.0, 1.0)
var weibull_val: float = StatMath.Distributions.randf_weibull(2.0, 1.5)
var binomial_val: int = StatMath.Distributions.randi_binomial(0.3, 10)

# Compute CDFs (cumulative distribution functions)
var normal_cdf: float = StatMath.CdfFunctions.normal_cdf(1.0, 0.0, 1.0)
var weibull_cdf: float = StatMath.CdfFunctions.weibull_cdf(2.0, 2.0, 1.5)

# Compute PPFs (percent point functions / quantiles)
var normal_quantile: float = StatMath.PpfFunctions.normal_ppf(0.95, 0.0, 1.0)
var weibull_quantile: float = StatMath.PpfFunctions.weibull_ppf(0.95, 2.0, 1.5)

# Mathematical helper functions
var binom_coeff: float = StatMath.HelperFunctions.binomial_coefficient(10, 3)
var gamma_val: float = StatMath.HelperFunctions.gamma_function(2.5)
var erf_val: float = StatMath.ErrorFunctions.error_function(1.0)

# Basic statistics - analyze player scores
var raw_scores = [95.5, "invalid", 87.2, null, 92.1, 88.8, 90.0]
var clean_scores: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(raw_scores)
var avg_score: float = StatMath.BasicStats.mean(clean_scores)
var score_std_dev: float = StatMath.BasicStats.standard_deviation(clean_scores)
var summary: Dictionary = StatMath.BasicStats.summary_statistics(clean_scores)

# Advanced sampling for procedural generation
var sobol_samples_1d: Array[float] = StatMath.SamplingGen.generate_samples(100, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
var sobol_samples_2d: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
var sobol_samples_3d: Array = StatMath.SamplingGen.generate_samples(100, 3, StatMath.SamplingGen.SamplingMethod.SOBOL)
```

## API Reference (Selected)

All modules are accessed as `StatMath.ModuleName.function_name(...)`.  
See the source for full documentation and comments.

### Distributions

**Integer Distributions:**
- `randi_bernoulli(p: float) -> int` - Returns 1 with probability `p`, 0 otherwise
- `randi_binomial(p: float, n: int) -> int` - Number of successes in `n` Bernoulli trials
- `randi_geometric(p: float) -> int` - Number of trials until first success
- `randi_poisson(lambda_param: float) -> int` - Number of events in fixed interval

**Continuous Distributions:**
- `randf_uniform(a: float, b: float) -> float` - Uniform distribution on [a, b]
- `randf_normal(mu: float = 0.0, sigma: float = 1.0) -> float` - Normal (Gaussian) distribution
- `randf_exponential(lambda_param: float) -> float` - Exponential distribution
- `randf_gamma(shape: float, scale: float = 1.0) -> float` - Gamma distribution
- `randf_beta(alpha: float, beta_param: float) -> float` - Beta distribution
- `randf_weibull(scale_param: float, shape_param: float) -> float` - Weibull distribution
- `randf_pareto(scale_param: float, shape_param: float) -> float` - Pareto distribution
- `randf_cauchy(location: float = 0.0, scale: float = 1.0) -> float` - Cauchy distribution
- `randf_triangular(min_value: float, max_value: float, mode_value: float) -> float` - Triangular distribution

### CDF Functions

- `uniform_cdf(x: float, a: float, b: float) -> float` - Uniform CDF
- `normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float` - Normal CDF
- `exponential_cdf(x: float, lambda_param: float) -> float` - Exponential CDF
- `gamma_cdf(x: float, k_shape: float, theta_scale: float) -> float` - Gamma CDF
- `beta_cdf(x: float, alpha: float, beta_param: float) -> float` - Beta CDF
- `weibull_cdf(x: float, scale_param: float, shape_param: float) -> float` - Weibull CDF
- `pareto_cdf(x: float, scale_param: float, shape_param: float) -> float` - Pareto CDF
- `binomial_cdf(k: int, n: int, p: float) -> float` - Binomial CDF
- `poisson_cdf(k: int, lambda_param: float) -> float` - Poisson CDF

### PPF Functions (Quantiles/Inverse CDFs)

- `uniform_ppf(p: float, a: float, b: float) -> float` - Uniform quantile function
- `normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float` - Normal quantile function
- `exponential_ppf(p: float, lambda_param: float) -> float` - Exponential quantile function
- `gamma_ppf(p: float, k_shape: float, theta_scale: float) -> float` - Gamma quantile function
- `beta_ppf(p: float, alpha_shape: float, beta_shape: float) -> float` - Beta quantile function
- `weibull_ppf(p: float, scale_param: float, shape_param: float) -> float` - Weibull quantile function
- `pareto_ppf(p: float, scale_param: float, shape_param: float) -> float` - Pareto quantile function
- `binomial_ppf(p: float, n: int, prob_success: float) -> int` - Binomial quantile function
- `poisson_ppf(p: float, lambda_param: float) -> int` - Poisson quantile function

### Helper Functions

- `binomial_coefficient(n: int, r: int) -> float` - Number of ways to choose `r` from `n`
- `gamma_function(z: float) -> float` - Gamma function Γ(z)
- `beta_function(a: float, b: float) -> float` - Beta function B(a,b)
- `incomplete_beta(x_val: float, a: float, b: float) -> float` - Regularized incomplete beta function
- `lower_incomplete_gamma_regularized(a: float, z: float) -> float` - Regularized lower incomplete gamma function
- `sanitize_numeric_array(input_array: Array) -> Array[float]` - Cleans and sorts an array, keeping only numeric values

### Basic Statistics

- `mean(data: Array[float]) -> float` - Arithmetic mean (average) of the dataset
- `median(data: Array[float]) -> float` - Middle value of a sorted dataset
- `variance(data: Array[float]) -> float` - Population variance of the dataset
- `standard_deviation(data: Array[float]) -> float` - Population standard deviation of the dataset
- `sample_variance(data: Array[float]) -> float` - Sample variance (with Bessel's correction)
- `sample_standard_deviation(data: Array[float]) -> float` - Sample standard deviation
- `median_absolute_deviation(data: Array[float]) -> float` - Robust measure of variability using median of absolute deviations
- `summary_statistics(data: Array[float]) -> Dictionary` - Comprehensive statistical summary including all basic statistics

### Error Functions

- `error_function(x: float) -> float` - Computes erf(x)
- `complementary_error_function(x: float) -> float` - Computes erfc(x) = 1 - erf(x)
- `error_function_inverse(y: float) -> float` - Inverse error function
- `complementary_error_function_inverse(y: float) -> float` - Inverse complementary error function

### Sampling (via StatMath.SamplingGen)

- `generate_samples(n_draws: int, dimensions: int = 1, method: SamplingMethod = SamplingMethod.RANDOM, starting_index: int = 0, sample_seed: int = -1) -> Variant` - Unified sampling function that returns Array[float] for 1D, Array[Vector2] for 2D, or Array[Array[float]] for higher dimensions
- `coordinated_shuffle(deck_size: int, method: SamplingMethod = SamplingMethod.SOBOL, point_index: int = 0, sample_seed: int = -1) -> Array[int]` - Performs coordinated Fisher-Yates shuffle using multi-dimensional sampling

**Sampling Methods:**
- `RANDOM` - Pseudo-random sampling
- `SOBOL` - Sobol quasi-random sequence
- `SOBOL_RANDOM` - Randomized Sobol sequence
- `HALTON` - Halton quasi-random sequence
- `HALTON_RANDOM` - Randomized Halton sequence
- `LATIN_HYPERCUBE` - Latin Hypercube space-filling design

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
        ```python
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

## Documentation

All functions are well-commented in the source code.  
For full details, see the scripts in `addons/godot-stat-math/core/`.

## License

Unlicense (public domain, see LICENSE)