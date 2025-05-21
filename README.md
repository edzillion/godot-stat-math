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