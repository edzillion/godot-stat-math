# Changelog - Godot Stat Math

## [Alpha 0.0.2] - 2025-06-15

### 🚨 BREAKING CHANGES
- **Complete SamplingGen API Rewrite**: All sampling functions now use a single unified interface
  - Removed: `generate_samples_1d()`, `generate_samples_2d()`, and all 2D helper functions
  - **New**: `generate_samples(n_draws: int, dimensions: int, method: SamplingMethod, starting_index: int = 0, sample_seed: int = -1)`
- **No Backward Compatibility**: This is a complete rewrite with zero compatibility guarantees for SamplingGen module

### ✨ NEW FEATURES

#### SamplingGen Module
- **Unified API**: Single function handles 1D, 2D, and N-dimensional generation (up to 52D)
- **Universal starting_index Support**: All sampling methods and dimensions support deterministic sequence continuation
- **N-Dimensional Sobol Sequences**: Full support using primitive polynomials for up to 52 dimensions
- **Coordinated Shuffling**: Deterministic Fisher-Yates shuffles for rare event simulation
- **Smart Return Types**: 
  - 1D: Returns `Array[float]`
  - 2D: Returns `Array[Vector2]` 
  - N-D: Returns `Array[Array[float]]`

#### Complete Statistical Library
- **8 Core Modules**: Distributions, CDF Functions, PPF Functions, PMF/PDF Functions, Error Functions, Helper Functions, Basic Stats, SamplingGen
- **30+ Distribution Functions**: Normal, Exponential, Gamma, Beta, Weibull, Pareto, Binomial, Poisson, and more
- **Global StatMath Singleton**: All functions accessible via `StatMath.ModuleName.function_name()`
- **Reproducible Results**: Comprehensive seeding system with project settings integration
- **Type Safety**: Full static typing throughout the codebase

### 🏗️ ARCHITECTURE
- **Single Code Path**: All sampling dimensions use the same N-dimensional implementation
- **No Code Duplication**: Eliminated separate 1D/2D implementations in SamplingGen
- **Threading Optimizations**: Automatic multi-threading for high-dimensional generation (3+ dimensions)
- **Memory Pooling**: Efficient deck pooling for shuffle operations
- **Thread-Safe Caching**: Efficient Sobol direction vector management

### 🔧 USAGE EXAMPLES

```gdscript
# Random number generation from various distributions
var normal_val: float = StatMath.Distributions.randf_normal(0.0, 1.0)
var weibull_val: float = StatMath.Distributions.randf_weibull(2.0, 1.5)
var binomial_val: int = StatMath.Distributions.randi_binomial(0.3, 10)

# Statistical analysis
var data: Array[float] = [95.5, 87.2, 92.1, 88.8, 90.0]
var avg: float = StatMath.BasicStats.mean(data)
var std_dev: float = StatMath.BasicStats.standard_deviation(data)

# Advanced sampling (NEW UNIFIED API)
var sobol_1d: Array[float] = StatMath.SamplingGen.generate_samples(100, 1, StatMath.SamplingGen.SamplingMethod.SOBOL)
var sobol_2d: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
var sobol_nd: Array = StatMath.SamplingGen.generate_samples(100, 10, StatMath.SamplingGen.SamplingMethod.SOBOL)

# Coordinated shuffling for rare events
var shuffled_deck: Array[int] = StatMath.SamplingGen.coordinated_shuffle(52, StatMath.SamplingGen.SamplingMethod.SOBOL)
```

### ⚡ PERFORMANCE
- **Optimized Mathematical Functions**: Lanczos approximation for Gamma function, numerical integration for Beta functions
- **Unified Sampling Implementation**: Single optimized code path for all dimensions
- **Thread-Safe Operations**: WorkerThreadPool integration for high-dimensional cases
- **Memory Efficiency**: Pooled memory management for frequently used operations

### 🧪 TESTING
- **Comprehensive Test Suite**: GDUnit4 integration with extensive coverage
- **37 SamplingGen Test Cases**: All passing with sequence continuity validation  
- **Mathematical Accuracy**: Tests validate statistical function correctness within acceptable tolerances
- **Integration Tests**: Full addon functionality verified

### 🔧 DEVELOPER EXPERIENCE
- **Static Typing**: All functions use proper GDScript typing for better IDE support
- **Error Handling**: Comprehensive error messages with parameter validation
- **Documentation**: Inline comments and examples throughout codebase
- **Reproducible Results**: Global and per-function seeding options

### 📦 PROJECT STRUCTURE
- **Proper File Headers**: All core files include full resource paths
- **Consistent Versioning**: Plugin and project versions synchronized
- **CI/CD Integration**: Automated build and test workflows
- **Clean Codebase**: Follows GDScript style guidelines and best practices

---

**⚠️ ALPHA SOFTWARE NOTICE**

This is alpha software with no backward compatibility guarantees. The API is actively evolving and breaking changes are expected. Use in production at your own risk.

**Migration from 0.0.1**: All SamplingGen function calls must be updated to use the new unified `generate_samples()` API.

---

## Previous Versions

### [0.0.1] - 2024-12-XX (DELETED)
*This version was incorrectly tagged as "beta" and has been removed from git history.* 