# Changelog - Godot Stat Math

## [v0.0.6] - 2025-30-06

### 🔧 Fixes

#### Documentation
- **HTML Formatting** - Fixed HTML formatting issues in documentation files
- **API Documentation** - Improved class descriptions in PPF Functions module
- **Documentation Generation** - Enhanced documentation generation script with better RST file handling
- **Cross-References** - Fixed various cross-reference links in documentation modules
- **Module Documentation** - Updated and improved documentation for Sampling Generation, Statistics, and other core modules

#### Technical Improvements
- **Version Management** - Improved version bump and release process
- **Build Process** - Enhanced CI/CD workflows for better documentation publishing
- **Build Optimization** - Removed test data files from standard release build, reducing size from 1.19MB to 92KB

### 📝 Notes
- This is primarily a documentation and tooling improvement release
- No breaking changes to the API
- All core functionality remains the same as v0.0.5

---

## [v0.0.5] - 2025-28-6

### 🎉 Initial Alpha Release

**Godot Stat Math** - A comprehensive statistical functions addon for Godot 4.0+

### ✨ Features

#### Core Statistical Modules
- **Basic Statistics** - Essential statistical measures (mean, median, mode, variance, standard deviation, skewness, kurtosis)
- **Probability Distributions** - Support for normal, uniform, exponential, gamma, beta, chi-squared, and more
- **Cumulative Distribution Functions (CDF)** - Probability calculations for various distributions
- **Probability Mass/Density Functions (PMF/PDF)** - Probability function implementations
- **Percent Point Functions (PPF)** - Inverse CDF calculations (quantile functions)
- **Error Functions** - Mathematical error functions including erf, erfc, and inverse variants
- **Sampling & Generation** - Advanced sampling methods including Sobol sequences and Latin Hypercube

#### Advanced Sampling Methods
- **Sobol Sequences** - Low-discrepancy quasi-random sampling
- **Latin Hypercube Sampling** - Efficient stratified sampling for multidimensional spaces
- **Random Sampling** - Traditional pseudo-random number generation
- **Halton Sequences** - Another quasi-random sampling method

#### Utility Functions
- **Helper Functions** - Array sanitization, validation, and mathematical utilities
- **Statistical Validation** - Input parameter validation and error handling

### 🔧 Technical Features
- **Type Safety** - Full static typing throughout the codebase
- **Performance Optimized** - Efficient implementations suitable for game development
- **Global Access** - Available through `StatMath` singleton autoload
- **Comprehensive Testing** - Extensive test suite with mathematical property validation
- **Documentation** - Complete API documentation with examples

### 📦 Distribution Formats
- **Standard Edition** - Production-ready, optimized for end users
- **Debug Edition** - Includes full test suite and performance benchmarks

### 🎮 Game Development Focus
This library is designed specifically for game developers who need statistical functions for:
- Procedural generation algorithms
- Random event systems
- Statistical analysis of gameplay data
- AI behavior modeling
- Monte Carlo simulations

### ⚠️ Alpha Release Notes
- API may change without notice in future versions
- Designed for game development use cases (not scientific accuracy)
- For scientific applications, use dedicated Python libraries like SciPy

### 🚀 Quick Start
```gdscript
# Access statistical functions via StatMath singleton
var random_val: float = StatMath.Distributions.randf_normal(0.0, 1.0)
var mean_val: float = StatMath.BasicStats.mean([1.0, 2.0, 3.0, 4.0, 5.0])
var samples: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, StatMath.SamplingGen.SamplingMethod.SOBOL)
```

