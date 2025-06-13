# Changelog - SamplingGen

## [Alpha] - 2024-12-XX

### 🚨 BREAKING CHANGES
- **Complete API Rewrite**: All sampling functions now use a single unified interface
  - Removed: `generate_samples_1d()`, `generate_samples_2d()`, and all 2D helper functions
  - **New**: `generate_samples(n_draws: int, dimensions: int, method: SamplingMethod, starting_index: int = 0, sample_seed: int = -1)`
- **No Backward Compatibility**: This is a complete rewrite with zero compatibility guarantees

### ✨ NEW FEATURES
- **Unified API**: Single function handles 1D, 2D, and N-dimensional generation (up to 52D)
- **Universal starting_index Support**: All sampling methods and dimensions support deterministic sequence continuation
- **N-Dimensional Sobol Sequences**: Full support using primitive polynomials for up to 52 dimensions
- **Coordinated Shuffling**: Deterministic Fisher-Yates shuffles for rare event simulation
- **Smart Return Types**: 
  - 1D: Returns `Array[float]`
  - 2D: Returns `Array[Vector2]` 
  - N-D: Returns `Array[Array[float]]`

### 🏗️ ARCHITECTURE
- **Single Code Path**: All dimensions use the same N-dimensional implementation
- **No Code Duplication**: Eliminated separate 1D/2D implementations entirely
- **Threading Optimizations**: Automatic multi-threading for high-dimensional generation (3+ dimensions)
- **Simplified Codebase**: Removed over 200 lines of redundant helper functions

### 🔧 USAGE

```gdscript
# 1D generation
var values: Array[float] = StatMath.SamplingGen.generate_samples(100, 1, SamplingMethod.SOBOL)

# 2D generation  
var points: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, SamplingMethod.SOBOL)

# N-dimensional generation
var nd_points: Array = StatMath.SamplingGen.generate_samples(100, 51, SamplingMethod.SOBOL)

# With starting_index for sequence continuation
var continuation: Array[float] = StatMath.SamplingGen.generate_samples(50, 1, SamplingMethod.SOBOL, 100)

# Coordinated shuffling for rare events
var shuffles: Array = StatMath.SamplingGen.coordinated_batch_shuffles(52, 1000, SamplingMethod.SOBOL)
```

### ⚡ PERFORMANCE
- **Unified Implementation**: Single optimized code path for all dimensions
- **Thread-Safe Caching**: Efficient Sobol direction vector management
- **Parallel Generation**: WorkerThreadPool integration for high-dimensional cases

### 🧪 TESTING
- **37 Test Cases**: All passing with comprehensive coverage
- **Unified API Tests**: All legacy function calls converted to new API
- **Starting Index Validation**: Sequence continuity verified across all methods

---
*Note: This is alpha software with no backward compatibility guarantees. Update all dependent code to use the new unified API.* 