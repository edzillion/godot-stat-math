# Changelog - SamplingGen

## [Alpha] - 2024-12-XX

### 🚨 BREAKING CHANGES
- **API Change**: `generate_samples()` now supports 1D, 2D, and N-dimensional generation
  - Old: `generate_samples(count: int, strategy: SelectionStrategy) -> Array[Vector2]`
  - New: `generate_samples(count: int, strategy: SelectionStrategy, dimensions: int = 2, starting_index: int = 0) -> Array`
- **New Enum**: Added `COORDINATED_FISHER_YATES` to `SelectionStrategy`

### ✨ NEW FEATURES
- **N-Dimensional Sobol Sequences**: Support for up to 52-dimensional Sobol sequences using primitive polynomials
- **Coordinated Shuffling**: New `coordinated_shuffle()` method for deterministic Fisher-Yates shuffles
  - Generates N-dimensional Sobol points to control each swap decision
  - Perfect for rare event simulation (e.g., royal flush estimation)
- **Starting Index Support**: All methods now accept `starting_index` parameter for deterministic sequence generation
- **Threading Optimizations**: Automatic multi-threading for high-dimensional generation (8+ dimensions) and batch shuffles (4+ shuffles)

### 🔧 USAGE UPDATES

**Before:**
```gdscript
# Only 2D Sobol generation supported
var points = sampling_gen.generate_samples(100, SamplingGen.SelectionStrategy.SOBOL)
```

**After:**
```gdscript
# 1D generation
var values = sampling_gen.generate_samples(100, SamplingGen.SelectionStrategy.SOBOL, 1)

# 2D generation (unchanged behavior)
var points = sampling_gen.generate_samples(100, SamplingGen.SelectionStrategy.SOBOL, 2)

# N-dimensional generation (NEW)
var nd_points = sampling_gen.generate_samples(100, SamplingGen.SelectionStrategy.SOBOL, 51)

# Coordinated shuffling (NEW)
var shuffles = sampling_gen.coordinated_shuffle(52, 1000, 0)  # 1000 coordinated 52-card shuffles
```

### ⚡ PERFORMANCE
- All sampling operations now use `WorkerThreadPool` for parallel generation
- Thread-safe direction vector caching reduces memory allocations
- Significant performance improvements for N-dimensional and batch operations

### 🔬 TESTING
- Comprehensive tests for N-dimensional generation
- Coordinated shuffle validation
- Threading performance verification

---
*Note: This is alpha software. No backward compatibility guarantees. Update dependent code accordingly.* 