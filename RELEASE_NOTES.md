# 🎲 Godot Stat Math - Alpha 0.0.2 Release Notes

**Release Date:** June 15, 2025  
**Status:** Alpha (Breaking Changes Expected)

---

## 🚀 What's New in Alpha 0.0.2

**Godot Stat Math** is a comprehensive statistical functions addon for Godot 4, providing everything from basic random number generation to advanced quasi-random sampling methods. This alpha release represents a major evolution with a completely rewritten sampling system.

### 🎯 **For Game Developers**

This addon is perfect for:
- **Procedural Generation**: Advanced sampling methods for creating varied, high-quality content
- **Game Balance**: Statistical analysis tools for analyzing player data and game metrics  
- **AI & Simulation**: Comprehensive probability distributions for realistic behaviors
- **Quality Assurance**: Reproducible random sequences for testing and debugging

### ✨ **Major Features**

#### 📊 **Complete Statistical Library**
- **30+ Probability Distributions**: Normal, Exponential, Gamma, Beta, Weibull, Pareto, Binomial, Poisson, and more
- **Statistical Analysis**: Mean, median, variance, standard deviation, and summary statistics
- **Advanced Math Functions**: Gamma function, Beta function, Error function, and their inverses
- **CDF & PPF Functions**: Cumulative distribution and quantile functions for probability calculations

#### 🎲 **Revolutionary Sampling System** (NEW in 0.0.2)
- **Unified API**: One function handles 1D, 2D, and up to 52-dimensional sampling
- **6 Sampling Methods**: Random, Sobol, Halton, Latin Hypercube, and randomized variants
- **Coordinated Shuffling**: Perfect for card games, deck management, and rare event simulation
- **Deterministic Sequences**: Reproducible results with starting index support

#### 🛠️ **Developer-Friendly Design**
- **Global Access**: All functions available via `StatMath.ModuleName.function_name()`
- **Type Safety**: Full static typing for better IDE support and error prevention
- **Comprehensive Seeding**: Control randomness at global or per-function level
- **Extensive Testing**: 37+ test cases ensure mathematical accuracy

---

## 🔥 **What Makes This Special**

### 🎯 **Unified Sampling API**
Before (0.0.1):
```gdscript
# Old way - multiple functions, inconsistent APIs
var points_1d = generate_samples_1d(100, method)
var points_2d = generate_samples_2d(100, method)  
# No support for higher dimensions
```

After (0.0.2):
```gdscript
# New way - one function, any dimensions
var points_1d: Array[float] = StatMath.SamplingGen.generate_samples(100, 1, method)
var points_2d: Array[Vector2] = StatMath.SamplingGen.generate_samples(100, 2, method)
var points_52d: Array = StatMath.SamplingGen.generate_samples(100, 52, method)
```

### 🃏 **Coordinated Shuffling**
Perfect for card games and rare event simulation:
```gdscript
# Deterministic, coordinated deck shuffles
var shuffled_deck: Array[int] = StatMath.SamplingGen.coordinated_shuffle(52, SamplingMethod.SOBOL)
```

### 📈 **Production-Ready Statistics**
```gdscript
# Analyze player performance data
var scores: Array[float] = [95.5, 87.2, 92.1, 88.8, 90.0, 78.3]
var clean_scores = StatMath.HelperFunctions.sanitize_numeric_array(raw_scores)
var summary = StatMath.BasicStats.summary_statistics(clean_scores)
# Returns: mean, median, variance, std_dev, min, max, count
```

---

## 💥 **Breaking Changes (Important!)**

**🚨 This is an alpha release with complete API changes for the sampling system.**

If you used v0.0.1 sampling functions, you MUST update your code:

```gdscript
# OLD (0.0.1) - Will not work
var samples = StatMath.SamplingGen.generate_samples_1d(100, method)

# NEW (0.0.2) - Required update  
var samples: Array[float] = StatMath.SamplingGen.generate_samples(100, 1, method)
```

**All other statistical functions remain unchanged and fully compatible.**

---

## 🎮 **Quick Start**

1. **Install**: Copy the addon to your `addons/` folder and enable in Project Settings
2. **Access**: Use `StatMath.ModuleName.function_name()` anywhere in your code
3. **Generate**: Create random numbers, analyze data, and sample high-quality sequences

```gdscript
# Basic usage examples
var random_damage: float = StatMath.Distributions.randf_normal(50.0, 10.0)
var crit_chance: int = StatMath.Distributions.randi_bernoulli(0.15)
var enemy_positions: Array[Vector2] = StatMath.SamplingGen.generate_samples(10, 2, SamplingMethod.SOBOL)
```

---

## ⚠️ **Alpha Software Notice**

**This is alpha software** - the API is actively evolving and breaking changes are expected in future releases. Perfect for:
- ✅ Prototyping and experimentation
- ✅ Learning advanced sampling techniques
- ✅ Non-critical game development

**Not recommended for:**
- ❌ Production games without expectation of updates
- ❌ Projects requiring API stability guarantees

---

## 🔗 **Links & Resources**

- **GitHub Repository**: [edzillion/godot-stat-math](https://github.com/edzillion/godot-stat-math)
- **Full Documentation**: See README.md for complete API reference
- **Issue Tracker**: Report bugs and request features on GitHub
- **Godot Version**: Requires Godot 4.0+

---

**Ready to revolutionize your game's random systems? Download Alpha 0.0.2 and explore the power of advanced statistical functions in Godot!** 🚀 