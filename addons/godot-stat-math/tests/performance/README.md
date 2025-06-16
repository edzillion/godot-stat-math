# SamplingGen Performance Testing

Simple baseline-driven performance regression detection for SamplingGen using GDUnit4.

## How It Works

1. **Generate Baseline**: Run `generate_baseline.gd` tool script in Godot editor
2. **Run Tests**: Use GDUnit4 to run `sampling_gen_perf_test.gd`
3. **Develop**: Make performance improvements
4. **Generate New Baseline**: Run tool script again when ready to update expectations

## Files

- `sampling_gen_perf_test.gd` - GDUnit4 performance test suite
- `generate_baseline.gd` - Tool script to generate baselines
- `baseline.json` - Current performance baseline (created by tool script)
- `archive/` - Previous baselines with timestamps
- `README.md` - This documentation

## Quick Start

```
# 1. Open Godot editor and load your project

# 2. Generate initial baseline:
#    - Open generate_baseline.gd in the editor
#    - Go to Tools > Execute Script
#    - Check output console for results

# 3. Run performance tests in GDUnit4 UI
#    - Tests will pass/fail based on 20% regression threshold

# 4. After making performance improvements:
#    - Run generate_baseline.gd again to update baseline
```

## Tool Script Usage

In the Godot editor:

1. **Open** `addons/godot-stat-math/tests/performance/generate_baseline.gd`
2. **Execute** via `Tools > Execute Script`
3. **Check Output** in the editor console

The tool script will:
- Archive existing baseline with date stamp (e.g., `baseline_2024-01-15.json`)
- Generate new baseline by measuring current performance
- Save new baseline for regression testing

## Configuration

Adjust thresholds in `sampling_gen_perf_test.gd`:

```gdscript
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 3           # JIT warmup runs
const MEASUREMENT_ITERATIONS: int = 5      # Statistical samples
```

## Test Coverage

- **generate_samples()**: 3 methods × 3 dimensions × 3 batch sizes = 27 tests
- **coordinated_shuffle()**: 3 methods × 3 deck sizes = 9 tests  
- **sample_indices()**: 3 selection strategies = 3 tests

**Total: 39 performance tests**

## Baseline Format

```json
{
  "generated_at": "2024-01-15T10:30:00",
  "tests": {
    "generate_samples_RANDOM_1d_256": 2.15,
    "generate_samples_RANDOM_2d_1024": 14.80,
    "coordinated_shuffle_SOBOL_52": 1.05,
    "sample_indices_FISHER_YATES_1000_100": 3.25
  }
}
```

Values are median execution times in milliseconds.

## Workflow

### Initial Setup
```
Run generate_baseline.gd tool script in Godot editor
```

### Development Cycle
```
# Make changes to SamplingGen
# Run GDUnit4 tests - they should pass if no regression

# After performance improvement:
# Run generate_baseline.gd tool script to update baseline expectations
```

### Test Results
```
=== Testing generate_samples Performance ===

--- Method: RANDOM ---
    📈 generate_samples_RANDOM_1d_256: 2.15 ms (baseline: 2.08 ms, +3.4%)
    📈 generate_samples_RANDOM_2d_1024: 18.50 ms (baseline: 14.80 ms, +25.0%)
    ❌ Performance regression detected: 25.0% slower than baseline
```

## Archive Management

The tool script automatically:
- Archives existing baseline with date stamp before creating new one
- Stores archives in `archive/` subdirectory
- Maintains history of performance baselines

Archives are named: `baseline_2024-01-15.json`

## Advantages of Tool Script

- ✅ **Cross-platform** - Works on all platforms Godot supports
- ✅ **No dependencies** - No need for PowerShell or external scripts
- ✅ **Direct access** - Can use StatMath autoload and all Godot APIs
- ✅ **Integrated** - Runs directly in the Godot editor
- ✅ **Simple** - Just open script and execute via Tools menu

## Customization

Edit `generate_baseline.gd` to:
- Add new test cases
- Adjust test parameters
- Change measurement methodology

The baseline generator and performance tests use identical test configurations to ensure consistency. 