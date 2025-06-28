# StatMath Performance Testing System

## Overview

The StatMath performance testing system provides automated baseline generation and regression detection for all StatMath modules. The system automatically manages baselines using successful test runs, eliminating the need for manual baseline generation.

## 🚀 **Automatic Baseline Generation**

The system automatically:
- ✅ Saves successful test runs as `pass_YYYY-MM-DD_HH-MM-SS.json`
- ✅ Updates `baseline.json` using the last 50 successful runs
- ✅ Uses robust median statistics for stable baselines
- ✅ Cleans up old snapshots automatically
- ✅ Applies hardware normalization for cross-machine consistency

## How It Works

### 1. **Test Execution**
- Run GDUnit4 performance tests normally: `addons/gdUnit4/runtest.cmd -a addons/godot-stat-math/tests/performance/`
- Each test measures performance and compares against current baseline
- Hardware normalization ensures consistent results across different machines

### 2. **Result Storage**
- **Successful runs**: Saved as `pass_YYYY-MM-DD_HH-MM-SS.json` + `latest.json`
- **Failed runs**: Saved as `fail_YYYY-MM-DD_HH-MM-SS.json` + `latest.json`
- Failed runs are saved for debugging but don't affect baseline calculations

### 3. **Automatic Baseline Updates**
- After each test run, the system automatically:
  - Loads up to 50 most recent successful runs (`pass_` files)
  - Calculates robust median statistics for each test
  - Updates `baseline.json` with new baseline values
  - Reports statistical confidence based on sample size

### 4. **File Management**
- Keeps 50 most recent `pass_` files for statistical robustness
- Optionally keeps `fail_` files (controlled by `KEEP_PREVIOUS_FAILURES` flag)
- Automatically cleans up old files to maintain storage efficiency

## File Structure

```
results/
├── baseline.json           # Current baseline (auto-generated from pass_ files)
├── latest.json             # Most recent test run results
├── pass_2025-01-15_10-30-45.json  # Successful test runs
├── pass_2025-01-15_11-15-20.json
├── fail_2025-01-15_09-45-10.json  # Failed test runs (optional)
└── ...
```

## Configuration

Key settings in `PerfTestManager`:

```gdscript
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const MAX_SNAPSHOTS: int = 50              # Keep 50 recent snapshots
const KEEP_PREVIOUS_FAILURES: bool = false # Save failure snapshots
```

## Performance Test Architecture

### Base Classes
- **`PerfTestBase`**: Base class for all performance test suites
- **`PerfTestManager`**: Centralized performance testing infrastructure

### Test Structure
```gdscript
func test_example_performance() -> void:
    var test_name: String = "example_operation"
    var baseline_data: Dictionary = _load_baseline()
    
    var current_results: Dictionary = _measure_test(test_name, func():
        # Your performance-critical code here
        for i in range(TEST_ITERATIONS):
            StatMath.SomeModule.some_function(test_data)
    )
    
    _check_performance_regression(get_module_name(), test_name, current_results, baseline_data)
```

## Hardware Normalization

The system automatically calibrates for hardware differences:
- **CPU Factor**: Based on floating-point operations benchmark
- **Memory Factor**: Based on array manipulation benchmark
- Tests are categorized as CPU-bound, memory-bound, or mixed workload
- Normalization ensures baselines are portable across development machines

## Statistical Robustness

- **Median-based baselines**: Robust against outliers
- **Confidence reporting**: Based on sample size (high ≥10, medium ≥5, low <5)
- **Variance detection**: Flags tests with high coefficient of variation (>15%)
- **Sample size tracking**: Reports statistical summary for each baseline update

## Migrating from Manual System

The old `generate_baseline.gd` script is **deprecated**. The new system:
- ❌ No more manual baseline generation
- ❌ No more `collect_performance_measurements()` methods
- ✅ Just run GDUnit4 tests normally
- ✅ Baselines update automatically

## Best Practices

1. **Run tests regularly** to build up statistical history
2. **Monitor variance** - investigate tests with high CV (>15%)
3. **Check confidence levels** - aim for ≥10 successful runs for reliable baselines
4. **Review failures** - failed tests indicate potential performance regressions
5. **Hardware consistency** - normalization helps, but consistent test environments are better

## Troubleshooting

### No Baseline Available
```
⚠️  No baseline data available for test_name - skipping regression check
```
**Solution**: Run tests successfully a few times to build initial baseline

### Low Statistical Confidence
```
🚨 Very limited data - results may be unstable (n=3)
```
**Solution**: Run more successful test cycles to increase sample size

### High Variance Warning
```
⚠️  High variance tests (CV > 15%): [test_name1, test_name2]
```
**Solution**: Investigate these tests for inconsistent performance patterns

## Architecture

### Individual Performance Test Suites
Each core StatMath module has its own performance test suite:

- `sampling_gen_perf_test.gd` - SamplingGen module tests
- `distributions_perf_test.gd` - Distributions module tests  
- `helper_functions_perf_test.gd` - HelperFunctions module tests
- `ppf_functions_perf_test.gd` - PpfFunctions module tests
- `basic_stats_perf_test.gd` - BasicStats module tests
- `cdf_functions_perf_test.gd` - CdfFunctions module tests
- `pmf_pdf_functions_perf_test.gd` - PmfPdfFunctions module tests
- `error_functions_perf_test.gd` - ErrorFunctions module tests

### Baseline Generation
**`generate_baseline.gd`** - Universal generator for all modules. All results are saved to a single `baseline.json` file with module prefixes (e.g., `samplinggen_generate_samples_RANDOM_1d_256`, `distributions_randf_normal`).

### Interface Pattern
Each performance test class implements:

```gdscript
## Public method for baseline generation - returns performance measurements
func collect_performance_measurements() -> Dictionary:
    var results: Dictionary = {}
    
    # Run performance tests and collect timing data
    var measurement: Dictionary = _measure_test("test_name", func():
        # Your test logic here
    )
    results["test_name"] = measurement.execution_time_ms
    
    return results
```

This allows the universal baseline generator to use the exact same test logic as the performance regression tests.

## Usage

### Generating Baselines

**Generate baselines for all modules:**
```
Run generate_baseline.gd (F6 in Godot)
```

This will run all configured modules and save results to a single `baseline.json` file.

### Running Performance Tests

Use GDUnit4 to run the individual performance test suites:
```
Run specific suite: sampling_gen_perf_test.gd
Run all suites: Select all *_perf_test.gd files
```

### Adding New Modules

1. Create a new performance test class with the interface pattern:
   ```gdscript
   class_name NewModulePerfTest extends GdUnitTestSuite
   
   func collect_performance_measurements() -> Dictionary:
       # Implement performance measurements
   ```

2. Add the module to `generate_baseline.gd`:
   ```gdscript
   {
       "name": "NewModule",
       "test_class": NewModulePerfTest
   }
   ```

## Archive Management

The system automatically:
- Saves timestamped results to `archive/modulename/` directories
- Keeps only the 3 most recent archive files per module
- Averages the 3 most recent results to create stable baselines
- Handles backward compatibility with old baseline formats

## Performance Monitoring

- **Measurement**: Median of 5 runs after 3 warmup iterations
- **Regression Threshold**: 20% slower than baseline triggers test failure
- **Batch Sizes**: Adjusted per module based on function complexity
- **Time-only**: Memory tracking was removed due to Godot GC limitations

## File Structure
```
performance/
├── README.md                           # This file
├── generate_baseline.gd               # Universal baseline generator
├── *_perf_test.gd                     # Individual test suites
├── baseline.json                      # Single baseline file (all modules)
└── archive/                           # Timestamped baseline history
    └── baseline_YYYY-MM-DD_HH-MM-SS.json
```

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