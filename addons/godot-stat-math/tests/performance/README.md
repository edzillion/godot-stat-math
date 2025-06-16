# StatMath Performance Testing System

This directory contains the performance testing and baseline generation system for the StatMath addon.

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