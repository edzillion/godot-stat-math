# res://addons/godot-stat-math/tests/performance/perf_test_base.gd
class_name PerfTestBase extends GdUnitTestSuite

## Base class for all performance test suites
##
## Each test suite now operates independently, saving its own results immediately
## to eliminate shared state conflicts between test suites.

# Configuration from centralized manager
const REGRESSION_THRESHOLD: float = PerfTestManager.REGRESSION_THRESHOLD
const WARMUP_ITERATIONS: int = PerfTestManager.WARMUP_ITERATIONS
const MEASUREMENT_ITERATIONS: int = PerfTestManager.MEASUREMENT_ITERATIONS

# Common test parameters that individual tests can override
const TEST_ITERATIONS: int = 100  # Number of function calls per performance test
const DATASET_SIZES: Array[int] = [100, 1000, 5000]  # Standard dataset sizes for stats

# Performance manager instance for this test suite
var _perf_manager: PerfTestManager

## Setup performance manager for this specific test suite
func before() -> void:
	_perf_manager = PerfTestManager.new()
	_perf_manager.set_module_name(get_module_name())
	print("🚀 Started independent performance testing for: %s" % get_module_name())

## Clean up this test suite independently
func after() -> void:
	var module_name: String = get_module_name()
	print("✅ Completed performance testing for: %s" % module_name)
	
	# Register completion with the global completion tracker
	await PerfTestManager.register_module_completion(module_name)

## Measure a test function's performance - delegates to manager
func _measure_test(test_name: String, test_func: Callable) -> Dictionary:
	return _perf_manager.measure_test(test_name, test_func)

## Load baseline data - delegates to manager
func _load_baseline() -> Dictionary:
	return _perf_manager.load_baseline()

## Check performance regression and save immediately - delegates to manager
func _check_performance_regression(module_name: String, test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> void:	
	var is_failure: bool = _perf_manager.check_performance_regression(module_name, test_name, current_results, baseline_data)
	assert_bool(is_failure).override_failure_message("Performance regression detected for '%s'" % test_name).is_false()
	
	

## Generate reproducible test data - delegates to manager
func _generate_test_data(size: int, seed: int = 12345) -> Array[float]:
	return _perf_manager.generate_test_data(size, seed)

## Get the module name for this test (override in subclasses)
func get_module_name() -> String:
	push_error("get_module_name() must be implemented by subclass")
	return "Unknown" 
