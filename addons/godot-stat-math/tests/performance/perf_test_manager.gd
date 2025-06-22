# res://addons/godot-stat-math/tests/performance/perf_test_manager.gd
class_name PerfTestManager extends RefCounted

## Centralized Performance Testing Infrastructure
##
## Each test suite saves its own results independently to an intermediate file.
## A final phase consolidates these files into a single `latest.json` and a timestamped snapshot.

# Common performance testing configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/results/baseline.json"
const RESULTS_DIR: String = "res://addons/godot-stat-math/tests/performance/results/"
const CORE_TEST_DIR: String = "res://addons/godot-stat-math/tests/performance/core/"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression (fallback for tests without statistical data)
const IMPROVEMENT_WARNING_THRESHOLD: float = 0.30  # 30% improvement triggers "consider updating baseline" warning
const WARMUP_ITERATIONS: int = 10
const MEASUREMENT_ITERATIONS: int = 5
const FUNCTION_CALLS_PER_MEASUREMENT: int = 100
const KEEP_PREVIOUS_FAILURES: bool = false
const MAX_SNAPSHOTS: int = 50  # Keep 50 most recent snapshots for robust statistics

# Dynamic threshold calculation parameters
const MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD: int = 5  # Minimum samples needed for dynamic thresholds
const MIN_THRESHOLD_PERCENT: float = 0.05  # Base minimum threshold (5%)
const ROBUST_MIN_THRESHOLD_PERCENT: float = 0.08  # Robust minimum threshold (8%) for more stable testing
const MAX_THRESHOLD_PERCENT: float = 0.25  # Maximum 25% threshold
const HIGH_CONFIDENCE_SAMPLES: int = 30  # 30+ samples = high confidence
const MEDIUM_CONFIDENCE_SAMPLES: int = 15  # 15+ samples = medium confidence

# Threshold refinement parameters
const THRESHOLD_SAFETY_BUFFER: float = 1.1  # 10% buffer for borderline cases
const STABLE_FUNCTION_CV_THRESHOLD: float = 0.05  # CV threshold for considering function "very stable"
const STABLE_FUNCTION_MIN_THRESHOLD: float = 0.12  # 12% minimum for very stable functions
const FAST_FUNCTION_THRESHOLD_MS: float = 0.5  # Functions under 0.5ms get special handling
const FAST_FUNCTION_MIN_THRESHOLD: float = 0.15  # 15% minimum for very fast functions

# Advanced threshold refinement for different volatility levels
const LOW_VOLATILITY_CV_THRESHOLD: float = 0.08  # Functions with CV < 8% are low volatility
const MEDIUM_VOLATILITY_CV_THRESHOLD: float = 0.15  # Functions with CV < 15% are medium volatility
const LOW_VOLATILITY_MIN_THRESHOLD: float = 0.15  # 15% minimum for low volatility functions
const MEDIUM_VOLATILITY_MIN_THRESHOLD: float = 0.20  # 20% minimum for medium volatility functions
const HIGH_VOLATILITY_MIN_THRESHOLD: float = 0.25  # 25% minimum for high volatility functions

# Mature baseline adjustments (for sample sizes >= 25)
const MATURE_BASELINE_SAMPLE_SIZE: int = 25  # Consider baseline "mature" at 25+ samples
const MATURE_BASELINE_MULTIPLIER: float = 1.3  # 30% higher thresholds for mature baselines

# Percentile-based threshold parameters
const PERCENTILE_THRESHOLD: float = 95.0  # Use 95th percentile (only 5% of runs slower)

# ========== STANDALONE STATISTICS ANALYSIS ==========

## Analyze measurements and return detailed statistics with dynamic threshold calculation
## This can be called independently without running performance tests
static func analyze_measurements(measurements: Array[float], baseline_median: float = NAN, print_details: bool = true) -> Dictionary:
	if measurements.is_empty():
		push_error("Cannot analyze empty measurements array")
		return {}
	
	# Calculate baseline median from measurements if not provided
	if is_nan(baseline_median):
		var sorted_measurements: Array[float] = measurements.duplicate()
		sorted_measurements.sort()
		baseline_median = StatMath.BasicStats.median(sorted_measurements)
	
	# Calculate basic statistics
	var mean: float = StatMath.BasicStats.mean(measurements)
	var std_dev: float = StatMath.BasicStats.standard_deviation(measurements)
	var cv: float = std_dev / baseline_median if baseline_median > 0.0 else 0.0
	var sample_size: int = measurements.size()
	
	# Calculate percentiles
	var sorted_measurements: Array[float] = measurements.duplicate()
	sorted_measurements.sort()
	var min_val: float = sorted_measurements[0]
	var max_val: float = sorted_measurements[-1]
	var p25: float = StatMath.BasicStats.percentile(sorted_measurements, 25.0)
	var p75: float = StatMath.BasicStats.percentile(sorted_measurements, 75.0)
	var p95: float = StatMath.BasicStats.percentile(sorted_measurements, 95.0)
	var p99: float = StatMath.BasicStats.percentile(sorted_measurements, 99.0)
	
	# Calculate dynamic threshold
	var dynamic_threshold: float = _calculate_dynamic_threshold(measurements, baseline_median)
	
	# Determine volatility level
	var volatility_level: String = "high"
	if cv < STABLE_FUNCTION_CV_THRESHOLD:
		volatility_level = "very stable"
	elif cv < LOW_VOLATILITY_CV_THRESHOLD:
		volatility_level = "low"
	elif cv < MEDIUM_VOLATILITY_CV_THRESHOLD:
		volatility_level = "medium"
	
	# Determine confidence level
	var confidence_level: String = "low"
	if sample_size >= HIGH_CONFIDENCE_SAMPLES:
		confidence_level = "high"
	elif sample_size >= MEDIUM_CONFIDENCE_SAMPLES:
		confidence_level = "medium"
	elif sample_size >= MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD:
		confidence_level = "limited"
	
	var analysis_result: Dictionary = {
		"sample_size": sample_size,
		"mean": mean,
		"median": baseline_median,
		"std_dev": std_dev,
		"coefficient_of_variation": cv,
		"min": min_val,
		"max": max_val,
		"range": max_val - min_val,
		"percentiles": {
			"p25": p25,
			"p75": p75,
			"p95": p95,
			"p99": p99
		},
		"dynamic_threshold": dynamic_threshold,
		"volatility_level": volatility_level,
		"confidence_level": confidence_level,
		"is_fast_function": baseline_median < FAST_FUNCTION_THRESHOLD_MS,
		"is_mature_baseline": sample_size >= MATURE_BASELINE_SAMPLE_SIZE,
		"can_use_dynamic_threshold": sample_size >= MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD
	}
	
	if print_details:
		print("📊 Measurement Analysis Results:")
		print("   Sample Size: %d (confidence: %s)" % [sample_size, confidence_level])
		print("   Central Tendency: mean=%.3fms, median=%.3fms" % [mean, baseline_median])
		print("   Variability: std=%.3fms, CV=%.1f%% (%s volatility)" % [std_dev, cv * 100, volatility_level])
		print("   Range: min=%.3fms, max=%.3fms (span: %.3fms)" % [min_val, max_val, max_val - min_val])
		print("   Percentiles: P25=%.3fms, P75=%.3fms, P95=%.3fms, P99=%.3fms" % [p25, p75, p95, p99])
		print("   Dynamic Threshold: %.1f%% (fast function: %s, mature: %s)" % [
			dynamic_threshold * 100, 
			"yes" if analysis_result.is_fast_function else "no",
			"yes" if analysis_result.is_mature_baseline else "no"
		])
	
	return analysis_result

## Analyze a specific test from the baseline file
static func analyze_baseline_test(test_name: String, print_details: bool = true) -> Dictionary:
	var baseline_data: Dictionary = _load_json_file(BASELINE_FILE)
	if baseline_data.is_empty():
		push_error("Could not load baseline file: %s" % BASELINE_FILE)
		return {}
	
	var baseline_tests: Dictionary = baseline_data.get("tests", {})
	if not baseline_tests.has(test_name):
		push_error("Test '%s' not found in baseline. Available tests: %s" % [test_name, str(baseline_tests.keys())])
		return {}
	
	var test_data: Dictionary = baseline_tests[test_name]
	var baseline_median: float = test_data.get("result_ms", 0.0)
	var sample_size: int = test_data.get("sample_size", 1)
	var cv: float = test_data.get("coefficient_of_variation", 0.0)
	var threshold: float = test_data.get("threshold_percent", REGRESSION_THRESHOLD)
	
	# Create a synthetic measurements array for analysis (limited info from baseline)
	# This is a simplified analysis since we don't have the raw measurements
	var analysis_result: Dictionary = {
		"test_name": test_name,
		"baseline_median": baseline_median,
		"sample_size": sample_size,
		"coefficient_of_variation": cv,
		"current_threshold": threshold,
		"status": test_data.get("status", "unknown"),
		"volatility_level": _get_volatility_level(cv),
		"confidence_level": _get_confidence_level(sample_size),
		"is_fast_function": baseline_median < FAST_FUNCTION_THRESHOLD_MS,
		"is_mature_baseline": sample_size >= MATURE_BASELINE_SAMPLE_SIZE,
		"can_use_dynamic_threshold": sample_size >= MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD
	}
	
	if print_details:
		print("📊 Baseline Test Analysis: %s" % test_name)
		print("   Baseline: %.3fms (n=%d, confidence: %s)" % [baseline_median, sample_size, analysis_result.confidence_level])
		print("   Variability: CV=%.1f%% (%s volatility)" % [cv * 100, analysis_result.volatility_level])
		print("   Current Threshold: %.1f%% (fast: %s, mature: %s)" % [
			threshold * 100,
			"yes" if analysis_result.is_fast_function else "no",
			"yes" if analysis_result.is_mature_baseline else "no"
		])
	
	return analysis_result

## Analyze all tests in the baseline file and return summary statistics
static func analyze_all_baseline_tests(print_summary: bool = true) -> Dictionary:
	var baseline_data: Dictionary = _load_json_file(BASELINE_FILE)
	if baseline_data.is_empty():
		push_error("Could not load baseline file: %s" % BASELINE_FILE)
		return {}
	
	var baseline_tests: Dictionary = baseline_data.get("tests", {})
	var baseline_meta: Dictionary = baseline_data.get("meta", {})
	
	if baseline_tests.is_empty():
		push_warning("No tests found in baseline file")
		return {}
	
	var test_analyses: Dictionary = {}
	var summary_stats: Dictionary = {
		"total_tests": 0,
		"high_confidence_tests": 0,
		"medium_confidence_tests": 0,
		"low_confidence_tests": 0,
		"very_stable_tests": 0,
		"low_volatility_tests": 0,
		"medium_volatility_tests": 0,
		"high_volatility_tests": 0,
		"fast_functions": 0,
		"mature_baselines": 0,
		"dynamic_threshold_eligible": 0,
		"avg_threshold": 0.0,
		"avg_cv": 0.0,
		"avg_sample_size": 0.0
	}
	
	var total_threshold: float = 0.0
	var total_cv: float = 0.0
	var total_sample_size: int = 0
	
	# Analyze each test
	for test_name in baseline_tests:
		var test_analysis: Dictionary = analyze_baseline_test(test_name, false)
		test_analyses[test_name] = test_analysis
		
		summary_stats.total_tests += 1
		
		# Count by confidence level
		match test_analysis.confidence_level:
			"high":
				summary_stats.high_confidence_tests += 1
			"medium":
				summary_stats.medium_confidence_tests += 1
			_:
				summary_stats.low_confidence_tests += 1
		
		# Count by volatility level
		match test_analysis.volatility_level:
			"very stable":
				summary_stats.very_stable_tests += 1
			"low":
				summary_stats.low_volatility_tests += 1
			"medium":
				summary_stats.medium_volatility_tests += 1
			_:
				summary_stats.high_volatility_tests += 1
		
		# Count special characteristics
		if test_analysis.is_fast_function:
			summary_stats.fast_functions += 1
		if test_analysis.is_mature_baseline:
			summary_stats.mature_baselines += 1
		if test_analysis.can_use_dynamic_threshold:
			summary_stats.dynamic_threshold_eligible += 1
		
		# Accumulate for averages
		total_threshold += test_analysis.current_threshold
		total_cv += test_analysis.coefficient_of_variation
		total_sample_size += test_analysis.sample_size
	
	# Calculate averages
	summary_stats.avg_threshold = total_threshold / summary_stats.total_tests
	summary_stats.avg_cv = total_cv / summary_stats.total_tests
	summary_stats.avg_sample_size = float(total_sample_size) / summary_stats.total_tests
	
	var result: Dictionary = {
		"baseline_meta": baseline_meta,
		"test_analyses": test_analyses,
		"summary": summary_stats
	}
	
	if print_summary:
		print("📊 Baseline Analysis Summary:")
		print("   Total Tests: %d" % summary_stats.total_tests)
		print("   Confidence Distribution: High=%d, Medium=%d, Low=%d" % [
			summary_stats.high_confidence_tests, 
			summary_stats.medium_confidence_tests, 
			summary_stats.low_confidence_tests
		])
		print("   Volatility Distribution: Very Stable=%d, Low=%d, Medium=%d, High=%d" % [
			summary_stats.very_stable_tests,
			summary_stats.low_volatility_tests,
			summary_stats.medium_volatility_tests,
			summary_stats.high_volatility_tests
		])
		print("   Special Characteristics: Fast=%d, Mature=%d, Dynamic Eligible=%d" % [
			summary_stats.fast_functions,
			summary_stats.mature_baselines,
			summary_stats.dynamic_threshold_eligible
		])
		print("   Averages: Threshold=%.1f%%, CV=%.1f%%, Sample Size=%.1f" % [
			summary_stats.avg_threshold * 100,
			summary_stats.avg_cv * 100,
			summary_stats.avg_sample_size
		])
	
	return result

## Helper function to get volatility level from coefficient of variation
static func _get_volatility_level(cv: float) -> String:
	if cv < STABLE_FUNCTION_CV_THRESHOLD:
		return "very stable"
	elif cv < LOW_VOLATILITY_CV_THRESHOLD:
		return "low"
	elif cv < MEDIUM_VOLATILITY_CV_THRESHOLD:
		return "medium"
	else:
		return "high"

## Helper function to get confidence level from sample size
static func _get_confidence_level(sample_size: int) -> String:
	if sample_size >= HIGH_CONFIDENCE_SAMPLES:
		return "high"
	elif sample_size >= MEDIUM_CONFIDENCE_SAMPLES:
		return "medium"
	elif sample_size >= MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD:
		return "limited"
	else:
		return "low"

# ========== COMPLETION TRACKING SYSTEM ==========

## Simple file-based completion tracking that works with GDUnit's test runner
## This replaces the signal-based approach which doesn't work in GDUnit context
static var _run_timestamp: String = ""
static var _disable_regression_checking: bool = true
static var _completion_tracker_initialized: bool = false

## Initialize completion tracking system
static func _initialize_completion_tracker() -> void:
	if _completion_tracker_initialized:
		return
	
	# Read regression checking setting from environment variable ONCE
	var regression_check_env: String = OS.get_environment("DISABLE_REGRESSION_CHECKING").to_lower()
	_disable_regression_checking = regression_check_env == "true" or _disable_regression_checking == true
	
	_run_timestamp = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	_completion_tracker_initialized = true
	
	# Clean up any orphaned intermediate files from previous incomplete runs
	_cleanup_orphaned_intermediate_files()
	
	print("🎯 Performance test run initialized:")
	print("   Run timestamp: %s" % _run_timestamp)
	print("   Regression Checking Disabled: %s" % _disable_regression_checking)

## Discover all test suite modules from the core directory
static func _discover_test_modules() -> Array[String]:
	var modules: Array[String] = []
	var dir: DirAccess = DirAccess.open(CORE_TEST_DIR)
	if dir == null:
		push_error("Cannot access core test directory: " + CORE_TEST_DIR)
		return modules
	
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.ends_with("_perf_test.gd"):
			var base_name: String = file_name.replace("_perf_test.gd", "")
			var module_name: String = _filename_to_module_name(base_name)
			modules.append(module_name)
		file_name = dir.get_next()
	
	modules.sort()
	return modules

## Convert filename to expected module name format
static func _filename_to_module_name(filename: String) -> String:
	match filename:
		"basic_stats": return "BasicStats"
		"cdf_functions": return "CdfFunctions"
		"distributions": return "Distributions"
		"error_functions": return "ErrorFunctions"
		"helper_functions": return "HelperFunctions"
		"pmf_pdf_functions": return "PmfPdfFunctions"
		"ppf_functions": return "PpfFunctions"
		"sampling_gen": return "SamplingGen"
		_:
			var parts: Array = filename.split("_")
			var result: String = ""
			for part in parts:
				if part.length() > 0:
					result += part.capitalize()
			return result

## Check if all modules have completed by scanning intermediate files
static func _check_all_modules_completed() -> void:
	if not _completion_tracker_initialized:
		_initialize_completion_tracker()
	
	# Get list of expected modules
	var expected_modules: Array[String] = _discover_test_modules()
	
	# Check which intermediate files exist
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	var completed_modules: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.ends_with("_%s.json" % _run_timestamp):
			# Extract module name from filename
			var module_name: String = file_name.replace("_%s.json" % _run_timestamp, "")
			if module_name in expected_modules:
				completed_modules.append(module_name)
		file_name = dir.get_next()
	
	print("📊 Completion check: %d/%d modules completed" % [completed_modules.size(), expected_modules.size()])
	
	# If all expected modules are complete, trigger consolidation immediately
	if completed_modules.size() >= expected_modules.size():
		print("🏁 All performance test modules completed! Starting consolidation...")
		_consolidate_run_results_immediate()

## Immediate consolidation without async - works in GDUnit context
static func _consolidate_run_results_immediate() -> void:
	print("📦 Consolidating results for run: %s" % _run_timestamp)
	
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		push_error("Cannot access results directory for consolidation: " + RESULTS_DIR)
		return

	# Find intermediate files for current run
	var intermediate_files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if file_name.ends_with("_%s.json" % _run_timestamp):
			intermediate_files.append(file_name)
		file_name = dir.get_next()
	
	if intermediate_files.is_empty():
		print("⚠️ No intermediate result files found for run: %s" % _run_timestamp)
		return

	var consolidated_tests: Dictionary = {}
	var total_tests: int = 0
	var failed_tests: int = 0
	var files_processed: int = 0

	# Load and merge all intermediate files
	for file in intermediate_files:
		var file_path: String = RESULTS_DIR.path_join(file)
		var module_data: Dictionary = _load_json_file(file_path)
		
		if module_data.is_empty() or not module_data.has("tests"):
			push_warning("Skipping invalid or empty result file: " + file)
			continue
		
		files_processed += 1

		for test_name in module_data.tests:
			consolidated_tests[test_name] = module_data.tests[test_name]
			total_tests += 1
			if module_data.tests[test_name].status == "fail":
				failed_tests += 1
		
	if consolidated_tests.is_empty():
		print("❌ No valid tests found in intermediate files for run: %s" % _run_timestamp)
		return
	
	print("📊 Processed %d intermediate files, consolidated %d tests" % [files_processed, total_tests])
	
	# Create consolidated result
	var consolidated_data: Dictionary = {
		"tests": consolidated_tests,
		"meta": {
			"type": "test_run",
			"generated_at": Time.get_datetime_string_from_system(),
			"original_timestamp": _run_timestamp,
			"total_tests": total_tests,
			"failed_tests": failed_tests,
			"passed_tests": total_tests - failed_tests,
			"modules_included": files_processed
		}
	}
	
	# Save results
	_save_json_file(RESULTS_DIR.path_join("latest.json"), consolidated_data)
	
	var has_failures: bool = failed_tests > 0
	if has_failures:
		print("\n❌ TEST RUN FAILED - %d of %d tests failed:" % [failed_tests, total_tests])
		_report_failed_tests(consolidated_tests)
	else:
		print("\n✅ TEST RUN PASSED - All %d tests passed!" % total_tests)
	
	# Save timestamped snapshot
	var snapshot_prefix: String = "fail" if has_failures else "pass"
	var snapshot_filepath: String = RESULTS_DIR.path_join("%s_%s.json" % [snapshot_prefix, _run_timestamp])
	
	if has_failures and not KEEP_PREVIOUS_FAILURES:
		print("🗑️ Discarding failure snapshot as per configuration.")
	else:
		_save_json_file(snapshot_filepath, consolidated_data)
		print("💾 Consolidated results: %s (%d tests, %d failures)" % [snapshot_filepath.get_file(), total_tests, failed_tests])

	# Clean up intermediate files immediately - THIS IS THE KEY FIX!
	_cleanup_intermediate_files_immediate(intermediate_files, dir)
	
	# Update baseline if successful
	if not has_failures:
		_update_baseline_from_snapshots()
		_cleanup_old_snapshots()
	
	print("🎉 Performance test run completed successfully!")
	print("   📊 Results saved in: %s" % RESULTS_DIR)
	print("   📈 Latest results: %slatest.json" % RESULTS_DIR)
	print("   📋 Baseline: %s" % BASELINE_FILE)

## Immediate cleanup of intermediate files - synchronous operation
static func _cleanup_intermediate_files_immediate(intermediate_files: Array[String], dir: DirAccess) -> void:
	print("🗑️  Cleaning up %d intermediate files..." % intermediate_files.size())
	
	var cleanup_success: int = 0
	var cleanup_failures: Array[String] = []
	
	# First pass - attempt to delete all files
	for file in intermediate_files:
		var err: Error = dir.remove(file)
		if err == OK:
			cleanup_success += 1
			print("🗑️  Cleaned up intermediate file: %s" % file)
		else:
			cleanup_failures.append(file)
			print("⚠️  Failed to clean up intermediate file: %s (error: %d)" % [file, err])
	
	# Retry failed deletions with a small delay
	if not cleanup_failures.is_empty():
		print("🔄 Retrying cleanup for %d failed files..." % cleanup_failures.size())
		
		# Small synchronous delay
		OS.delay_msec(200)
		
		var final_failures: Array[String] = []
		for file in cleanup_failures:
			var err: Error = dir.remove(file)
			if err == OK:
				cleanup_success += 1
				print("🗑️  Cleaned up intermediate file (retry): %s" % file)
			else:
				final_failures.append(file)
		
		if final_failures.is_empty():
			print("✅ All intermediate files cleaned up successfully (%d files)" % cleanup_success)
		else:
			push_error("❌ Failed to clean up %d intermediate files: %s" % [final_failures.size(), str(final_failures)])
	else:
		print("✅ All intermediate files cleaned up successfully (%d files)" % cleanup_success)

## Clean up orphaned intermediate files from incomplete test runs
static func _cleanup_orphaned_intermediate_files() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	var orphaned_files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		# Look for module intermediate files (format: ModuleName_timestamp.json)
		# These should not start with "pass_", "fail_", or be "latest.json", "baseline.json"
		if (file_name.ends_with(".json") and 
			not file_name.begins_with("pass_") and 
			not file_name.begins_with("fail_") and
			file_name != "latest.json" and 
			file_name != "baseline.json"):
			orphaned_files.append(file_name)
		file_name = dir.get_next()
	
	# Remove orphaned intermediate files
	for orphaned_file in orphaned_files:
		var err: Error = dir.remove(orphaned_file)
		if err == OK:
			print("🗑️  Cleaned up orphaned intermediate file: %s" % orphaned_file)
		else:
			push_warning("Failed to remove orphaned file: %s (error: %d)" % [orphaned_file, err])

## Convert PascalCase module name to snake_case for test naming
static func _module_name_to_snake_case(module_name: String) -> String:
	var result: String = ""
	for i in range(module_name.length()):
		var c: String = module_name[i]
		if c >= "A" and c <= "Z" and i > 0:
			result += "_"
		result += c.to_lower()
	return result

## Register module completion and check for final phase trigger
static func register_module_completion(module_name: String) -> void:
	if not _completion_tracker_initialized:
		_initialize_completion_tracker()
	
	print("✅ Module completed: %s" % module_name)
	
	# Check if all modules are now complete - this triggers immediate consolidation
	_check_all_modules_completed()

## Reset completion tracker (for testing purposes)
static func reset_completion_tracker() -> void:
	_completion_tracker_initialized = false

# ========== INDEPENDENT TEST SUITE INTERFACE ==========

# Instance-specific data (no more shared static state!)
var _module_results: Dictionary = {}  # This instance's results
var _module_name: String = ""

## Initialize hardware normalization for this instance
func _init() -> void:
	# Initialize completion tracker on first instance creation
	if not _completion_tracker_initialized:
		_initialize_completion_tracker()

## Set the module name for this test suite instance
func set_module_name(module_name: String) -> void:
	_module_name = module_name
	print("🚀 Module started: %s" % module_name)

## Measure a test function performance with warmup and multiple iterations
func measure_test(test_name: String, test_func: Callable) -> Dictionary:
	# Warmup runs
	for i in range(WARMUP_ITERATIONS):
		test_func.call()
	
	# Actual measurements
	var times: Array[float] = []
	for i in range(MEASUREMENT_ITERATIONS):
		var start_time: int = Time.get_ticks_usec()
		test_func.call()
		var end_time: int = Time.get_ticks_usec()
		
		var execution_time_ms: float = (end_time - start_time) / 1000.0
		times.append(execution_time_ms)
	
	# Return median time to reduce noise from outliers
	times.sort()
	var median_index: int = times.size() / 2
	return {"execution_time_ms": times[median_index]}

## Load baseline data from file
func load_baseline() -> Dictionary:
	if not FileAccess.file_exists(BASELINE_FILE):
		push_warning("No baseline file found at: %s" % BASELINE_FILE)
		return {}
	
	var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.READ)
	if file == null:
		push_error("Failed to read baseline file: %s" % BASELINE_FILE)
		return {}
	
	var json_string: String = file.get_as_text()
	file.close()
	
	var json: JSON = JSON.new()
	var parse_result: Error = json.parse(json_string)
	if parse_result != OK:
		push_error("Failed to parse baseline JSON: %s" % BASELINE_FILE)
		return {}
	
	return json.data

## Check performance regression and save result immediately (independent per suite)
func check_performance_regression(module_name: String, test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> bool:
	# Set module name if not already set
	if _module_name.is_empty():
		_module_name = module_name
	
	var baseline_meta: Dictionary = baseline_data.get("meta", {})
	var baseline_tests: Dictionary = baseline_data.get("tests", {})
	var expected_regressions: Array = baseline_meta.get("expected_regressions", [])
	
	var raw_time_ms: float = current_results.execution_time_ms
	var prefixed_test_name: String = "%s_%s" % [_module_name_to_snake_case(module_name), test_name]
	
	var baseline_time_ms: float = NAN
	var is_failure: bool = false
	var status: String = "pass"
	
	if not baseline_tests.is_empty():
		# Look for test with module prefix since all modules are in one baseline file
		if baseline_tests.has(prefixed_test_name):
			var baseline_test_data: Dictionary = baseline_tests[prefixed_test_name]
			baseline_time_ms = baseline_test_data.result_ms
			var time_change: float = (raw_time_ms - baseline_time_ms) / baseline_time_ms
			
			# Use dynamic threshold if available, fallback to fixed threshold
			var effective_threshold: float = REGRESSION_THRESHOLD
			if baseline_test_data.has("threshold_percent"):
				effective_threshold = baseline_test_data.threshold_percent
			
			# Always perform the regression calculation for consistent measurement overhead
			var calculated_failure: bool = time_change > effective_threshold
			
			if _disable_regression_checking:
				# Do all the same work but force pass result
				print("☑️ Regression checking disabled. Using current result for '%s'. (would be: %.1f%% change)" % [prefixed_test_name, time_change * 100.0])
				status = "pass (disabled)"
				is_failure = false  # Force pass regardless of calculation
			else:
				# Normal regression checking logic
				is_failure = calculated_failure
				
				if is_failure:
					if prefixed_test_name in expected_regressions:
						print("⚠️  Expected regression for '%s'. Marking as pass." % prefixed_test_name)
						is_failure = false # Override failure for overall run status
						status = "pass (expected)"
					else:
						status = "fail"
				elif time_change < -IMPROVEMENT_WARNING_THRESHOLD:
					var msg = "Significant improvement for '%s' (%.1f%%) - consider updating baseline." % [prefixed_test_name, time_change * 100.0]
					print("📈 %s" % msg)
					push_warning(msg)
				
				# Enhanced reporting with dynamic threshold info
				var threshold_info: String = ""
				if baseline_test_data.has("threshold_percent"):
					threshold_info = " (thresh: %.1f%%)" % (effective_threshold * 100.0)
				
				print("📊 %s: %.2f ms vs baseline %.2f ms (%.1f%% change)%s" % [
					test_name, raw_time_ms, baseline_time_ms, time_change * 100.0, threshold_info
				])
		else:
			if _disable_regression_checking:
				print("☑️ Regression checking disabled. No baseline found for '%s' - treating as new test." % prefixed_test_name)
			else:
				print("⚠️  No baseline found for %s - treating as new test" % prefixed_test_name)
			baseline_time_ms = raw_time_ms  # Use current as baseline for new tests
	else:
		if _disable_regression_checking:
			print("☑️ Regression checking disabled. No baseline data available for '%s'." % prefixed_test_name)
			status = "pass (disabled)"
		else: # This case handles when baseline_tests is empty
			print("⚠️  No baseline data available - treating as new test: %s" % prefixed_test_name)
		baseline_time_ms = raw_time_ms  # Use current as baseline
	
	# Store result for this module with threshold information
	var result_data: Dictionary = {
		"result_ms": raw_time_ms,
		"baseline_ms": baseline_time_ms,
		"diff_percent": ((raw_time_ms - baseline_time_ms) / baseline_time_ms) * 100.0 if not is_nan(baseline_time_ms) and baseline_time_ms > 0.0 else 0.0,
		"status": status,
	}
	
	# Include threshold information if available
	if not baseline_tests.is_empty() and baseline_tests.has(prefixed_test_name):
		var baseline_test_data: Dictionary = baseline_tests[prefixed_test_name]
		if baseline_test_data.has("threshold_percent"):
			result_data["threshold_percent"] = baseline_test_data.threshold_percent
			result_data["sample_size"] = baseline_test_data.get("sample_size", 0)
			result_data["coefficient_of_variation"] = baseline_test_data.get("coefficient_of_variation", 0.0)
	
	_module_results[prefixed_test_name] = result_data
	
	# Save immediately after each test
	_save_module_results()
	
	return is_failure

## Generate reproducible test data for performance tests
func generate_test_data(size: int, seed: int = 12345) -> Array[float]:
	var data: Array[float] = []
	data.resize(size)
	
	# Generate reproducible test data using fixed seed
	var rng: RandomNumberGenerator = RandomNumberGenerator.new()
	rng.seed = seed
	
	for i in range(size):
		data[i] = rng.randf_range(-100.0, 100.0)
	
	return data

## ========== INDEPENDENT MODULE RESULTS MANAGEMENT ==========

## Save this module's results immediately
func _save_module_results() -> void:
	if _module_results.is_empty():
		return
	
	# Ensure results directory exists
	var dir: DirAccess = DirAccess.open("res://")
	if not dir.dir_exists(RESULTS_DIR):
		dir.make_dir_recursive(RESULTS_DIR)
	
	# Save to module-specific intermediate file using the shared run timestamp
	var module_filepath: String = RESULTS_DIR + "%s_%s.json" % [_module_name, _run_timestamp]
	
	var module_data: Dictionary = {
		"module": _module_name,
		"timestamp": _run_timestamp,
		"tests": _module_results,
		"meta": {
			"generated_at": Time.get_datetime_string_from_system(),
			"total_tests": _module_results.size()
		}
	}
	
	_save_json_file(module_filepath, module_data)
	print("💾 Saved intermediate results for %s: %s" % [_module_name, module_filepath.get_file()])

## Report details of failed tests with their stats and thresholds
static func _report_failed_tests(consolidated_tests: Dictionary) -> void:
	var failed_test_details: Array[Dictionary] = []
	
	# Collect failed test information
	for test_name in consolidated_tests:
		var test_data: Dictionary = consolidated_tests[test_name]
		if test_data.status == "fail":
			failed_test_details.append({
				"name": test_name,
				"data": test_data
			})
	
	# Sort failed tests by severity (highest diff_percent first)
	failed_test_details.sort_custom(func(a, b): return a.data.diff_percent > b.data.diff_percent)
	
	# Report each failed test
	for i in range(failed_test_details.size()):
		var test_info: Dictionary = failed_test_details[i]
		var test_name: String = test_info.name
		var test_data: Dictionary = test_info.data
		
		var current_ms: float = test_data.get("result_ms", 0.0)
		var baseline_ms: float = test_data.get("baseline_ms", 0.0)
		var diff_percent: float = test_data.get("diff_percent", 0.0)
		var threshold_percent: float = test_data.get("threshold_percent", REGRESSION_THRESHOLD) * 100.0
		var sample_size: int = test_data.get("sample_size", 0)
		var cv: float = test_data.get("coefficient_of_variation", 0.0) * 100.0
		
		print("   %d. %s:" % [i + 1, test_name])
		print("      Current: %.3f ms | Baseline: %.3f ms | Change: %.1f%%" % [current_ms, baseline_ms, diff_percent])
		print("      Threshold: %.1f%% | Sample size: %d | CV: %.1f%%" % [threshold_percent, sample_size, cv])

## Calculate dynamic threshold for a test based on percentile analysis with refinements
static func _calculate_dynamic_threshold(measurements: Array[float], baseline_median: float) -> float:
	var sample_size: int = measurements.size()
	
	# If insufficient data, use fallback threshold
	if sample_size < MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD:
		return REGRESSION_THRESHOLD
	
	# Sort measurements for percentile calculation
	var sorted_measurements: Array[float] = measurements.duplicate()
	sorted_measurements.sort()
	
	# Calculate coefficient of variation for stability assessment
	var cv: float = StatMath.BasicStats.standard_deviation(measurements) / baseline_median
	
	# Use 95th percentile - only 5% of historical runs were slower than this
	var percentile_95_threshold: float = StatMath.BasicStats.percentile(sorted_measurements, PERCENTILE_THRESHOLD)
	var percentile_threshold_percent: float = (percentile_95_threshold - baseline_median) / baseline_median
	
	# Apply confidence interval adjustment based on sample size
	# Smaller samples get slightly higher thresholds due to uncertainty
	var confidence_multiplier: float = 1.0
	if sample_size < MEDIUM_CONFIDENCE_SAMPLES:
		confidence_multiplier = 1.2  # 20% higher threshold for small samples
	elif sample_size < HIGH_CONFIDENCE_SAMPLES:
		confidence_multiplier = 1.1  # 10% higher threshold for medium samples
	
	var calculated_threshold: float = percentile_threshold_percent * confidence_multiplier
	
	# REFINEMENT 1: Use robust minimum threshold instead of aggressive 5%
	var base_min_threshold: float = ROBUST_MIN_THRESHOLD_PERCENT
	
	# REFINEMENT 2: Add safety buffer for borderline cases
	calculated_threshold = calculated_threshold * THRESHOLD_SAFETY_BUFFER
	
	# REFINEMENT 3: Special handling for very fast functions (under 0.5ms)
	# Small absolute variations create large percentage changes in fast functions
	if baseline_median < FAST_FUNCTION_THRESHOLD_MS:
		base_min_threshold = max(base_min_threshold, FAST_FUNCTION_MIN_THRESHOLD)
	
	# REFINEMENT 4: Advanced CV-based threshold scaling with volatility levels
	# Different minimum thresholds based on function volatility patterns
	var volatility_min_threshold: float = base_min_threshold
	
	if cv < STABLE_FUNCTION_CV_THRESHOLD:
		# Very stable functions (CV < 5%) - original logic
		volatility_min_threshold = max(volatility_min_threshold, STABLE_FUNCTION_MIN_THRESHOLD)
	elif cv < LOW_VOLATILITY_CV_THRESHOLD:
		# Low volatility functions (CV 5-8%) - need higher thresholds
		volatility_min_threshold = max(volatility_min_threshold, LOW_VOLATILITY_MIN_THRESHOLD)
	elif cv < MEDIUM_VOLATILITY_CV_THRESHOLD:
		# Medium volatility functions (CV 8-15%) - moderate thresholds
		volatility_min_threshold = max(volatility_min_threshold, MEDIUM_VOLATILITY_MIN_THRESHOLD)
	else:
		# High volatility functions (CV > 15%) - standard thresholds
		volatility_min_threshold = max(volatility_min_threshold, HIGH_VOLATILITY_MIN_THRESHOLD)
	
	# REFINEMENT 5: Mature baseline adjustment
	# With 25+ samples, we have high confidence in the baseline but need more tolerance
	# for natural performance variation in production environments
	if sample_size >= MATURE_BASELINE_SAMPLE_SIZE:
		volatility_min_threshold = volatility_min_threshold * MATURE_BASELINE_MULTIPLIER
	
	# Apply the refined minimum threshold
	var final_threshold: float = max(calculated_threshold, volatility_min_threshold)
	
	# Clamp to maximum bounds
	final_threshold = min(final_threshold, MAX_THRESHOLD_PERCENT)
	
	return final_threshold

## Helper to load a JSON file and return its data
static func _load_json_file(filepath: String) -> Dictionary:
	if not FileAccess.file_exists(filepath):
		push_warning("File not found: " + filepath)
		return {}

	var file: FileAccess = FileAccess.open(filepath, FileAccess.READ)
	if file == null:
		push_error("Failed to read file: " + filepath)
		return {}

	var json_string: String = file.get_as_text()
	file.close()
	
	var json: JSON = JSON.new()
	var err: Error = json.parse(json_string)
	if err != OK:
		push_error("Failed to parse JSON in %s: %s" % [filepath, json.get_error_message()])
		return {}
	
	return json.data

## Helper function to save JSON files
static func _save_json_file(filepath: String, data: Dictionary) -> void:
	var file: FileAccess = FileAccess.open(filepath, FileAccess.WRITE)
	if file == null:
		push_error("Failed to save file: " + filepath)
		return
	
	file.store_string(JSON.stringify(data, "\t"))
	file.close()

## Update baseline from successful snapshots automatically
static func _update_baseline_from_snapshots() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		push_error("Cannot access results directory: " + RESULTS_DIR)
		return
	
	# Get all pass_ files (successful test runs) - fail_ files are intentionally ignored
	var pass_files: Array[String] = []
	dir.list_dir_begin()
	var current_file: String = dir.get_next()
	
	while current_file != "":
		if current_file.begins_with("pass_") and current_file.ends_with(".json"):
			pass_files.append(current_file)
		current_file = dir.get_next()
	
	if pass_files.is_empty():
		print("⚠️  No successful test runs found - baseline unchanged")
		return
	
	# Sort and use up to MAX_SNAPSHOTS most recent successful runs
	pass_files.sort()
	var recent_files: Array[String] = pass_files.slice(-MAX_SNAPSHOTS) if pass_files.size() > MAX_SNAPSHOTS else pass_files
	
	print("📊 Updating baseline from %d successful test runs" % recent_files.size())
	
	# Load and accumulate results using statistical analysis
	var test_data_arrays: Dictionary = {}  # test_name -> Array[float] of all measurements
	
	for results_file in recent_files:
		var file_path: String = RESULTS_DIR + results_file
		var file_data: Dictionary = _load_json_file(file_path)
		if file_data.is_empty() or not file_data.has("tests"):
			continue
		
		# Collect each test result into arrays for statistical analysis
		for test_name in file_data.tests:
			var test_data = file_data.tests[test_name]
			var execution_time: float = test_data.result_ms
			
			if not test_data_arrays.has(test_name):
				test_data_arrays[test_name] = []
			
			test_data_arrays[test_name].append(execution_time)
	
	if test_data_arrays.is_empty():
		print("❌ No valid test data found in successful runs")
		return
	
	# Calculate robust statistics and dynamic thresholds using StatMath library
	var baseline_tests: Dictionary = {}
	
	for test_name in test_data_arrays:
		var measurements: Array[float] = []
		for measurement: float in test_data_arrays[test_name]:
			measurements.append(measurement)
		
		if measurements.is_empty():
			continue
		
		# Use median for robustness against outliers
		var test_median: float = StatMath.BasicStats.median(measurements)
		var test_mean: float = StatMath.BasicStats.mean(measurements)
		var test_std: float = StatMath.BasicStats.standard_deviation(measurements) if measurements.size() > 1 else 0.0
		var coefficient_of_variation: float = (test_std / test_mean) if test_mean > 0.0 else 0.0
		
		# Calculate dynamic threshold based on statistical analysis
		var dynamic_threshold: float = _calculate_dynamic_threshold(measurements, test_median)
		
		baseline_tests[test_name] = {
			"result_ms": test_median,
			"baseline_ms": test_median, 
			"diff_percent": 0.0,
			"status": "pass",
			"threshold_percent": dynamic_threshold,
			"sample_size": measurements.size(),
			"coefficient_of_variation": coefficient_of_variation
		}
	
	# Save the updated baseline with dynamic thresholds
	var baseline_data: Dictionary = {
		"tests": baseline_tests,
		"meta": {
			"expected_regressions": [],
			"type": "automatic_baseline_percentile_based",
			"generated_at": Time.get_datetime_string_from_system(),
			"total_tests": baseline_tests.size(),
			"source_snapshots": recent_files.size(),
			"statistical_method": "percentile_based_thresholds"
		}
	}
	
	_save_json_file(BASELINE_FILE, baseline_data)
	print("💾 Updated baseline from %d successful runs (%d tests)" % [recent_files.size(), baseline_tests.size()])

## Clean up old snapshot files to maintain MAX_SNAPSHOTS limit
static func _cleanup_old_snapshots() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	# Get all pass_ and fail_ files separately
	var pass_files: Array[String] = []
	var fail_files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.begins_with("pass_") and file_name.ends_with(".json"):
			pass_files.append(file_name)
		elif file_name.begins_with("fail_") and file_name.ends_with(".json"):
			fail_files.append(file_name)
		file_name = dir.get_next()
	
	# Clean up pass_ files (keep MAX_SNAPSHOTS most recent)
	pass_files.sort()
	while pass_files.size() > MAX_SNAPSHOTS:
		var oldest_file: String = pass_files.pop_front()
		dir.remove(oldest_file)
		print("🗑️  Removed old success snapshot: %s" % oldest_file)
	
	# Clean up fail_ files (keep MAX_SNAPSHOTS most recent if KEEP_PREVIOUS_FAILURES)
	if KEEP_PREVIOUS_FAILURES:
		fail_files.sort()
		while fail_files.size() > MAX_SNAPSHOTS:
			var oldest_file: String = fail_files.pop_front()
			dir.remove(oldest_file)
			print("🗑️  Removed old failure snapshot: %s" % oldest_file)
