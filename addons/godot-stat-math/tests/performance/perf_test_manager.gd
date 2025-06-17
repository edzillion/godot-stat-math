# res://addons/godot-stat-math/tests/performance/perf_test_manager.gd
class_name PerfTestManager extends RefCounted

## Centralized Performance Testing Infrastructure
##
## Each test suite saves its own results independently to eliminate shared state conflicts.
## A collation process merges module results into consolidated files for baseline generation.
## The completion tracking system monitors all test suites and triggers final phase actions.

# Common performance testing configuration
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/results/baseline.json"
const RESULTS_DIR: String = "res://addons/godot-stat-math/tests/performance/results/"
const CORE_TEST_DIR: String = "res://addons/godot-stat-math/tests/performance/core/"
const REGRESSION_THRESHOLD: float = 0.20  # 20% slower = regression
const WARMUP_ITERATIONS: int = 10
const MEASUREMENT_ITERATIONS: int = 5
const FUNCTION_CALLS_PER_MEASUREMENT: int = 100
const KEEP_PREVIOUS_FAILURES: bool = false
const MAX_SNAPSHOTS: int = 50  # Keep 50 most recent snapshots for robust statistics

# Hardware normalization constants
const BASELINE_CPU_SCORE: float = 5000000.0  # Reference CPU performance score (ops/sec)
const BASELINE_MEMORY_SCORE: float = 25.0  # Reference memory performance score (MB/s)

# Test categorization for targeted normalization
const CPU_BOUND_TESTS: Array[String] = [
	"distributions_", "cdffunctions_", "pmfpdffunctions_", "ppffunctions_", 
	"errorfunctions_", "helperfunctions_gamma", "helperfunctions_beta", 
	"helperfunctions_incomplete_beta", "helperfunctions_binomial"
]

const MEMORY_BOUND_TESTS: Array[String] = [
	"basicstats_", "samplinggen_", "helperfunctions_sanitize"
]

# ========== COMPLETION TRACKING SYSTEM ==========

## Static completion tracker - shared across all test suite instances
static var _discovered_modules: Array[String] = []
static var _completed_modules: Array[String] = []
static var _completion_tracker_initialized: bool = false
static var _final_phase_triggered: bool = false

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
			# Extract module name from filename and convert to expected format
			var base_name: String = file_name.replace("_perf_test.gd", "")
			var module_name: String = _filename_to_module_name(base_name)
			modules.append(module_name)
		file_name = dir.get_next()
	
	modules.sort()
	return modules

## Convert filename to expected module name format
static func _filename_to_module_name(filename: String) -> String:
	match filename:
		"basic_stats":
			return "BasicStats"
		"cdf_functions":
			return "CdfFunctions"
		"distributions":
			return "Distributions"
		"error_functions":
			return "ErrorFunctions"
		"helper_functions":
			return "HelperFunctions"
		"pmf_pdf_functions":
			return "PmfPdfFunctions"
		"ppf_functions":
			return "PpfFunctions"
		"sampling_gen":
			return "SamplingGen"
		_:
			# Fallback: convert snake_case to PascalCase
			var parts: Array = filename.split("_")
			var result: String = ""
			for part in parts:
				if part.length() > 0:
					result += part.capitalize()
			return result

## Initialize completion tracking system
static func _initialize_completion_tracker() -> void:
	if _completion_tracker_initialized:
		return
	
	_discovered_modules = _discover_test_modules()
	_completed_modules.clear()
	_final_phase_triggered = false
	_completion_tracker_initialized = true
	
	print("🎯 Performance test run initialized:")
	print("   Expected modules: %s" % str(_discovered_modules))
	print("   Total modules: %d" % _discovered_modules.size())

## Register module completion and check for final phase trigger
static func register_module_completion(module_name: String) -> void:
	if not _completion_tracker_initialized:
		_initialize_completion_tracker()
	
	if module_name in _completed_modules:
		return  # Already registered
	
	if not module_name in _discovered_modules:
		push_warning("Unknown module completed: %s (expected: %s)" % [module_name, str(_discovered_modules)])
		return
	
	_completed_modules.append(module_name)
	print("✅ Module completed: %s (%d/%d)" % [module_name, _completed_modules.size(), _discovered_modules.size()])
	
	# Check if all modules are complete
	if _completed_modules.size() == _discovered_modules.size() and not _final_phase_triggered:
		await _trigger_final_phase()

## Trigger final phase actions when all test suites are complete
static func _trigger_final_phase() -> void:
	if _final_phase_triggered:
		return
	
	_final_phase_triggered = true
	print("🏁 All performance test suites completed! Triggering final phase...")
	
	# Wait a moment for any pending operations to complete
	await Engine.get_main_loop().process_frame
	await Engine.get_main_loop().process_frame
	
	# Force final collation of any remaining results
	await _collate_all_pending_results()
	
	# Generate final completion report
	_generate_completion_report()
	
	# Additional wait to ensure all file operations complete
	await Engine.get_main_loop().process_frame
	await Engine.get_main_loop().create_timer(0.1).timeout
	
	print("🎉 Performance test run completed successfully!")
	print("   📊 Results saved in: %s" % RESULTS_DIR)
	print("   📈 Latest results: %slatest.json" % RESULTS_DIR)
	print("   📋 Baseline: %s" % BASELINE_FILE)

## Force collation of any pending module results
static func _collate_all_pending_results() -> void:
	print("📦 Final collation of pending results...")
	_collate_module_results_final()
	
	# Wait for collation to complete
	await Engine.get_main_loop().create_timer(0.5).timeout

## Generate final completion report
static func _generate_completion_report() -> void:
	var is_success: bool = _completed_modules.size() == _discovered_modules.size()
	
	# Just log completion status - no need to save completion reports
	if is_success:
		print("✅ All %d modules completed successfully" % _completed_modules.size())
	else:
		print("❌ Only %d of %d modules completed" % [_completed_modules.size(), _discovered_modules.size()])

## Reset completion tracker (for testing purposes)
static func reset_completion_tracker() -> void:
	_completion_tracker_initialized = false
	_discovered_modules.clear()
	_completed_modules.clear()
	_final_phase_triggered = false

# Instance-specific data (no more shared static state!)
var cpu_factor: float = 1.0
var memory_factor: float = 1.0
var _module_results: Dictionary = {}  # This instance's results
var _module_name: String = ""
var _timestamp: String = ""

## Initialize hardware normalization for this instance
func _init() -> void:
	# Initialize completion tracker on first instance creation
	if not _completion_tracker_initialized:
		_initialize_completion_tracker()
	
	_calibrate_hardware()
	_timestamp = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")

## Set the module name for this test suite instance
func set_module_name(module_name: String) -> void:
	_module_name = module_name

## Hardware calibration for this instance
func _calibrate_hardware() -> void:
	print("🔧 Calibrating hardware performance...")
	
	# Run CPU benchmark (drop first 10 results)
	var cpu_scores: Array[float] = []
	for i in range(15):  # 10 warmup + 5 measurement
		var score: float = _benchmark_cpu()
		if i >= 10:  # Only keep last 5
			cpu_scores.append(score)
	
	# Run memory benchmark (drop first 10 results)
	var memory_scores: Array[float] = []
	for i in range(15):  # 10 warmup + 5 measurement
		var score: float = _benchmark_memory()
		if i >= 10:  # Only keep last 5
			memory_scores.append(score)
	
	# Calculate average scores using StatMath library
	var avg_cpu_score: float = StatMath.BasicStats.mean(cpu_scores)
	var avg_memory_score: float = StatMath.BasicStats.mean(memory_scores)
	
	# Calculate normalization factors (baseline / current)
	cpu_factor = BASELINE_CPU_SCORE / avg_cpu_score
	memory_factor = BASELINE_MEMORY_SCORE / avg_memory_score
	
	print("🔧 Hardware calibration complete:")
	print("   CPU: %.1f score (factor: %.3f)" % [avg_cpu_score, cpu_factor])
	print("   Memory: %.1f MB/s (factor: %.3f)" % [avg_memory_score, memory_factor])

## CPU benchmark - floating point operations
static func _benchmark_cpu() -> float:
	var start_time: int = Time.get_ticks_usec()
	var result: float = 0.0
	
	# Perform 100,000 floating point operations
	for i in range(100000):
		result += sin(i * 0.001) * cos(i * 0.001) * sqrt(i + 1)
	
	var end_time: int = Time.get_ticks_usec()
	var duration_ms: float = (end_time - start_time) / 1000.0
	
	# Return operations per second as score
	return 100000.0 / (duration_ms / 1000.0)

## Memory benchmark - array operations
static func _benchmark_memory() -> float:
	var start_time: int = Time.get_ticks_usec()
	
	# Create and manipulate large arrays
	var data: Array[float] = []
	data.resize(50000)
	
	# Fill array
	for i in range(50000):
		data[i] = randf() * 1000.0
	
	# Sort array (memory intensive)
	data.sort()
	
	var end_time: int = Time.get_ticks_usec()
	var duration_ms: float = (end_time - start_time) / 1000.0
	
	# Return MB/s (approximate)
	var data_size_mb: float = 50000 * 8 / 1024.0 / 1024.0  # 8 bytes per float
	return data_size_mb / (duration_ms / 1000.0)

## Determine normalization factor for a specific test
func _get_test_normalization_factor(test_name: String) -> float:
	# Check if test is CPU-bound
	for cpu_pattern in CPU_BOUND_TESTS:
		if test_name.begins_with(cpu_pattern):
			return cpu_factor
	
	# Check if test is memory-bound  
	for memory_pattern in MEMORY_BOUND_TESTS:
		if test_name.begins_with(memory_pattern):
			return memory_factor
	
	# Default: mixed workload (70% CPU, 30% memory)
	return (cpu_factor * 0.7 + memory_factor * 0.3)

## ========== INDEPENDENT TEST SUITE INTERFACE ==========

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
	
	var data: Dictionary = json.data
	if not data.has("tests"):
		push_error("Baseline file missing 'tests' key: " + BASELINE_FILE)
		return {}
	
	return data["tests"]

## Check performance regression and save result immediately (independent per suite)
func check_performance_regression(module_name: String, test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> void:
	# Set module name if not already set
	if _module_name.is_empty():
		_module_name = module_name
	
	var current_time: float = current_results.execution_time_ms
	var baseline_time: float = NAN
	var is_failure: bool = false
	
	if not baseline_data.is_empty():
		# Look for test with module prefix since all modules are in one baseline file
		var prefixed_test_name: String = "%s_%s" % [module_name.to_lower(), test_name]
		if baseline_data.has(prefixed_test_name):
			baseline_time = baseline_data[prefixed_test_name].result_ms
			var time_change: float = (current_time - baseline_time) / baseline_time
			is_failure = time_change > REGRESSION_THRESHOLD
			
			print("📊 %s: %.2f ms vs baseline %.2f ms (%.1f%% change)" % [
				test_name, current_time, baseline_time, time_change * 100.0
			])
		else:
			print("⚠️  No baseline found for %s - treating as new test" % prefixed_test_name)
			baseline_time = current_time  # Use current as baseline for new tests
	else:
		print("⚠️  No baseline data available - treating as new test: %s" % test_name)
		baseline_time = current_time  # Use current as baseline
	
	# Apply hardware normalization to current time
	var normalization_factor: float = _get_test_normalization_factor(test_name)
	var normalized_time_ms: float = current_time * normalization_factor
	
	# Store result for this module
	var prefixed_test_name: String = "%s_%s" % [module_name.to_lower(), test_name]
	_module_results[prefixed_test_name] = {
		"result_ms": normalized_time_ms,  # Normalized value
		"baseline_ms": baseline_time, 
		"raw_ms": current_time,  # Raw value for debugging
		"diff_percent": ((normalized_time_ms - baseline_time) / baseline_time) * 100.0 if not is_nan(baseline_time) else 0.0,
		"status": "fail" if is_failure else "pass",
		"hardware_factor": normalization_factor
	}
	
	# Save immediately after each test
	_save_module_results()
	
	# Trigger collation check (will collate if multiple modules are ready)
	_trigger_collation_check()

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
	if not DirAccess.dir_exists_absolute(RESULTS_DIR):
		DirAccess.open("res://").make_dir_recursive(RESULTS_DIR)
	
	# Save to module-specific file
	var module_filepath: String = RESULTS_DIR + "%s_%s.json" % [_module_name, _timestamp]
	
	var module_data: Dictionary = {
		"module": _module_name,
		"timestamp": _timestamp,
		"tests": _module_results,
		"meta": {
			"generated_at": Time.get_datetime_string_from_system(),
			"total_tests": _module_results.size(),
			"hardware_normalization": {
				"cpu_factor": cpu_factor,
				"memory_factor": memory_factor
			}
		}
	}
	
	_save_json_file(module_filepath, module_data)
	print("💾 Saved %s results: %s (%d tests)" % [_module_name, module_filepath, _module_results.size()])

## Trigger collation check - will collate if conditions are met
func _trigger_collation_check() -> void:
	# Use a simple timer-based approach to collate results after a delay
	# This allows multiple modules to finish before collation
	var timer: Timer = Timer.new()
	Engine.get_main_loop().current_scene.add_child(timer)
	timer.wait_time = 1.0  # 1 second delay
	timer.one_shot = true
	timer.timeout.connect(func():
		_collate_module_results()
		timer.queue_free()
	)
	timer.start()

## Collate module results from the same timestamp into consolidated files
static func _collate_module_results() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	# Find all module files by timestamp
	var module_files_by_timestamp: Dictionary = {}
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		# Look for pattern: ModuleName_YYYY-MM-DD_HH-MM-SS.json
		# But EXCLUDE pass_, fail_, and other result files
		if (file_name.ends_with(".json") and 
			file_name.count("_") >= 2 and
			not file_name.begins_with("pass_") and
			not file_name.begins_with("fail_") and
			not file_name.begins_with("latest") and
			not file_name.begins_with("baseline")):
			
			var parts: Array = file_name.replace(".json", "").split("_")
			if parts.size() >= 3:  # ModuleName_YYYY-MM-DD_HH-MM-SS
				var module_name: String = parts[0]
				var timestamp: String = "_".join(parts.slice(1))  # Everything after first underscore
				
				if not module_files_by_timestamp.has(timestamp):
					module_files_by_timestamp[timestamp] = []
				
				module_files_by_timestamp[timestamp].append({
					"file": file_name,
					"module": module_name,
					"timestamp": timestamp
				})
		
		file_name = dir.get_next()
	
	# Collate each timestamp group that has multiple modules or is old enough
	for timestamp in module_files_by_timestamp:
		var module_files: Array = module_files_by_timestamp[timestamp]
		
		# Only collate if we have multiple modules - wait for all modules to complete
		var should_collate: bool = module_files.size() > 1
		
		if should_collate:
			_collate_timestamp_group(timestamp, module_files)

## Force collation of ALL pending module files (used in final phase)
static func _collate_module_results_final() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	# Find all module files by timestamp
	var module_files_by_timestamp: Dictionary = {}
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		# Look for pattern: ModuleName_YYYY-MM-DD_HH-MM-SS.json
		# But EXCLUDE pass_, fail_, and other result files
		if (file_name.ends_with(".json") and 
			file_name.count("_") >= 2 and
			not file_name.begins_with("pass_") and
			not file_name.begins_with("fail_") and
			not file_name.begins_with("latest") and
			not file_name.begins_with("baseline")):
			
			var parts: Array = file_name.replace(".json", "").split("_")
			if parts.size() >= 3:  # ModuleName_YYYY-MM-DD_HH-MM-SS
				var module_name: String = parts[0]
				var timestamp: String = "_".join(parts.slice(1))  # Everything after first underscore
				
				if not module_files_by_timestamp.has(timestamp):
					module_files_by_timestamp[timestamp] = []
				
				module_files_by_timestamp[timestamp].append({
					"file": file_name,
					"module": module_name,
					"timestamp": timestamp
				})
		
		file_name = dir.get_next()
	
	# Force collate ALL timestamp groups (even single modules)
	for timestamp in module_files_by_timestamp:
		var module_files: Array = module_files_by_timestamp[timestamp]
		_collate_timestamp_group(timestamp, module_files)

## Collate all module files from the same timestamp into one consolidated file
static func _collate_timestamp_group(timestamp: String, module_files: Array) -> void:
	print("📦 Collating %d module files for timestamp: %s" % [module_files.size(), timestamp])
	
	var consolidated_tests: Dictionary = {}
	var total_tests: int = 0
	var failed_tests: int = 0
	var all_hardware_factors: Dictionary = {}
	
	# Load and merge all module files
	for module_info in module_files:
		var file_path: String = RESULTS_DIR + module_info.file
		var file: FileAccess = FileAccess.open(file_path, FileAccess.READ)
		if file == null:
			push_warning("Could not read module file: " + file_path)
			continue
		
		var json_string: String = file.get_as_text()
		file.close()
		
		var json: JSON = JSON.new()
		var parse_result: Error = json.parse(json_string)
		if parse_result != OK:
			push_warning("Failed to parse JSON in: " + file_path)
			continue
		
		var module_data: Dictionary = json.data
		if not module_data.has("tests"):
			push_warning("Module file missing 'tests' key: " + file_path)
			continue
		
		# Merge tests
		for test_name in module_data.tests:
			consolidated_tests[test_name] = module_data.tests[test_name]
			total_tests += 1
			if module_data.tests[test_name].status == "fail":
				failed_tests += 1
		
		# Track hardware factors
		if module_data.has("meta") and module_data.meta.has("hardware_normalization"):
			var module_name: String = module_info.module
			all_hardware_factors[module_name] = module_data.meta.hardware_normalization
	
	if consolidated_tests.is_empty():
		print("❌ No valid tests found in module files for timestamp: %s" % timestamp)
		return
	
	# Create consolidated result
	var consolidated_data: Dictionary = {
		"tests": consolidated_tests,
		"meta": {
			"type": "test_run",
			"generated_at": Time.get_datetime_string_from_system(),
			"original_timestamp": timestamp,
			"total_tests": total_tests,
			"failed_tests": failed_tests,
			"passed_tests": total_tests - failed_tests,
			"modules_included": module_files.size(),
			"hardware_normalization": all_hardware_factors
		}
	}
	
	# Save as latest.json
	var latest_filepath: String = RESULTS_DIR + "latest.json"
	_save_json_file(latest_filepath, consolidated_data)
	
	# Save as pass/fail snapshot
	var has_failures: bool = failed_tests > 0
	var snapshot_prefix: String = "fail" if has_failures else "pass"
	var snapshot_filepath: String = RESULTS_DIR + "%s_%s.json" % [snapshot_prefix, timestamp]
	_save_json_file(snapshot_filepath, consolidated_data)
	
	print("💾 Consolidated results: %s_%s.json (%d tests, %d failures)" % [
		snapshot_prefix, timestamp, total_tests, failed_tests
	])
	
	# Clean up module files after successful collation
	for module_info in module_files:
		var file_path: String = RESULTS_DIR + module_info.file
		DirAccess.open(RESULTS_DIR).remove(module_info.file)
		print("🗑️  Cleaned up: %s" % module_info.file)
	
	# Trigger baseline update if this was a successful run
	if not has_failures:
		_update_baseline_from_snapshots()
		_cleanup_old_snapshots()

## Helper function to save JSON files
static func _save_json_file(filepath: String, data: Dictionary) -> void:
	var file: FileAccess = FileAccess.open(filepath, FileAccess.WRITE)
	if file == null:
		push_error("Failed to save file: " + filepath)
		return
	
	file.store_string(JSON.stringify(data, "\t"))
	file.close()

## Update baseline from successful snapshots automatically
## NOTE: Only uses pass_ files for baseline calculations - fail_ files are ignored
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
	if recent_files.size() >= 10:
		print("   🎯 Sufficient data for statistical confidence (n≥10)")
	elif recent_files.size() >= 5:
		print("   ⚠️  Limited data - consider running more tests (n=%d)" % recent_files.size())
	else:
		print("   🚨 Very limited data - results may be unstable (n=%d)" % recent_files.size())
	
	# Load and accumulate results using statistical analysis
	var test_data_arrays: Dictionary = {}  # test_name -> Array[float] of all measurements
	
	for results_file in recent_files:
		var file_path: String = RESULTS_DIR + results_file
		var file: FileAccess = FileAccess.open(file_path, FileAccess.READ)
		if file == null:
			push_warning("Could not read results file: " + file_path)
			continue
		
		var json_string: String = file.get_as_text()
		file.close()
		
		var json: JSON = JSON.new()
		var parse_result: Error = json.parse(json_string)
		if parse_result != OK:
			push_warning("Failed to parse JSON in: " + file_path)
			continue
		
		var file_data: Dictionary = json.data
		if not file_data.has("tests"):
			push_warning("Results file missing 'tests' key: " + file_path)
			continue
		
		# Collect each test result into arrays for statistical analysis
		for test_name in file_data.tests:
			var test_data = file_data.tests[test_name]
			
			# Use normalized result_ms value for baseline calculation
			var execution_time: float = test_data.result_ms
			
			if not test_data_arrays.has(test_name):
				test_data_arrays[test_name] = []
			
			test_data_arrays[test_name].append(execution_time)
	
	if test_data_arrays.is_empty():
		print("❌ No valid test data found in successful runs")
		return
	
	# Calculate robust statistics using StatMath library
	var baseline_tests: Dictionary = {}
	var statistics_summary: Dictionary = {}
	
	for test_name in test_data_arrays:
		var measurements: Array[float]
		for measurement: float in test_data_arrays[test_name]:
			measurements.append(measurement)
		
		if measurements.size() == 0:
			continue
		
		# Use median for robustness against outliers
		var test_median: float = StatMath.BasicStats.median(measurements)
		baseline_tests[test_name] = {
			"result_ms": test_median,
			"baseline_ms": test_median, 
			"diff_percent": 0.0,
			"status": "pass"
		}
		
		# Store statistics for reporting
		if measurements.size() > 1:
			var test_mean: float = StatMath.BasicStats.mean(measurements)
			var test_std: float = StatMath.BasicStats.standard_deviation(measurements)
			statistics_summary[test_name] = {
				"mean": test_mean,
				"median": test_median,
				"std_dev": test_std,
				"sample_size": measurements.size(),
				"coefficient_of_variation": (test_std / test_mean) if test_mean > 0.0 else 0.0
			}
	
	# Report statistical summary for well-sampled tests
	print("📈 Statistical Summary:")
	var high_variance_tests: Array[String] = []
	for test_name in statistics_summary:
		var stats: Dictionary = statistics_summary[test_name]
		var cv: float = stats.coefficient_of_variation
		if cv > 0.15:  # Flag tests with >15% coefficient of variation
			high_variance_tests.append(test_name)
		
		if stats.sample_size >= 5:  # Log detailed stats for reasonably sampled tests
			print("   %s: median=%.3fms, cv=%.1f%%, n=%d" % [
				test_name, stats.median, cv * 100.0, stats.sample_size
			])
	
	if not high_variance_tests.is_empty():
		print("⚠️  High variance tests (CV > 15%%): %s" % str(high_variance_tests.slice(0, 5)))
	
	# Save the updated baseline
	var baseline_data: Dictionary = {
		"tests": baseline_tests,
		"meta": {
			"type": "automatic_baseline",
			"generated_at": Time.get_datetime_string_from_system(),
			"total_tests": baseline_tests.size(),
			"source_snapshots": recent_files.size(),
			"statistical_method": "robust_median",
			"confidence": "high" if recent_files.size() >= 10 else ("medium" if recent_files.size() >= 5 else "low")
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
	else:
		# Remove all fail_ files if not keeping failures
		for fail_file in fail_files:
			dir.remove(fail_file)
			print("🗑️  Removed failure snapshot: %s" % fail_file)
