# res://addons/godot-stat-math/tests/performance/test_run_collector.gd
class_name TestRunCollector extends RefCounted

## Test Run Result Collector
##
## Automatically collects all performance test results during a test run.
## If any test fails, saves all results to results directory with CI-friendly naming.

# Configuration
const RESULTS_DIR: String = "res://addons/godot-stat-math/tests/performance/results/"
const KEEP_PREVIOUS_FAILURES: bool = false  # true = keep all failure files, false = only keep latest

# Singleton for signal handling
static var _instance: TestRunCollector = null

# Static data - shared across all test instances in a run
static var _current_run_results: Dictionary = {}
static var _failed_tests: Array[String] = []
static var _completed_modules: Array[String] = []
static var _expected_modules: Array[String] = ["BasicStats", "SamplingGen", "Distributions", "HelperFunctions", "CdfFunctions", "PmfPdfFunctions", "PpfFunctions", "ErrorFunctions"]
static var _has_active_run: bool = false
static var _is_saving: bool = false

# Signal for completion notification
signal save_completed

## Get singleton instance
static func get_instance() -> TestRunCollector:
	if _instance == null:
		_instance = TestRunCollector.new()
	return _instance

## Start a new test run collection
static func start_test_run() -> void:
	if _has_active_run:
		return  # Already started, don't restart
	
	_current_run_results.clear()
	_failed_tests.clear()
	_completed_modules.clear()
	_has_active_run = true
	_is_saving = false
	
	print("📋 Expected modules: %s" % str(_expected_modules))

## Add a test result to the current run
static func add_test_result(module_name: String, test_name: String, current_time_ms: float, baseline_time_ms: float, is_failure: bool) -> void:
	if not _has_active_run:
		start_test_run()
	
	var prefixed_test_name: String = "%s_%s" % [module_name.to_lower(), test_name]
	
	_current_run_results[prefixed_test_name] = {
		"current_time_ms": current_time_ms,
		"baseline_time_ms": baseline_time_ms,
		"regression_percent": ((current_time_ms - baseline_time_ms) / baseline_time_ms) * 100.0,
		"failed": is_failure
	}
	
	# Track module completion
	if not _completed_modules.has(module_name):
		_completed_modules.append(module_name)
		print("✅ Module completed: %s" % module_name)
	
	if is_failure:
		print("🚨 %s failed: %.2f ms vs baseline %.2f ms (%.1f%% change)" % [
			prefixed_test_name, current_time_ms, baseline_time_ms, ((current_time_ms - baseline_time_ms) / baseline_time_ms) * 100.0
		])
		_failed_tests.append(prefixed_test_name)

## Finish the test run - save results if there were failures
## Returns an awaitable that completes when save operation finishes
static func finish_test_run() -> Variant:
	if not _has_active_run:
		return null
	
	# Check if all expected modules have completed
	if not _all_modules_completed():
		print("🔄 Test run incomplete: %d/%d modules completed (%s)" % [
			_completed_modules.size(), 
			_expected_modules.size(),
			str(_completed_modules)
		])
		return null
	
	# If already saving, wait for completion
	if _is_saving:
		await get_instance().save_completed
		return get_instance().save_completed
	
	# Start the save process
	_is_saving = true
	print("✅ All performance tests completed (%d tests)" % _current_run_results.size())
	
	if not _failed_tests.is_empty():
		_save_test_run_results()
		print("🔒 File save completed, test run finished")
	else:
		print("✅ All tests passed - no results file needed")
	
	_current_run_results.clear()
	_failed_tests.clear()
	_completed_modules.clear()
	_has_active_run = false
	_is_saving = false
	
	# Emit completion signal after save completes
	var instance: TestRunCollector = get_instance()
	instance.save_completed.emit()
	
	# Return awaitable that resolves immediately since save is done
	return instance.save_completed

## Save results from a test run (always save as latest.json for CI)
static func _save_test_run_results() -> void:
	# Ensure results directory exists
	if not DirAccess.dir_exists_absolute(RESULTS_DIR):
		DirAccess.open("res://").make_dir_recursive(RESULTS_DIR)
	
	# Always save as latest.json for CI
	var latest_filepath: String = RESULTS_DIR + "latest.json"
	
	# Create enhanced structure with consistent data for UI consumption
	var enhanced_tests: Dictionary = {}
	
	for test_name in _current_run_results:
		var test_data: Dictionary = _current_run_results[test_name]
		
		enhanced_tests[test_name] = {
			"result_ms": test_data.current_time_ms,
			"baseline_ms": test_data.baseline_time_ms, 
			"diff_percent": test_data.regression_percent,
			"status": "fail" if test_data.failed else "pass"
		}
	
	var results_data: Dictionary = {
		"tests": enhanced_tests,
		"meta": {
			"generated_at": Time.get_datetime_string_from_system(),
			"total_tests": _current_run_results.size(),
			"failed_tests": _failed_tests.size(),
			"passed_tests": _current_run_results.size() - _failed_tests.size(),
			"type": "test_run"
		}
	}
	
	# Save as latest.json
	_save_json_file(latest_filepath, results_data)
	print("💾 Test run results saved as latest.json (%d tests, %d failures)" % [_current_run_results.size(), _failed_tests.size()])
	
	# If keeping previous failures and there were failures, also save timestamped version
	if KEEP_PREVIOUS_FAILURES and not _failed_tests.is_empty():
		var timestamp: String = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
		var timestamped_filepath: String = RESULTS_DIR + "failure_%s.json" % timestamp
		_save_json_file(timestamped_filepath, results_data)
		print("💾 Also saved timestamped failure: failure_%s.json" % timestamp)
	
	if not _failed_tests.is_empty():
		print("   Failed tests: %s" % str(_failed_tests))

## Helper function to save JSON files
static func _save_json_file(filepath: String, data: Dictionary) -> void:
	var file: FileAccess = FileAccess.open(filepath, FileAccess.WRITE)
	if file == null:
		push_error("Failed to save test run results: " + filepath)
		return
	
	file.store_string(JSON.stringify(data, "\t"))
	file.close()

## Check if all expected modules have completed
static func _all_modules_completed() -> bool:
	# Check if all expected modules have been recorded
	for expected_module in _expected_modules:
		if not _completed_modules.has(expected_module):
			return false
	
	return true

## Delete previous failure files (when keeping only latest)
static func _delete_previous_failure_files() -> void:
	var dir: DirAccess = DirAccess.open(RESULTS_DIR)
	if dir == null:
		return
	
	var files_to_delete: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.begins_with("failure_") and file_name.ends_with(".json"):
			files_to_delete.append(file_name)
		file_name = dir.get_next()
	
	for file in files_to_delete:
		dir.remove(file)
		print("🗑️  Removed previous failure file: %s" % file)
