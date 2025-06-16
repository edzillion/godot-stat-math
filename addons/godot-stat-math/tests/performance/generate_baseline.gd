# res://addons/godot-stat-math/tests/performance/generate_baseline.gd
extends Node

## Universal Baseline Generator for All StatMath Performance Tests
## 
## Run this script in the editor (Run Current Scene / F6) to:
## 1. Run performance tests for ALL modules and save results to archive with timestamp
## 2. Keep only the 3 most recent archive files (delete older ones)
## 3. Average the results from the 3 most recent files
## 4. Save the averaged results as the single baseline.json

# Paths
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const ARCHIVE_DIR: String = "res://addons/godot-stat-math/tests/performance/archive/"

# Module configuration - all results go into one baseline file
var MODULES: Array[Dictionary] = [
	{
		"name": "SamplingGen",
		"test_class": SamplingGenPerfTest
	},
	{
		"name": "Distributions", 
		"test_class": DistributionsPerfTest
	},
	{
		"name": "HelperFunctions",
		"test_class": HelperFunctionsPerfTest
	},
	{
		"name": "BasicStats",
		"test_class": BasicStatsPerfTest
	},
	{
		"name": "CdfFunctions",
		"test_class": CdfFunctionsPerfTest
	},
	{
		"name": "PmfPdfFunctions",
		"test_class": PmfPdfFunctionsPerfTest
	},
	{
		"name": "PpfFunctions",
		"test_class": PpfFunctionsPerfTest
	},
	{
		"name": "ErrorFunctions",
		"test_class": ErrorFunctionsPerfTest
	}
]

func _ready() -> void:
	print("======================================")
	print("StatMath Universal Baseline Generator")
	print("======================================")
	print("Time: ", Time.get_datetime_string_from_system())
	print("Generating baselines for %d modules..." % MODULES.size())
	print("======================================")
	
	# Quick SobolData verification
	print("SobolData max dimension: ", SobolData.get_max_dimension())
	print("SobolData.has_dimension(2): ", SobolData.has_dimension(2))
	print("SobolData.has_dimension(3): ", SobolData.has_dimension(3))
	
	# Clear Sobol direction vectors cache to ensure fresh measurements
	SamplingGen._sobol_direction_vectors_cache.clear()
	SamplingGen._max_cached_dimension = -1
	print("Cleared Sobol cache, forcing re-initialization...")
	print("======================================")
	
	# Run the baseline generation process
	await _generate_all_baselines()
	
	print("======================================")
	print("✅ All baselines generated successfully!")
	print("======================================")
	get_tree().quit()

func _generate_all_baselines() -> void:
	print("🚀 Running performance tests for all modules...")
	
	# Collect results from all modules into one big result set
	var all_results: Dictionary = {}
	
	for module_info in MODULES:
		var module_name: String = module_info.name
		var test_class = module_info.test_class
		
		print("\n🎯 Running %s performance tests..." % module_name)
		
		# Create an instance of the test class
		var test_instance = test_class.new()
		
		# Check if the test class has the collect_performance_measurements method
		if not test_instance.has_method("collect_performance_measurements"):
			print("  ❌ %s does not have collect_performance_measurements() method" % module_name)
			continue
		
		# Run the test class's own measurement logic
		var module_results: Dictionary = test_instance.collect_performance_measurements()
		
		if module_results.is_empty():
			print("  ❌ No results collected for %s" % module_name)
			continue
		
		# Merge module results into the main results with module prefix
		for test_name in module_results:
			var prefixed_name: String = "%s_%s" % [module_name.to_lower(), test_name]
			all_results[prefixed_name] = module_results[test_name]
		
		print("  ✅ Collected %d tests from %s" % [module_results.size(), module_name])
	
	if all_results.is_empty():
		print("❌ No performance data collected from any module!")
		return
	
	print("\n📊 Total collected: %d performance measurements" % all_results.size())
	
	# Save to archive with timestamp
	var timestamp: String = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	var archive_filename: String = "baseline_%s.json" % timestamp
	var archive_path: String = ARCHIVE_DIR + archive_filename
	
	# Ensure archive directory exists
	if not DirAccess.dir_exists_absolute(ARCHIVE_DIR):
		DirAccess.open("res://").make_dir_recursive(ARCHIVE_DIR)
	
	_save_results_to_file(archive_path, {"tests": all_results})
	print("📁 Saved to archive: %s" % archive_path)
	
	# Clean up old archive files (keep only 3 most recent)
	_cleanup_archive_files()
	
	# Average the 3 most recent files and save as baseline
	var averaged_results: Dictionary = _average_recent_files()
	_save_results_to_file(BASELINE_FILE, averaged_results)
	print("💾 Updated baseline: %s" % BASELINE_FILE)

func _save_results_to_file(file_path: String, results: Dictionary) -> void:
	var file: FileAccess = FileAccess.open(file_path, FileAccess.WRITE)
	if file == null:
		push_error("Failed to open file for writing: " + file_path)
		return
	
	file.store_string(JSON.stringify(results, "\t"))
	file.close()

func _cleanup_archive_files() -> void:
	var dir: DirAccess = DirAccess.open(ARCHIVE_DIR)
	if dir == null:
		return
	
	var files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.begins_with("baseline_") and file_name.ends_with(".json"):
			files.append(file_name)
		file_name = dir.get_next()
	
	# Sort by filename (which includes timestamp)
	files.sort()
	
	# Remove oldest files, keep only 3 most recent
	while files.size() > 3:
		var oldest_file: String = files.pop_front()
		dir.remove(oldest_file)
		print("🗑️  Removed old archive: %s" % oldest_file)

func _average_recent_files() -> Dictionary:
	var dir: DirAccess = DirAccess.open(ARCHIVE_DIR)
	if dir == null:
		push_error("Cannot access archive directory: " + ARCHIVE_DIR)
		return {}
	
	# Get all archive files
	var files: Array[String] = []
	dir.list_dir_begin()
	var current_file: String = dir.get_next()
	
	while current_file != "":
		if current_file.begins_with("baseline_") and current_file.ends_with(".json"):
			files.append(current_file)
		current_file = dir.get_next()
	
	if files.is_empty():
		push_error("No archive files found to average")
		return {}
	
	# Sort and take up to 3 most recent
	files.sort()
	var recent_files: Array[String] = files.slice(-3)  # Last 3 files
	
	print("📊 Averaging %d files: %s" % [recent_files.size(), recent_files])
	
	# Load and accumulate results
	var test_sums: Dictionary = {}
	var test_counts: Dictionary = {}
	
	for archive_file in recent_files:
		var file_path: String = ARCHIVE_DIR + archive_file
		var file: FileAccess = FileAccess.open(file_path, FileAccess.READ)
		if file == null:
			push_warning("Could not read archive file: " + file_path)
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
			push_warning("Archive file missing 'tests' key: " + file_path)
			continue
		
		# Add each test result to the sum
		for test_name in file_data.tests:
			var test_data = file_data.tests[test_name]
			
			# Handle backward compatibility: old format (float) vs new format (Dictionary)
			var execution_time: float
			
			if test_data is float:
				# Old format: just execution time
				execution_time = test_data
			else:
				# This shouldn't happen since we're going back to time-only, but handle it
				execution_time = test_data
			
			if not test_sums.has(test_name):
				test_sums[test_name] = 0.0
				test_counts[test_name] = 0
			
			test_sums[test_name] += execution_time
			test_counts[test_name] += 1
	
	# Calculate averages
	var averaged_tests: Dictionary = {}
	for test_name in test_sums:
		if test_counts[test_name] > 0:
			averaged_tests[test_name] = test_sums[test_name] / test_counts[test_name]
	
	return {"tests": averaged_tests} 
