# res://addons/godot-stat-math/tests/performance/generate_baseline.gd
extends Node

## Scene based script to generate performance baselines for SamplingGen
## 
## Run this script in the editor (Run Current Scene / F6) to:
## 1. Run performance tests and save results to archive with timestamp
## 2. Keep only the 3 most recent archive files (delete older ones)
## 3. Average the results from the 3 most recent files
## 4. Save the averaged results as the new baseline.json

# Paths
const BASELINE_FILE: String = "res://addons/godot-stat-math/tests/performance/baseline.json"
const ARCHIVE_DIR: String = "res://addons/godot-stat-math/tests/performance/archive/"

# Test configuration - must match the performance test suite
const BATCH_SIZES: Array[int] = [256, 1024, 4096]
const DIMENSIONS: Array[int] = [1, 2, 3]
const GENERATORS: Array[SamplingGen.Generator] = [
	SamplingGen.Generator.RANDOM,
	SamplingGen.Generator.SOBOL,
	SamplingGen.Generator.SOBOL_RANDOM,
	SamplingGen.Generator.HALTON
]

# Performance measurement
const WARMUP_ITERATIONS: int = 3
const MEASUREMENT_ITERATIONS: int = 5

func _ready() -> void:
	print("======================================")
	print("SamplingGen Baseline Generator")
	print("======================================")
	print("Time: ", Time.get_datetime_string_from_system())
	
	# Quick SobolData verification
	print("SobolData max dimension: ", SamplingGen.SobolData.get_max_dimension())
	print("DIRECTION_NUMBERS.size(): ", SamplingGen.SobolData.DIRECTION_NUMBERS.size())
	print("SobolData.has_dimension(2): ", SamplingGen.SobolData.has_dimension(2))
	print("SobolData.get_direction_numbers(2): ", SamplingGen.SobolData.get_direction_numbers(2))
	print("SobolData.has_dimension(3): ", SamplingGen.SobolData.has_dimension(3))
	print("SobolData.get_direction_numbers(3): ", SamplingGen.SobolData.get_direction_numbers(3))
	print("SobolData.has_dimension(11): ", SamplingGen.SobolData.has_dimension(11))
	print("SobolData.get_direction_numbers(11): ", SamplingGen.SobolData.get_direction_numbers(11))
	print("SobolData.has_dimension(13): ", SamplingGen.SobolData.has_dimension(13))
	print("SobolData.get_direction_numbers(13): ", SamplingGen.SobolData.get_direction_numbers(13))
	
	# Clear cache to ensure fresh measurements
	SamplingGen._sobol_cache.clear()
	print("Cleared Sobol cache, forcing re-initialization...")
	print("======================================")
	
	# Run the baseline generation process
	await _generate_baseline()
	
	print("✅ Baseline generation complete!")
	get_tree().quit()

func _generate_baseline() -> void:
	print("🚀 Running performance tests...")
	
	# Run performance tests
	var current_results: Dictionary = await _run_performance_tests()
	
	# Save to archive with timestamp
	var timestamp: String = Time.get_datetime_string_from_system().replace(":", "-").replace("T", "_")
	var archive_filename: String = "baseline_%s.json" % timestamp
	var archive_path: String = ARCHIVE_DIR + archive_filename
	
	# Ensure archive directory exists
	if not DirAccess.dir_exists_absolute(ARCHIVE_DIR):
		DirAccess.open("res://").make_dir_recursive(ARCHIVE_DIR)
	
	_save_results_to_file(archive_path, current_results)
	print("📁 Saved to archive: %s" % archive_path)
	
	# Clean up old archive files (keep only 3 most recent)
	_cleanup_archive_files()
	
	# Average the 3 most recent files and save as baseline
	var averaged_results: Dictionary = _average_recent_files()
	_save_results_to_file(BASELINE_FILE, averaged_results)
	print("💾 Updated baseline: %s" % BASELINE_FILE)

func _run_performance_tests() -> Dictionary:
	var results: Dictionary = {}
	
	print("\nMeasuring generate_samples...")
	for generator in GENERATORS:
		for dimension in DIMENSIONS:
			for batch_size in BATCH_SIZES:
				var test_name: String = "generate_samples_%s_%dd_%d" % [
					SamplingGen.Generator.keys()[generator], dimension, batch_size
				]
				
				var execution_time: float = _measure_test(test_name, func(): 
					return SamplingGen.generate_samples(generator, batch_size, dimension)
				)
				
				results[test_name] = execution_time
				print("  %s: %.2f ms" % [test_name, execution_time])
	
	return {"tests": results}

func _measure_test(test_name: String, test_func: Callable) -> float:
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
	return times[median_index]

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
	var file_name: String = dir.get_next()
	
	while file_name != "":
		if file_name.begins_with("baseline_") and file_name.ends_with(".json"):
			files.append(file_name)
		file_name = dir.get_next()
	
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
	
	for file_name in recent_files:
		var file_path: String = ARCHIVE_DIR + file_name
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
