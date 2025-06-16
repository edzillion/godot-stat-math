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
const ITERATIONS_PER_TEST: int = 200
const SAMPLING_METHODS: Array = [
	StatMath.SamplingGen.SamplingMethod.RANDOM,
	StatMath.SamplingGen.SamplingMethod.SOBOL,
	StatMath.SamplingGen.SamplingMethod.SOBOL_RANDOM,
	StatMath.SamplingGen.SamplingMethod.HALTON,
	StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE
]

func _ready() -> void:

	if not StatMath:
		push_error("StatMath is not loaded, plugin must be enabled in project settings")
		return

	print("======================================")
	print("SamplingGen Baseline Generator")
	print("======================================")
	print("Time: %s" % Time.get_datetime_string_from_system())
	print("SobolData max dimension: %d" % SobolData.get_max_dimension())
	print("DIRECTION_NUMBERS.size(): %d" % SobolData.DIRECTION_NUMBERS.size())
	print("SobolData.has_dimension(2): ", SobolData.has_dimension(2))
	var dim2_data = SobolData.get_direction_numbers(2)
	print("SobolData.get_direction_numbers(2): ", dim2_data, " [type: ", typeof(dim2_data), ", size: ", dim2_data.size() if dim2_data != null else "null", "]")
	print("SobolData.has_dimension(3): ", SobolData.has_dimension(3))
	var dim3_data = SobolData.get_direction_numbers(3)
	print("SobolData.get_direction_numbers(3): ", dim3_data, " [type: ", typeof(dim3_data), ", size: ", dim3_data.size() if dim3_data != null else "null", "]")
	print("SobolData.has_dimension(11): ", SobolData.has_dimension(11))
	var dim11_data = SobolData.get_direction_numbers(11)
	print("SobolData.get_direction_numbers(11): ", dim11_data, " [type: ", typeof(dim11_data), ", size: ", dim11_data.size() if dim11_data != null else "null", "]")
	print("SobolData.has_dimension(13): ", SobolData.has_dimension(13))
	var dim13_data = SobolData.get_direction_numbers(13)
	print("SobolData.get_direction_numbers(13): ", dim13_data, " [type: ", typeof(dim13_data), ", size: ", dim13_data.size() if dim13_data != null else "null", "]")
	
	# Clear cache and force re-initialization
	StatMath.SamplingGen._sobol_direction_vectors_cache.clear()
	StatMath.SamplingGen._max_cached_dimension = -1
	print("Cleared Sobol cache, forcing re-initialization...")
	
	print("======================================")
	
	_run_performance_tests()
	_manage_archive_files()
	_generate_averaged_baseline()
	
	print("======================================")
	print("✅ Baseline generation complete!")
	print("Now run GDUnit4 performance tests to validate against this baseline.")
	print("======================================")

func _run_performance_tests() -> void:
	print("🚀 Running performance tests...")
	print("")
	
	# Ensure archive directory exists
	if not DirAccess.dir_exists_absolute(ARCHIVE_DIR):
		DirAccess.open("res://").make_dir_recursive(ARCHIVE_DIR.trim_prefix("res://"))
	
	var test_data: Dictionary = {
		"generated_at": Time.get_datetime_string_from_system(),
		"tests": {}
	}
	
	_measure_generate_samples(test_data)
	_measure_coordinated_shuffle(test_data)
	_measure_sample_indices(test_data)
	
	# Save current results to archive with date stamp
	var date_stamp: String = Time.get_date_string_from_system().replace("/", "-")
	var archive_name: String = "baseline_%s.json" % date_stamp
	var archive_path: String = ARCHIVE_DIR + archive_name
	
	var file: FileAccess = FileAccess.open(archive_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(test_data, "  "))
		file.close()
		print("💾 Saved test results to archive: %s" % archive_name)
	else:
		printerr("❌ Failed to save test results to archive")

func _manage_archive_files() -> void:
	print("📁 Managing archive files...")
	
	var dir: DirAccess = DirAccess.open(ARCHIVE_DIR)
	if not dir:
		print("⚠️  Could not access archive directory")
		return
	
	# Get all baseline files in archive
	var archive_files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if file_name.begins_with("baseline_") and file_name.ends_with(".json"):
			archive_files.append(file_name)
		file_name = dir.get_next()
	
	# Sort files by name (which includes timestamp)
	archive_files.sort()
	
	print("📊 Found %d archive files" % archive_files.size())
	
	# Keep only the 3 most recent files
	if archive_files.size() > 3:
		var files_to_delete: int = archive_files.size() - 3
		for i in range(files_to_delete):
			var file_to_delete: String = archive_files[i]
			var full_path: String = ARCHIVE_DIR + file_to_delete
			DirAccess.remove_absolute(full_path)
			print("🗑️  Deleted old archive file: %s" % file_to_delete)

func _generate_averaged_baseline() -> void:
	print("📊 Generating averaged baseline from archive files...")
	
	var dir: DirAccess = DirAccess.open(ARCHIVE_DIR)
	if not dir:
		print("❌ Could not access archive directory")
		return
	
	# Get all remaining baseline files in archive
	var archive_files: Array[String] = []
	dir.list_dir_begin()
	var file_name: String = dir.get_next()
	while file_name != "":
		if file_name.begins_with("baseline_") and file_name.ends_with(".json"):
			archive_files.append(file_name)
		file_name = dir.get_next()
	
	# Sort files by name to get most recent ones
	archive_files.sort()
	
	if archive_files.is_empty():
		print("❌ No archive files found to average")
		return
	
	print("📈 Averaging results from %d archive files" % archive_files.size())
	
	# Load and average the data from archive files
	var averaged_data: Dictionary = {
		"generated_at": Time.get_datetime_string_from_system(),
		"tests": {},
		"source_files": archive_files,
		"average_count": archive_files.size()
	}
	
	var test_sums: Dictionary = {}
	var test_counts: Dictionary = {}
	
	for archive_file in archive_files:
		var full_path: String = ARCHIVE_DIR + archive_file
		var file: FileAccess = FileAccess.open(full_path, FileAccess.READ)
		if not file:
			print("⚠️  Could not read archive file: %s" % archive_file)
			continue
		
		var json_text: String = file.get_as_text()
		file.close()
		
		var json: JSON = JSON.new()
		if json.parse(json_text) != OK:
			print("⚠️  Could not parse JSON in: %s" % archive_file)
			continue
		
		var file_data: Dictionary = json.data
		if not file_data.has("tests"):
			print("⚠️  No tests data in: %s" % archive_file)
			continue
		
		# Add each test result to the sum
		for test_name in file_data.tests:
			var test_value: float = file_data.tests[test_name]
			if not test_sums.has(test_name):
				test_sums[test_name] = 0.0
				test_counts[test_name] = 0
			test_sums[test_name] += test_value
			test_counts[test_name] += 1
	
	# Calculate averages
	for test_name in test_sums:
		if test_counts[test_name] > 0:
			averaged_data.tests[test_name] = test_sums[test_name] / test_counts[test_name]
	
	print("📊 Averaged %d test results" % averaged_data.tests.size())
	
	# Save averaged baseline
	_save_baseline(averaged_data)

func _measure_generate_samples(baseline_data: Dictionary) -> void:
	print("Measuring generate_samples...")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		for dimensions in DIMENSIONS:
			for batch_size in BATCH_SIZES:
				var test_name: String = "generate_samples_%s_%dd_%d" % [method_name, dimensions, batch_size]
				var time: float = _measure_test(func():
					return StatMath.SamplingGen.generate_samples(batch_size, dimensions, method, 0, -1)
				)
				baseline_data.tests[test_name] = time
				print("  %s: %.2f ms" % [test_name, time])

func _measure_coordinated_shuffle(baseline_data: Dictionary) -> void:
	print("Measuring coordinated_shuffle...")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		for deck_size in [52, 256, 512]:
			var test_name: String = "coordinated_shuffle_%s_%d" % [method_name, deck_size]
			var time: float = _measure_test(func():
				return StatMath.SamplingGen.coordinated_shuffle(deck_size, method, 0, -1)
			)
			baseline_data.tests[test_name] = time
			print("  %s: %.2f ms" % [test_name, time])
	
	print("Measuring coordinated_batch_shuffles...")
	
	for method in SAMPLING_METHODS:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		for config in [{"deck": 52, "batch": 10}, {"deck": 256, "batch": 5}]:
			var test_name: String = "batch_shuffles_%s_%d_%d" % [method_name, config.deck, config.batch]
			var time: float = _measure_test(func():
				return StatMath.SamplingGen.coordinated_batch_shuffles(config.deck, config.batch, method, 0, -1)
			)
			baseline_data.tests[test_name] = time
			print("  %s: %.2f ms" % [test_name, time])

func _measure_sample_indices(baseline_data: Dictionary) -> void:
	print("Measuring sample_indices...")
	
	var test_cases: Array = [
		{"strategy": StatMath.SamplingGen.SelectionStrategy.WITH_REPLACEMENT, "pop": 1000, "draw": 100},
		{"strategy": StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, "pop": 1000, "draw": 100},
		{"strategy": StatMath.SamplingGen.SelectionStrategy.RESERVOIR, "pop": 10000, "draw": 1000}
	]

	# Test with RANDOM method (baseline)
	for test_case in test_cases:
		var strategy_name: String = StatMath.SamplingGen.SelectionStrategy.keys()[test_case.strategy]
		var test_name: String = "sample_indices_%s_RANDOM_%d_%d" % [strategy_name, test_case.pop, test_case.draw]
		var time: float = _measure_test(func():
			return StatMath.SamplingGen.sample_indices(test_case.pop, test_case.draw, test_case.strategy, StatMath.SamplingGen.SamplingMethod.RANDOM, -1)
		)
		baseline_data.tests[test_name] = time
		print("  %s: %.2f ms" % [test_name, time])
	
	# Test FISHER_YATES with different sampling methods
	print("Measuring sample_indices with different sampling methods...")
	for method in [StatMath.SamplingGen.SamplingMethod.SOBOL, StatMath.SamplingGen.SamplingMethod.LATIN_HYPERCUBE]:
		var method_name: String = StatMath.SamplingGen.SamplingMethod.keys()[method]
		var test_name: String = "sample_indices_FISHER_YATES_%s_1000_100" % method_name
		var time: float = _measure_test(func():
			return StatMath.SamplingGen.sample_indices(1000, 100, StatMath.SamplingGen.SelectionStrategy.FISHER_YATES, method, -1)
		)
		baseline_data.tests[test_name] = time
		print("  %s: %.2f ms" % [test_name, time])

func _measure_test(test_func: Callable) -> float:
	# Warmup - run the operation a few times to account for JIT compilation
	for i in range(10):
		var result = test_func.call()
		if result is Array:
			result.clear()
	
	# Main measurement - time to complete N iterations
	var start_time: int = Time.get_ticks_usec()
	for i in range(ITERATIONS_PER_TEST):
		var result = test_func.call()
		if result is Array:
			result.clear()
	var end_time: int = Time.get_ticks_usec()
	
	# Return total time for all iterations in milliseconds
	return (end_time - start_time) / 1000.0

func _save_baseline(baseline_data: Dictionary) -> void:
	var file: FileAccess = FileAccess.open(BASELINE_FILE, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(baseline_data, "  "))
		file.close()
		print("")
		print("💾 Averaged baseline saved to: %s" % BASELINE_FILE)
		print("📊 Generated %d averaged test baselines" % baseline_data.tests.size())
		if baseline_data.has("source_files"):
			print("📁 Based on %d archive files: %s" % [baseline_data.source_files.size(), baseline_data.source_files])
	else:
		printerr("❌ Failed to save baseline file") 
