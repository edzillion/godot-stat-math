StatMath.PerfTestManager
========================

Centralized Performance Testing Infrastructure
Each test suite saves its own results independently to an intermediate file.

A final phase consolidates these files into a single `latest.json` and a timestamped snapshot.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PerfTestManager.function_name(parameters)

Constants
---------

.. data:: BASELINE_FILE

   Value: ``String = "res://addons/godot-stat-math/tests/performance/results/baseline.json"``

.. data:: RESULTS_DIR

   Value: ``String = "res://addons/godot-stat-math/tests/performance/results/"``

.. data:: CORE_TEST_DIR

   Value: ``String = "res://addons/godot-stat-math/tests/performance/core/"``

.. data:: REGRESSION_THRESHOLD

   Value: ``float = 0.20  # 20% slower = regression (fallback for tests without statistical data)``

.. data:: IMPROVEMENT_WARNING_THRESHOLD

   Value: ``float = 0.30  # 30% improvement triggers "consider updating baseline" warning``

.. data:: WARMUP_ITERATIONS

   Value: ``int = 10``

.. data:: MEASUREMENT_ITERATIONS

   Value: ``int = 5``

.. data:: FUNCTION_CALLS_PER_MEASUREMENT

   Value: ``int = 100``

.. data:: KEEP_PREVIOUS_FAILURES

   Value: ``bool = false``

.. data:: MAX_SNAPSHOTS

   Value: ``int = 50  # Keep 50 most recent snapshots for robust statistics``

.. data:: MIN_SAMPLES_FOR_DYNAMIC_THRESHOLD

   Value: ``int = 5  # Minimum samples needed for dynamic thresholds``

.. data:: MIN_THRESHOLD_PERCENT

   Value: ``float = 0.05  # Base minimum threshold (5%)``

.. data:: ROBUST_MIN_THRESHOLD_PERCENT

   Value: ``float = 0.08  # Robust minimum threshold (8%) for more stable testing``

.. data:: MAX_THRESHOLD_PERCENT

   Value: ``float = 0.25  # Maximum 25% threshold``

.. data:: HIGH_CONFIDENCE_SAMPLES

   Value: ``int = 30  # 30+ samples = high confidence``

.. data:: MEDIUM_CONFIDENCE_SAMPLES

   Value: ``int = 15  # 15+ samples = medium confidence``

.. data:: THRESHOLD_SAFETY_BUFFER

   Value: ``float = 1.1  # 10% buffer for borderline cases``

.. data:: STABLE_FUNCTION_CV_THRESHOLD

   Value: ``float = 0.05  # CV threshold for considering function "very stable"``

.. data:: STABLE_FUNCTION_MIN_THRESHOLD

   Value: ``float = 0.12  # 12% minimum for very stable functions``

.. data:: FAST_FUNCTION_THRESHOLD_MS

   Value: ``float = 0.5  # Functions under 0.5ms get special handling``

.. data:: FAST_FUNCTION_MIN_THRESHOLD

   Value: ``float = 0.15  # 15% minimum for very fast functions``

.. data:: LOW_VOLATILITY_CV_THRESHOLD

   Value: ``float = 0.08  # Functions with CV < 8% are low volatility``

.. data:: MEDIUM_VOLATILITY_CV_THRESHOLD

   Value: ``float = 0.15  # Functions with CV < 15% are medium volatility``

.. data:: LOW_VOLATILITY_MIN_THRESHOLD

   Value: ``float = 0.15  # 15% minimum for low volatility functions``

.. data:: MEDIUM_VOLATILITY_MIN_THRESHOLD

   Value: ``float = 0.20  # 20% minimum for medium volatility functions``

.. data:: HIGH_VOLATILITY_MIN_THRESHOLD

   Value: ``float = 0.25  # 25% minimum for high volatility functions``

.. data:: MATURE_BASELINE_SAMPLE_SIZE

   Value: ``int = 25  # Consider baseline "mature" at 25+ samples``

.. data:: MATURE_BASELINE_MULTIPLIER

   Value: ``float = 1.3  # 30% higher thresholds for mature baselines``

.. data:: PERCENTILE_THRESHOLD

   Value: ``float = 95.0  # Use 95th percentile (only 5% of runs slower)``

Functions
---------

.. function:: analyze_measurements(measurements: Array[float], baseline_median: float = NAN, print_details: bool = true) -> Dictionary:

   Analyze measurements and return detailed statistics with dynamic threshold calculation
   This can be called independently without running performance tests

.. function:: analyze_baseline_test(test_name: String, print_details: bool = true) -> Dictionary:

   Analyze a specific test from the baseline file

.. function:: analyze_all_baseline_tests(print_summary: bool = true) -> Dictionary:

   Analyze all tests in the baseline file and return summary statistics

.. function:: _get_volatility_level(cv: float) -> String:

   Helper function to get volatility level from coefficient of variation

.. function:: _get_confidence_level(sample_size: int) -> String:

   Helper function to get confidence level from sample size

.. function:: _initialize_completion_tracker() -> void:

   Initialize completion tracking system

.. function:: _discover_test_modules() -> Array[String]:

   Discover all test suite modules from the core directory

.. function:: _filename_to_module_name(filename: String) -> String:

   Convert filename to expected module name format

.. function:: _check_all_modules_completed() -> void:

   Check if all modules have completed by scanning intermediate files

.. function:: _consolidate_run_results_immediate() -> void:

   Immediate consolidation without async - works in GDUnit context

.. function:: _cleanup_intermediate_files_immediate(intermediate_files: Array[String], dir: DirAccess) -> void:

   Immediate cleanup of intermediate files - synchronous operation

.. function:: _cleanup_orphaned_intermediate_files() -> void:

   Clean up orphaned intermediate files from incomplete test runs

.. function:: _module_name_to_snake_case(module_name: String) -> String:

   Convert PascalCase module name to snake_case for test naming

.. function:: register_module_completion(module_name: String) -> void:

   Register module completion and check for final phase trigger

.. function:: reset_completion_tracker() -> void:

   Reset completion tracker (for testing purposes)

.. function:: _init() -> void:

   Initialize hardware normalization for this instance

.. function:: set_module_name(module_name: String) -> void:

   Set the module name for this test suite instance

.. function:: measure_test(test_name: String, test_func: Callable) -> Dictionary:

   Measure a test function performance with warmup and multiple iterations

.. function:: load_baseline() -> Dictionary:

   Load baseline data from file

.. function:: check_performance_regression(module_name: String, test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> bool:

   Check performance regression and save result immediately (independent per suite)

.. function:: generate_test_data(size: int, seed: int = 12345) -> Array[float]:

   Generate reproducible test data for performance tests

.. function:: _save_module_results() -> void:

   Save this module's results immediately

.. function:: _report_failed_tests(consolidated_tests: Dictionary) -> void:

   Report details of failed tests with their stats and thresholds

.. function:: _calculate_dynamic_threshold(measurements: Array[float], baseline_median: float) -> float:

   Calculate dynamic threshold for a test based on percentile analysis with refinements

.. function:: _load_json_file(filepath: String) -> Dictionary:

   Helper to load a JSON file and return its data

.. function:: _save_json_file(filepath: String, data: Dictionary) -> void:

   Helper function to save JSON files

.. function:: _update_baseline_from_snapshots() -> void:

   Update baseline from successful snapshots automatically

.. function:: _cleanup_old_snapshots() -> void:

   Clean up old snapshot files to maintain MAX_SNAPSHOTS limit

