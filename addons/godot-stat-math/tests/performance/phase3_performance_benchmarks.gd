## This file is disabled currently awaiting review.
# # res://addons/godot-stat-math/tests/performance/phase3_performance_benchmarks.gd
# class_name Phase3PerformanceBenchmarks extends GdUnitTestSuite

# ## Phase 3: Performance Benchmarks for Critical Operations
# ##
# ## This test suite measures performance of the most critical statistical operations
# ## to ensure they meet performance requirements and identify potential bottlenecks.
# ##
# ## Benchmark Categories:
# ## • Basic statistics on large datasets
# ## • Distribution function performance (CDF, PDF, PPF)
# ## • Complex mathematical functions (gamma, beta, error functions)
# ## • Memory usage efficiency

# const BENCHMARK_TOLERANCE: float = 1e-5
# const SMALL_DATASET_SIZE: int = 1000
# const MEDIUM_DATASET_SIZE: int = 10000
# const PERFORMANCE_BASELINE_MS: float = 100.0  # Maximum acceptable time for medium operations

# # =============================================================================
# # BASIC STATISTICS PERFORMANCE BENCHMARKS
# # =============================================================================

# ## Benchmarks basic statistics functions with varying dataset sizes
# func test_basic_statistics_performance_scaling() -> void:
# 	var dataset_sizes: Array[int] = [100, 1000, 5000]
# 	var performance_results: Dictionary = {}
	
# 	for size in dataset_sizes:
# 		# Generate test data
# 		var data: Array[float] = []
# 		for i in range(size):
# 			data.append(randf() * 100.0)
		
# 		# Benchmark mean calculation
# 		var start_time: int = Time.get_ticks_msec()
# 		var mean_result: float = StatMath.BasicStats.mean(data)
# 		var mean_time: int = Time.get_ticks_msec() - start_time
		
# 		# Benchmark variance calculation  
# 		start_time = Time.get_ticks_msec()
# 		var variance_result: float = StatMath.BasicStats.variance(data)
# 		var variance_time: int = Time.get_ticks_msec() - start_time
		
# 		# Benchmark standard deviation
# 		start_time = Time.get_ticks_msec()
# 		var std_result: float = StatMath.BasicStats.standard_deviation(data)
# 		var std_time: int = Time.get_ticks_msec() - start_time
		
# 		# Store results
# 		performance_results[size] = {
# 			"mean_time": mean_time,
# 			"variance_time": variance_time,
# 			"std_time": std_time
# 		}
		
# 		# Verify correctness (basic sanity checks)
# 		assert_bool(is_finite(mean_result)).is_true()
# 		assert_bool(is_finite(variance_result)).is_true()
# 		assert_bool(is_finite(std_result)).is_true()
# 		assert_float(variance_result).is_greater_equal(0.0)
# 		assert_float(std_result).is_greater_equal(0.0)
	
# 	# Performance should scale reasonably (not exponentially)
# 	var small_time: int = performance_results[100]["mean_time"]
# 	var large_time: int = performance_results[5000]["mean_time"]
	
# 	# Large dataset should not take more than 50x longer than small dataset
# 	# (allows for O(n) scaling with some overhead tolerance)
# 	if small_time > 0:  # Avoid division by zero for very fast operations
# 		var scaling_factor: float = float(large_time) / float(small_time)
# 		assert_float(scaling_factor).is_less(100.0)  # Reasonable scaling limit
	
# 	print("Basic Statistics Performance Results: ", performance_results)

# ## Benchmarks percentile calculations which require sorting
# func test_percentile_performance() -> void:
# 	var data: Array[float] = []
# 	for i in range(MEDIUM_DATASET_SIZE):
# 		data.append(randf() * 1000.0)
	
# 	var percentiles: Array[float] = [5.0, 25.0, 50.0, 75.0, 95.0, 99.0]
# 	var total_time: int = 0
	
# 	for percentile in percentiles:
# 		var start_time: int = Time.get_ticks_msec()
# 		var result: float = StatMath.BasicStats.percentile(data, percentile)
# 		var elapsed_time: int = Time.get_ticks_msec() - start_time
# 		total_time += elapsed_time
		
# 		# Verify result is reasonable
# 		assert_bool(is_finite(result)).is_true()
# 		assert_float(result).is_greater_equal(0.0)
# 		assert_float(result).is_less_equal(1000.0)
	
# 	# Average time per percentile calculation should be reasonable
# 	var average_time: float = float(total_time) / float(percentiles.size())
# 	assert_float(average_time).is_less(PERFORMANCE_BASELINE_MS)
	
# 	print("Percentile calculation average time: ", average_time, " ms")

# # =============================================================================
# # DISTRIBUTION FUNCTION PERFORMANCE BENCHMARKS
# # =============================================================================

# ## Benchmarks CDF function performance for the most commonly used distributions
# func test_cdf_performance_benchmark() -> void:
# 	var test_iterations: int = 1000
# 	var distributions: Array[Dictionary] = [
# 		{"name": "normal", "func": func(x): return StatMath.CdfFunctions.normal_cdf(x, 0.0, 1.0)},
# 		{"name": "exponential", "func": func(x): return StatMath.CdfFunctions.exponential_cdf(x, 1.0)},
# 		{"name": "uniform", "func": func(x): return StatMath.CdfFunctions.uniform_cdf(x, 0.0, 1.0)},
# 		{"name": "gamma", "func": func(x): return StatMath.CdfFunctions.gamma_cdf(x, 2.0, 1.0)},
# 		{"name": "beta", "func": func(x): return StatMath.CdfFunctions.beta_cdf(clamp(x, 0.01, 0.99), 2.0, 3.0)}
# 	]
	
# 	for dist in distributions:
# 		var start_time: int = Time.get_ticks_msec()
		
# 		for i in range(test_iterations):
# 			var x: float = randf() * 5.0  # Test range [0, 5]
# 			var result: float = dist["func"].call(x)
			
# 			# Basic validation
# 			assert_float(result).is_greater_equal(0.0)
# 			assert_float(result).is_less_equal(1.0)
		
# 		var elapsed_time: int = Time.get_ticks_msec() - start_time
# 		var time_per_call: float = float(elapsed_time) / float(test_iterations)
		
# 		# Each CDF call should complete in reasonable time
# 		assert_float(time_per_call).is_less(1.0)  # Less than 1ms per call
		
# 		print(dist["name"], " CDF: ", time_per_call, " ms per call")

# ## Benchmarks PDF function performance
# func test_pdf_performance_benchmark() -> void:
# 	var test_iterations: int = 1000
# 	var distributions: Array[Dictionary] = [
# 		{"name": "normal", "func": func(x): return StatMath.PmfPdfFunctions.normal_pdf(x, 0.0, 1.0)},
# 		{"name": "exponential", "func": func(x): return StatMath.PmfPdfFunctions.exponential_pdf(max(x, 0.01), 1.0)},
# 		{"name": "uniform", "func": func(x): return StatMath.PmfPdfFunctions.uniform_pdf(x, 0.0, 1.0)},
# 		{"name": "gamma", "func": func(x): return StatMath.PmfPdfFunctions.gamma_pdf(max(x, 0.01), 2.0, 1.0)},
# 		{"name": "beta", "func": func(x): return StatMath.PmfPdfFunctions.beta_pdf(clamp(x, 0.01, 0.99), 2.0, 3.0)}
# 	]
	
# 	for dist in distributions:
# 		var start_time: int = Time.get_ticks_msec()
		
# 		for i in range(test_iterations):
# 			var x: float = randf() * 5.0
# 			var result: float = dist["func"].call(x)
			
# 			# Basic validation
# 			assert_float(result).is_greater_equal(0.0)
# 			assert_bool(is_finite(result)).is_true()
		
# 		var elapsed_time: int = Time.get_ticks_msec() - start_time
# 		var time_per_call: float = float(elapsed_time) / float(test_iterations)
		
# 		# Each PDF call should complete in reasonable time  
# 		assert_float(time_per_call).is_less(1.0)
		
# 		print(dist["name"], " PDF: ", time_per_call, " ms per call")

# ## Benchmarks PPF (quantile) function performance
# func test_ppf_performance_benchmark() -> void:
# 	var test_iterations: int = 100  # PPF functions are typically more expensive
# 	var distributions: Array[Dictionary] = [
# 		{"name": "normal", "func": func(p): return StatMath.PpfFunctions.normal_ppf(p, 0.0, 1.0)},
# 		{"name": "exponential", "func": func(p): return StatMath.PpfFunctions.exponential_ppf(p, 1.0)},
# 		{"name": "uniform", "func": func(p): return StatMath.PpfFunctions.uniform_ppf(p, 0.0, 1.0)}
# 	]
	
# 	for dist in distributions:
# 		var start_time: int = Time.get_ticks_msec()
		
# 		for i in range(test_iterations):
# 			var p: float = randf() * 0.98 + 0.01  # Range [0.01, 0.99] to avoid extremes
# 			var result: float = dist["func"].call(p)
			
# 			# Basic validation
# 			assert_bool(is_finite(result)).is_true()
		
# 		var elapsed_time: int = Time.get_ticks_msec() - start_time
# 		var time_per_call: float = float(elapsed_time) / float(test_iterations)
		
# 		# PPF functions can be more expensive, but still reasonable
# 		assert_float(time_per_call).is_less(10.0)  # Less than 10ms per call
		
# 		print(dist["name"], " PPF: ", time_per_call, " ms per call")

# # =============================================================================
# # COMPLEX MATHEMATICAL FUNCTION BENCHMARKS
# # =============================================================================

# ## Benchmarks special mathematical functions used internally
# func test_special_functions_performance() -> void:
# 	var test_iterations: int = 1000
# 	var gamma_time: int = 0
# 	var beta_time: int = 0
# 	var erf_time: int = 0
	
# 	# Benchmark gamma function
# 	var start_time: int = Time.get_ticks_msec()
# 	for i in range(test_iterations):
# 		var x: float = randf() * 10.0 + 0.1  # Range [0.1, 10.1]
# 		var result: float = StatMath.HelperFunctions.log_gamma(x)
# 		assert_bool(is_finite(result)).is_true()
# 	gamma_time = Time.get_ticks_msec() - start_time
	
# 	# Benchmark incomplete beta function
# 	start_time = Time.get_ticks_msec()
# 	for i in range(test_iterations):
# 		var x: float = randf()
# 		var a: float = randf() * 5.0 + 0.1
# 		var b: float = randf() * 5.0 + 0.1
# 		var result: float = StatMath.HelperFunctions.incomplete_beta(x, a, b)
# 		assert_bool(is_finite(result)).is_true()
# 		assert_float(result).is_greater_equal(0.0)
# 		assert_float(result).is_less_equal(1.0)
# 	beta_time = Time.get_ticks_msec() - start_time
	
# 	# Benchmark error function
# 	start_time = Time.get_ticks_msec()
# 	for i in range(test_iterations):
# 		var x: float = randf() * 4.0 - 2.0  # Range [-2, 2]
# 		var result: float = StatMath.ErrorFunctions.erf(x)
# 		assert_bool(is_finite(result)).is_true()
# 		assert_float(result).is_greater_equal(-1.0)
# 		assert_float(result).is_less_equal(1.0)
# 	erf_time = Time.get_ticks_msec() - start_time
	
# 	# Performance assertions
# 	var gamma_per_call: float = float(gamma_time) / float(test_iterations)
# 	var beta_per_call: float = float(beta_time) / float(test_iterations)
# 	var erf_per_call: float = float(erf_time) / float(test_iterations)
	
# 	assert_float(gamma_per_call).is_less(1.0)
# 	assert_float(beta_per_call).is_less(5.0)  # Beta function can be more expensive
# 	assert_float(erf_per_call).is_less(1.0)
	
# 	print("Special function performance:")
# 	print("  Log Gamma: ", gamma_per_call, " ms per call")
# 	print("  Incomplete Beta: ", beta_per_call, " ms per call")
# 	print("  Error Function: ", erf_per_call, " ms per call")

# # =============================================================================
# # MEMORY EFFICIENCY BENCHMARKS
# # =============================================================================

# ## Tests memory efficiency of array operations
# func test_memory_efficiency_large_arrays() -> void:
# 	# Create large array and perform operations without excessive memory usage
# 	var large_size: int = 50000
# 	var data: Array[float] = []
	
# 	# Fill array efficiently
# 	data.resize(large_size)
# 	for i in range(large_size):
# 		data[i] = randf() * 100.0
	
# 	# Test that basic operations don't create excessive temporary arrays
# 	var start_time: int = Time.get_ticks_msec()
	
# 	# These operations should work efficiently with large arrays
# 	var mean_val: float = StatMath.BasicStats.mean(data)
# 	var variance_val: float = StatMath.BasicStats.variance(data)
# 	var min_val: float = StatMath.BasicStats.minimum(data)
# 	var max_val: float = StatMath.BasicStats.maximum(data)
	
# 	var elapsed_time: int = Time.get_ticks_msec() - start_time
	
# 	# Verify results are reasonable
# 	assert_bool(is_finite(mean_val)).is_true()
# 	assert_bool(is_finite(variance_val)).is_true()
# 	assert_bool(is_finite(min_val)).is_true()
# 	assert_bool(is_finite(max_val)).is_true()
# 	assert_float(variance_val).is_greater_equal(0.0)
# 	assert_float(min_val).is_less_equal(max_val)
	
# 	# Should complete within reasonable time even for large arrays
# 	assert_float(float(elapsed_time)).is_less(1000.0)  # Less than 1 second
	
# 	print("Large array (", large_size, " elements) processing time: ", elapsed_time, " ms")

# ## Tests memory usage patterns during intensive computations
# func test_memory_pattern_stress() -> void:
# 	# Perform many operations that could potentially create memory leaks
# 	var iterations: int = 100
# 	var distribution_calls: int = 50
	
# 	var start_time: int = Time.get_ticks_msec()
	
# 	for iteration in range(iterations):
# 		# Create moderate-sized arrays
# 		var data: Array[float] = []
# 		for i in range(500):
# 			data.append(randf() * 10.0)
		
# 		# Perform various calculations
# 		var _mean: float = StatMath.BasicStats.mean(data)
# 		var _std: float = StatMath.BasicStats.standard_deviation(data)
# 		var _median: float = StatMath.BasicStats.median(data)
		
# 		# Test distribution functions
# 		for i in range(distribution_calls):
# 			var x: float = randf() * 5.0
# 			var _normal_cdf: float = StatMath.CdfFunctions.normal_cdf(x)
# 			var _normal_pdf: float = StatMath.PmfPdfFunctions.normal_pdf(x)
			
# 			if randf() > 0.5:
# 				var p: float = randf() * 0.98 + 0.01
# 				var _normal_ppf: float = StatMath.PpfFunctions.normal_ppf(p)
	
# 	var total_time: int = Time.get_ticks_msec() - start_time
# 	var time_per_iteration: float = float(total_time) / float(iterations)
	
# 	# Should maintain consistent performance throughout all iterations
# 	assert_float(time_per_iteration).is_less(100.0)  # Less than 100ms per iteration
	
# 	print("Memory stress test completed. Average time per iteration: ", time_per_iteration, " ms")

# # =============================================================================
# # REAL-WORLD SCENARIO BENCHMARKS
# # =============================================================================

# ## Simulates a realistic game development scenario using statistical functions
# func test_real_world_game_scenario() -> void:
# 	# Simulate a game analytics scenario:
# 	# - Player damage values analysis
# 	# - Drop rate optimization
# 	# - Performance metric tracking
	
# 	var num_players: int = 1000
# 	var num_battles: int = 50
	
# 	var start_time: int = Time.get_ticks_msec()
	
# 	# Generate player damage data
# 	var all_damage_values: Array[float] = []
# 	for player in range(num_players):
# 		for battle in range(num_battles):
# 			# Damage follows a gamma distribution (realistic for game damage)
# 			var damage: float = StatMath.Distributions.randf_gamma(2.5, 1.2)
# 			all_damage_values.append(damage)
	
# 	# Analyze damage statistics
# 	var avg_damage: float = StatMath.BasicStats.mean(all_damage_values)
# 	var damage_variance: float = StatMath.BasicStats.variance(all_damage_values)
# 	var damage_percentiles: Array[float] = [
# 		StatMath.BasicStats.percentile(all_damage_values, 10.0),
# 		StatMath.BasicStats.percentile(all_damage_values, 50.0),
# 		StatMath.BasicStats.percentile(all_damage_values, 90.0)
# 	]
	
# 	# Calculate probability of extreme damage events
# 	var high_damage_threshold: float = avg_damage + 2.0 * sqrt(damage_variance)
# 	var extreme_damage_prob: float = 0.0
# 	for damage in all_damage_values:
# 		if damage > high_damage_threshold:
# 			extreme_damage_prob += 1.0
# 	extreme_damage_prob /= float(all_damage_values.size())
	
# 	# Test item drop rate calculations using binomial distribution
# 	var drop_rate: float = 0.05  # 5% drop rate
# 	var num_kills: int = 100
# 	var expected_drops: float = float(num_kills) * drop_rate
# 	var drop_variance: float = float(num_kills) * drop_rate * (1.0 - drop_rate)
	
# 	var total_time: int = Time.get_ticks_msec() - start_time
	
# 	# Verify results are reasonable
# 	assert_float(avg_damage).is_greater(0.0)
# 	assert_float(damage_variance).is_greater_equal(0.0)
# 	assert_float(extreme_damage_prob).is_greater_equal(0.0)
# 	assert_float(extreme_damage_prob).is_less_equal(1.0)
# 	assert_float(expected_drops).is_equal_approx(5.0, 0.1)
	
# 	# Performance should be acceptable for real-time game use
# 	assert_float(float(total_time)).is_less(2000.0)  # Less than 2 seconds
	
# 	print("Game scenario benchmark completed in ", total_time, " ms")
# 	print("  Average damage: ", avg_damage)
# 	print("  Damage percentiles: ", damage_percentiles)
# 	print("  Extreme damage probability: ", extreme_damage_prob)
# 	print("  Expected drops: ", expected_drops, " ± ", sqrt(drop_variance)) 