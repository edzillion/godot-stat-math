# res://addons/godot-stat-math/tests/performance/core/pmf_pdf_functions_perf_test.gd
class_name PmfPdfFunctionsPerfTest extends PerfTestBase

## Performance Test Suite for StatMath.PmfPdfFunctions Module
##
## Tests probability mass and density function calculations
## across discrete and continuous distributions.

# Test parameters specific to this module
const TEST_VALUES: Array[float] = [0.5, 1.0, 2.0, 5.0]

# Override module name for result tracking
func get_module_name() -> String:
	return "PmfPdfFunctions"


func test_normal_pdf_performance() -> void:
	var test_name: String = "normal_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.PmfPdfFunctions.normal_pdf(x_val, 0.0, 1.0)
				StatMath.PmfPdfFunctions.normal_pdf(x_val, 5.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_exponential_pdf_performance() -> void:
	var test_name: String = "exponential_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 0:  # Exponential PDF only defined for x >= 0
					StatMath.PmfPdfFunctions.exponential_pdf(x_val, 1.0)
					StatMath.PmfPdfFunctions.exponential_pdf(x_val, 2.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_uniform_pdf_performance() -> void:
	var test_name: String = "uniform_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.PmfPdfFunctions.uniform_pdf(x_val, 0.0, 10.0)
				StatMath.PmfPdfFunctions.uniform_pdf(x_val, -5.0, 5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_gamma_pdf_performance() -> void:
	var test_name: String = "gamma_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val > 0:  # Gamma PDF only defined for x > 0
					StatMath.PmfPdfFunctions.gamma_pdf(x_val, 2.0, 1.0)
					StatMath.PmfPdfFunctions.gamma_pdf(x_val, 5.0, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_beta_pdf_performance() -> void:
	var test_name: String = "beta_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			# Use values in (0,1) for beta distribution
			var test_x_vals: Array[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
			for x_val in test_x_vals:
				StatMath.PmfPdfFunctions.beta_pdf(x_val, 2.0, 3.0)
				StatMath.PmfPdfFunctions.beta_pdf(x_val, 1.0, 1.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_binomial_pmf_performance() -> void:
	var test_name: String = "binomial_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.PmfPdfFunctions.binomial_pmf(5, 20, 0.3)
			StatMath.PmfPdfFunctions.binomial_pmf(10, 30, 0.4)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_poisson_pmf_performance() -> void:
	var test_name: String = "poisson_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.PmfPdfFunctions.poisson_pmf(3, 2.5)
			StatMath.PmfPdfFunctions.poisson_pmf(8, 5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_chi_squared_pdf_performance() -> void:
	var test_name: String = "chi_squared_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val > 0:  # Chi-squared PDF only defined for x > 0
					StatMath.PmfPdfFunctions.chi_squared_pdf(x_val, 2.0)
					StatMath.PmfPdfFunctions.chi_squared_pdf(x_val, 5.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_t_pdf_performance() -> void:
	var test_name: String = "t_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.PmfPdfFunctions.t_pdf(x_val, 3.0)
				StatMath.PmfPdfFunctions.t_pdf(x_val, 10.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_f_pdf_performance() -> void:
	var test_name: String = "f_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val > 0:  # F PDF only defined for x > 0
					StatMath.PmfPdfFunctions.f_pdf(x_val, 2.0, 3.0)
					StatMath.PmfPdfFunctions.f_pdf(x_val, 5.0, 7.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_negative_binomial_pmf_performance() -> void:
	var test_name: String = "negative_binomial_pmf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			StatMath.PmfPdfFunctions.negative_binomial_pmf(5, 2, 0.4)
			StatMath.PmfPdfFunctions.negative_binomial_pmf(10, 3, 0.6)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_weibull_pdf_performance() -> void:
	var test_name: String = "weibull_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val > 0:  # Weibull PDF only defined for x > 0
					StatMath.PmfPdfFunctions.weibull_pdf(x_val, 2.0, 1.5)
					StatMath.PmfPdfFunctions.weibull_pdf(x_val, 1.0, 2.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_pareto_pdf_performance() -> void:
	var test_name: String = "pareto_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val >= 1.0:  # Pareto PDF only defined for x >= scale_param (using 1.0 as scale)
					StatMath.PmfPdfFunctions.pareto_pdf(x_val, 1.0, 2.0)
					StatMath.PmfPdfFunctions.pareto_pdf(x_val, 1.0, 3.0)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_cauchy_pdf_performance() -> void:
	var test_name: String = "cauchy_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				StatMath.PmfPdfFunctions.cauchy_pdf(x_val, 0.0, 1.0)
				StatMath.PmfPdfFunctions.cauchy_pdf(x_val, 2.0, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


func test_lognormal_pdf_performance() -> void:
	var test_name: String = "lognormal_pdf"
	var baseline_data: Dictionary = _load_baseline()
	
	var current_results: Dictionary = _measure_test(test_name, func():
		for i in range(TEST_ITERATIONS):
			for x_val in TEST_VALUES:
				if x_val > 0:  # Lognormal PDF only defined for x > 0
					StatMath.PmfPdfFunctions.lognormal_pdf(x_val, 0.0, 1.0)
					StatMath.PmfPdfFunctions.lognormal_pdf(x_val, 1.0, 0.5)
	)
	
	_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)
