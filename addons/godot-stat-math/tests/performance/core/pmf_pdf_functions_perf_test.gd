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


#func test_normal_pdf_performance() -> void:
	#var test_name: String = "normal_pdf"
	#var baseline_data: Dictionary = _load_baseline()
	#
	#var current_results: Dictionary = _measure_test(test_name, func():
		#for i in range(TEST_ITERATIONS):
			#for x_val in TEST_VALUES:
				#StatMath.PmfPdfFunctions.normal_pdf(x_val, 0.0, 1.0)
				#StatMath.PmfPdfFunctions.normal_pdf(x_val, 5.0, 2.0)
	#)
	#
	#_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


#func test_exponential_pdf_performance() -> void:
	#var test_name: String = "exponential_pdf"
	#var baseline_data: Dictionary = _load_baseline()
	#
	#var current_results: Dictionary = _measure_test(test_name, func():
		#for i in range(TEST_ITERATIONS):
			#for x_val in TEST_VALUES:
				#if x_val >= 0:  # Exponential PDF only defined for x >= 0
					#StatMath.PmfPdfFunctions.exponential_pdf(x_val, 1.0)
					#StatMath.PmfPdfFunctions.exponential_pdf(x_val, 2.5)
	#)
	#
	#_check_performance_regression(get_module_name(), test_name, current_results, baseline_data)


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




 
