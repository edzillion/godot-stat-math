# addons/godot-stat-math/tests/core/pmf_pdf_functions_test.gd
class_name PmfPdfFunctionsTest extends GdUnitTestSuite

# --- Binomial PMF ---
func test_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(2, 5, 0.5)
	assert_float(result).is_equal_approx(0.3125, 1e-7) # C(5,2) * 0.5^2 * 0.5^3 = 10 * 0.25 * 0.125 = 0.3125

func test_binomial_pmf_k_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(0, 5, 0.5)
	assert_float(result).is_equal_approx(0.03125, 1e-7)

func test_binomial_pmf_k_equals_n() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(5, 5, 0.5)
	assert_float(result).is_equal_approx(0.03125, 1e-7)

func test_binomial_pmf_k_greater_than_n() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(6, 5, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_p_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(0, 5, 0.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.binomial_pmf(1, 5, 0.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_p_one() -> void:
	var result: float = StatMath.PmfPdfFunctions.binomial_pmf(5, 5, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.binomial_pmf(4, 5, 1.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_binomial_pmf_invalid_n_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, -1, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of trials (n_trials) must be non-negative.")

func test_binomial_pmf_invalid_p() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.binomial_pmf(2, 5, -0.1)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be between 0.0 and 1.0.")

# --- Poisson PMF ---
func test_poisson_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(2, 3.0)
	assert_float(result).is_equal_approx(0.2240418, 1e-7) # (3^2 * e^-3) / 2! = 9 * e^-3 / 2

func test_poisson_pmf_k_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(0, 3.0)
	assert_float(result).is_equal_approx(exp(-3.0), 1e-7)

func test_poisson_pmf_lambda_zero() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(0, 0.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.poisson_pmf(1, 0.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_poisson_pmf_k_negative() -> void:
	var result: float = StatMath.PmfPdfFunctions.poisson_pmf(-1, 3.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_poisson_pmf_invalid_lambda_negative() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.poisson_pmf(2, -1.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Rate parameter (lambda_param) must be non-negative.")

# --- Negative Binomial PMF ---
func test_negative_binomial_pmf_basic() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(5, 2, 0.5)
	assert_float(result).is_equal_approx(0.125, 1e-7) # C(4,1) * 0.5^2 * 0.5^3 = 4 * 0.25 * 0.125 = 0.125

func test_negative_binomial_pmf_k_equals_r() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.5)
	assert_float(result).is_equal_approx(0.25, 1e-7)

func test_negative_binomial_pmf_k_less_than_r() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(1, 2, 0.5)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_pmf_p_one() -> void:
	var result: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 1.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)
	var result2: float = StatMath.PmfPdfFunctions.negative_binomial_pmf(3, 2, 1.0)
	assert_float(result2).is_equal_approx(0.0, 1e-7)

func test_negative_binomial_pmf_invalid_r_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 0, 0.5)
	await assert_error(test_call).is_runtime_error("Assertion failed: Number of required successes (r_successes) must be positive.")

func test_negative_binomial_pmf_invalid_p_zero() -> void:
	var test_call: Callable = func():
		StatMath.PmfPdfFunctions.negative_binomial_pmf(2, 2, 0.0)
	await assert_error(test_call).is_runtime_error("Assertion failed: Success probability (p_prob) must be in (0,1].") 