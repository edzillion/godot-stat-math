# addons/godot-stat-math/tests/core/ppf_functions_test.gd
class_name PpfFunctionsTest extends GdUnitTestSuite

# --- Uniform PPF ---
func test_uniform_ppf_basic() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.5, 2.0, 4.0)
	assert_float(result).is_equal_approx(3.0, 1e-7)

func test_uniform_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.0, 2.0, 4.0)
	assert_float(result).is_equal_approx(2.0, 1e-7)

func test_uniform_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(1.0, 2.0, 4.0)
	assert_float(result).is_equal_approx(4.0, 1e-7)

func test_uniform_ppf_invalid_p() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(-0.1, 2.0, 4.0)
	assert_bool(is_nan(result)).is_true()

func test_uniform_ppf_invalid_b_less_than_a() -> void:
	var result: float = StatMath.PpfFunctions.uniform_ppf(0.5, 4.0, 2.0)
	assert_bool(is_nan(result)).is_true()

# --- Normal PPF ---
func test_normal_ppf_standard_normal() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5)
	assert_float(result).is_equal_approx(0.0, 1e-6)

func test_normal_ppf_mu_sigma() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5, 2.0, 3.0)
	assert_float(result).is_equal_approx(2.0, 1e-6)

func test_normal_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_normal_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(1.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_normal_ppf_invalid_sigma() -> void:
	var result: float = StatMath.PpfFunctions.normal_ppf(0.5, 0.0, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Exponential PPF ---
func test_exponential_ppf_basic() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.5, 2.0)
	assert_float(result).is_equal_approx(-log(0.5)/2.0, 1e-7)

func test_exponential_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_exponential_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(1.0, 2.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_exponential_ppf_invalid_lambda() -> void:
	var result: float = StatMath.PpfFunctions.exponential_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Beta PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_beta_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_beta_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(1.0, 1e-7)

func test_beta_ppf_invalid_alpha() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_beta_ppf_invalid_beta() -> void:
	var result: float = StatMath.PpfFunctions.beta_ppf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Gamma PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_gamma_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_gamma_ppf_p_one() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(1.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(INF, 1e-6)

func test_gamma_ppf_invalid_shape() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

func test_gamma_ppf_invalid_scale() -> void:
	var result: float = StatMath.PpfFunctions.gamma_ppf(0.5, 2.0, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Chi-Square PPF ---
func test_chi_square_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.chi_square_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_chi_square_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.chi_square_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- F PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_f_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.f_ppf(0.0, 2.0, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_f_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.f_ppf(0.5, 0.0, 2.0)
	assert_bool(is_nan(result)).is_true()

# --- t PPF (edge/parameter tests only, as CDF is placeholder) ---
func test_t_ppf_p_zero() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.0, 2.0)
	assert_float(result).is_equal_approx(-INF, 1e-6)

func test_t_ppf_p_half() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.5, 2.0)
	assert_float(result).is_equal_approx(0.0, 1e-7)

func test_t_ppf_invalid_df() -> void:
	var result: float = StatMath.PpfFunctions.t_ppf(0.5, 0.0)
	assert_bool(is_nan(result)).is_true()

# --- Binomial PPF ---
func test_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(0.0, 5, 0.5)
	assert_int(result).is_equal(0)

func test_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(1.0, 5, 0.5)
	assert_int(result).is_equal(5)

func test_binomial_ppf_invalid_n() -> void:
	var result: int = StatMath.PpfFunctions.binomial_ppf(0.5, -1, 0.5)
	assert_int(result).is_equal(-1)

# --- Poisson PPF ---
func test_poisson_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.poisson_ppf(0.0, 3.0)
	assert_int(result).is_equal(0)

func test_poisson_ppf_invalid_lambda() -> void:
	var result: int = StatMath.PpfFunctions.poisson_ppf(0.5, -1.0)
	assert_int(result).is_equal(-1)

# --- Geometric PPF ---
func test_geometric_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(0.0, 0.5)
	assert_int(result).is_equal(1)

func test_geometric_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(1.0, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_geometric_ppf_invalid_p() -> void:
	var result: int = StatMath.PpfFunctions.geometric_ppf(-0.1, 0.5)
	assert_int(result).is_equal(-1)

# --- Negative Binomial PPF ---
func test_negative_binomial_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(0.0, 2, 0.5)
	assert_int(result).is_equal(2)

func test_negative_binomial_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(1.0, 2, 0.5)
	assert_int(result).is_equal(StatMath.INT_MAX_REPRESENTING_INF)

func test_negative_binomial_ppf_invalid_r() -> void:
	var result: int = StatMath.PpfFunctions.negative_binomial_ppf(0.5, 0, 0.5)
	assert_int(result).is_equal(-1)

# --- Bernoulli PPF ---
func test_bernoulli_ppf_p_less_than_one_minus_prob_success() -> void:
	# CDF(0) = 1 - prob_success. If p <= CDF(0), result is 0.
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.2. 0.2 <= 0.3, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.2, 0.7)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_p_greater_than_one_minus_prob_success() -> void:
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.4. 0.4 > 0.3, so expect 1.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.4, 0.7)
	assert_int(result).is_equal(1)

func test_bernoulli_ppf_p_equals_one_minus_prob_success() -> void:
	# prob_success = 0.7, 1 - prob_success = 0.3. p = 0.3. 0.3 <= 0.3, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.3, 0.7)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_p_zero() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.0, 0.7)
	assert_int(result).is_equal(0) # Smallest k is 0

func test_bernoulli_ppf_p_one() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(1.0, 0.7)
	assert_int(result).is_equal(1) # Smallest k to make CDF >= 1.0 is 1

func test_bernoulli_ppf_prob_success_zero() -> void:
	# prob_success = 0.0. CDF(0) = 1.0.
	# For p = 0.5, 0.5 <= 1.0, so expect 0.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 0.0)
	assert_int(result).is_equal(0)

func test_bernoulli_ppf_prob_success_one() -> void:
	# prob_success = 1.0. CDF(0) = 0.0. CDF(1) = 1.0.
	# For p = 0.5, 0.5 > 0.0, so expect 1.
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 1.0)
	assert_int(result).is_equal(1)

func test_bernoulli_ppf_invalid_p_too_low() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(-0.1, 0.5)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_p_too_high() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(1.1, 0.5)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_prob_success_too_low() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, -0.1)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters

func test_bernoulli_ppf_invalid_prob_success_too_high() -> void:
	var result: int = StatMath.PpfFunctions.bernoulli_ppf(0.5, 1.1)
	assert_int(result).is_equal(-1) # Expect -1 for invalid parameters


# --- Discrete Histogram PPF ---
func test_discrete_histogram_ppf_basic_cases() -> void:
	var values: Array[String] = ["A", "B", "C"]
	var probabilities: Array[float] = [0.2, 0.5, 0.3] # CDF: A=0.2, B=0.7, C=1.0
	
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.1, values, probabilities)).is_equal("A")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.2, values, probabilities)).is_equal("A") # p == CDF(A)
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.20001, values, probabilities)).is_equal("B")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.69999, values, probabilities)).is_equal("B")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.7, values, probabilities)).is_equal("B") # p == CDF(B)
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.70001, values, probabilities)).is_equal("C")

func test_discrete_histogram_ppf_p_zero() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [0.5, 0.5]
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.0, values, probabilities)).is_equal("A")

func test_discrete_histogram_ppf_p_one_exact_sum() -> void:
	var values: Array[String] = ["A", "B", "C"]
	var probabilities: Array[float] = [0.2, 0.5, 0.3] # Sums to 1.0
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("C")

func test_discrete_histogram_ppf_p_one_sum_less_than_one_fallback() -> void:
	var values: Array[String] = ["X", "Y"]
	var probabilities: Array[float] = [0.1, 0.1] # Sums to 0.2
	# Even if p=1.0, and sum_probs is less, it should return the last value due to the fallback logic.
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("Y")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.15, values, probabilities)).is_equal("Y") # Should also fall into the last bucket here based on logic

func test_discrete_histogram_ppf_single_value() -> void:
	var values: Array[String] = ["OnlyOne"]
	var probabilities: Array[float] = [1.0]
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.0, values, probabilities)).is_equal("OnlyOne")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal("OnlyOne")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(1.0, values, probabilities)).is_equal("OnlyOne")

func test_discrete_histogram_ppf_values_can_be_numbers() -> void:
	var values: Array[int] = [10, 20, 30]
	var probabilities: Array[float] = [0.2, 0.5, 0.3]
	assert_int(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal(20)

func test_discrete_histogram_ppf_invalid_p_too_low() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(-0.1, values, probabilities)).is_null()

func test_discrete_histogram_ppf_invalid_p_too_high() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(1.1, values, probabilities)).is_null()

func test_discrete_histogram_ppf_empty_values() -> void:
	var values: Array = []
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_empty_probabilities() -> void:
	var values: Array[String] = ["A"]
	var probabilities: Array[float] = []
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_size_mismatch() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [1.0]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_negative_probability() -> void:
	var values: Array[String] = ["A", "B"]
	var probabilities: Array[float] = [-0.1, 1.1]
	assert_that(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_null()

func test_discrete_histogram_ppf_probabilities_not_sum_to_one_warning() -> void:
	# This test mainly checks if the function still works correctly based on cumulative logic.
	# The warning itself is harder to check in a standard unit test without log capture.
	var values: Array[String] = ["Low", "High"]
	var probabilities: Array[float] = [0.1, 0.1] # Sums to 0.2, not 1.0
	# CDF: Low=0.1, High=0.2
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.05, values, probabilities)).is_equal("Low")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.1, values, probabilities)).is_equal("Low")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.15, values, probabilities)).is_equal("High")
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.2, values, probabilities)).is_equal("High")
	# For p > sum of probabilities (0.2), it should return the last element
	assert_str(StatMath.PpfFunctions.discrete_histogram_ppf(0.5, values, probabilities)).is_equal("High") 
