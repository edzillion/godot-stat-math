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