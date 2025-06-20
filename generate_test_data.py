import scipy.stats as stats
import scipy.special as special
import numpy as np
import os

def generate_test_data():
    """
    Generates GDScript files with pre-calculated values for statistical functions
    to be used in unit tests. All values are validated against scipy.
    """
    
    # PMF/PDF test data
    pmf_pdf_data = {
        "geometric_pmf": [
            # k, p, expected
            {"params": [2, 0.5], "expected": stats.geom.pmf(2, 0.5)},
            {"params": [1, 0.3], "expected": stats.geom.pmf(1, 0.3)},
            {"params": [5, 0.1], "expected": stats.geom.pmf(5, 0.1)},
        ],
        "lognormal_pdf": [
            # x, mu, sigma, expected
            {"params": [1.0, 0.0, 1.0], "expected": stats.lognorm.pdf(1.0, s=1.0, scale=np.exp(0.0))},
            {"params": [2.0, 0.0, 0.5], "expected": stats.lognorm.pdf(2.0, s=0.5, scale=np.exp(0.0))},
            {"params": [0.5, 1.0, 0.2], "expected": stats.lognorm.pdf(0.5, s=0.2, scale=np.exp(1.0))},
        ],
        "weibull_pdf": [
            # x, scale_lambda, shape_k, expected (corrected parameter order)
            # Note: scipy's weibull_min is equivalent to Godot's with scale=lambda, shape=k
            {"params": [1.0, 1.0, 1.0], "expected": stats.weibull_min.pdf(1.0, c=1.0, scale=1.0)},
            {"params": [1.5, 1.0, 2.0], "expected": stats.weibull_min.pdf(1.5, c=2.0, scale=1.0)},
            {"params": [2.0, 2.0, 3.0], "expected": stats.weibull_min.pdf(2.0, c=3.0, scale=2.0)},
        ],
        "f_pdf": [
            # x, d1, d2, expected
            {"params": [1.0, 2.0, 3.0], "expected": 0.27885480}, # Value from user
            {"params": [2.5, 5.0, 10.0], "expected": stats.f.pdf(2.5, 5, 10)},
            {"params": [0.8, 10.0, 5.0], "expected": stats.f.pdf(0.8, 10, 5)},
        ],
        "students_t_pdf": [
            # x, df (degrees of freedom), expected
            # Standard t-distribution cases
            {"params": [0.0, 1.0], "expected": stats.t.pdf(0.0, 1.0)},  # Cauchy distribution
            {"params": [1.0, 2.0], "expected": stats.t.pdf(1.0, 2.0)},  # Classic case
            {"params": [2.0, 5.0], "expected": stats.t.pdf(2.0, 5.0)},  # Higher df
            # Special case: as df→∞, approaches standard normal
            {"params": [0.5, 30.0], "expected": stats.t.pdf(0.5, 30.0)},
        ],
        "beta_pdf": [
            # x, alpha, beta, expected
            # Standard cases
            {"params": [0.5, 1.0, 1.0], "expected": stats.beta.pdf(0.5, 1.0, 1.0)},  # Uniform distribution
            {"params": [0.3, 2.0, 5.0], "expected": stats.beta.pdf(0.3, 2.0, 5.0)},  # Skewed
            {"params": [0.7, 5.0, 2.0], "expected": stats.beta.pdf(0.7, 5.0, 2.0)},  # Opposite skew
            # Special cases
            {"params": [0.5, 0.5, 0.5], "expected": stats.beta.pdf(0.5, 0.5, 0.5)},  # U-shaped
            {"params": [0.5, 2.0, 2.0], "expected": stats.beta.pdf(0.5, 2.0, 2.0)},  # Bell-shaped
        ],
        "chi_squared_pdf": [
            # x, df (degrees of freedom), expected
            # Standard cases
            {"params": [1.0, 1.0], "expected": stats.chi2.pdf(1.0, 1)},  # df=1
            {"params": [2.0, 2.0], "expected": stats.chi2.pdf(2.0, 2)},  # df=2 (exponential)
            {"params": [5.0, 4.0], "expected": stats.chi2.pdf(5.0, 4)},  # Higher df
            # Special case: larger df approaches normal
            {"params": [10.0, 10.0], "expected": stats.chi2.pdf(10.0, 10)},
        ],
    }
    
    # CDF test data
    cdf_data = {
        "normal_cdf": [
            # x, mu, sigma, expected
            {"params": [1.96, 0.0, 1.0], "expected": stats.norm.cdf(1.96, 0.0, 1.0)},
            {"params": [-1.96, 0.0, 1.0], "expected": stats.norm.cdf(-1.96, 0.0, 1.0)},
            {"params": [0.0, 2.0, 0.5], "expected": stats.norm.cdf(0.0, 2.0, 0.5)},  # The failing case
            {"params": [0.0, 0.0, 1.0], "expected": stats.norm.cdf(0.0, 0.0, 1.0)},  # Standard normal median
            {"params": [2.0, 2.0, 1.0], "expected": stats.norm.cdf(2.0, 2.0, 1.0)},  # Median case
        ],
        "weibull_cdf": [
            # x, scale_lambda, shape_k, expected (corrected parameter order)
            {"params": [1.5, 1.0, 2.0], "expected": stats.weibull_min.cdf(1.5, c=2.0, scale=1.0)},
            {"params": [2.0, 2.0, 2.0], "expected": stats.weibull_min.cdf(2.0, c=2.0, scale=2.0)},
            {"params": [0.5, 1.0, 1.0], "expected": stats.weibull_min.cdf(0.5, c=1.0, scale=1.0)},
        ],
        "exponential_cdf": [
            # x, scale (1/rate), expected
            {"params": [2.0, 0.5], "expected": stats.expon.cdf(2.0, scale=0.5)},
            {"params": [1.0, 1.0], "expected": stats.expon.cdf(1.0, scale=1.0)},
            {"params": [0.693147, 1.0], "expected": stats.expon.cdf(0.693147, scale=1.0)},
        ],
        "gamma_cdf": [
            # x, shape, scale, expected
            {"params": [2.0, 2.0, 1.0], "expected": stats.gamma.cdf(2.0, a=2.0, scale=1.0)},
            {"params": [1.0, 1.0, 1.0], "expected": stats.gamma.cdf(1.0, a=1.0, scale=1.0)},
            {"params": [3.841, 1.0, 1.0], "expected": stats.gamma.cdf(3.841, a=1.0, scale=1.0)},
        ],
        "beta_cdf": [
            # x, alpha, beta, expected
            {"params": [0.5, 2.0, 2.0], "expected": stats.beta.cdf(0.5, 2.0, 2.0)},
            {"params": [0.25, 2.0, 3.0], "expected": stats.beta.cdf(0.25, 2.0, 3.0)},
            {"params": [0.75, 3.0, 2.0], "expected": stats.beta.cdf(0.75, 3.0, 2.0)},
        ],
        "chi_square_cdf": [
            # x, df, expected
            {"params": [3.841, 1.0], "expected": stats.chi2.cdf(3.841, 1.0)},
            {"params": [5.991, 2.0], "expected": stats.chi2.cdf(5.991, 2.0)},
            {"params": [7.815, 3.0], "expected": stats.chi2.cdf(7.815, 3.0)},
        ],
        "t_cdf": [
            # x, df, expected
            {"params": [1.0, 10.0], "expected": stats.t.cdf(1.0, 10.0)},
            {"params": [0.0, 5.0], "expected": stats.t.cdf(0.0, 5.0)},
            {"params": [2.0, 3.0], "expected": stats.t.cdf(2.0, 3.0)},
        ],
        "f_cdf": [
            # x, dfn, dfd, expected
            {"params": [1.5, 2.0, 2.0], "expected": stats.f.cdf(1.5, 2.0, 2.0)},
            {"params": [2.0, 5.0, 10.0], "expected": stats.f.cdf(2.0, 5.0, 10.0)},
            {"params": [0.8, 3.0, 7.0], "expected": stats.f.cdf(0.8, 3.0, 7.0)},
        ],
    }
    
    # Error function test data
    error_functions_data = {
        "erf": [
            # x, expected
            {"params": [0.5], "expected": special.erf(0.5)},
            {"params": [1.0], "expected": special.erf(1.0)},
            {"params": [2.0], "expected": special.erf(2.0)},
        ],
        "erfc": [
            # x, expected
            {"params": [0.5], "expected": special.erfc(0.5)},
            {"params": [1.0], "expected": special.erfc(1.0)},
            {"params": [2.0], "expected": special.erfc(2.0)},
        ],
        "erf_inv": [
            # y, expected
            {"params": [0.5], "expected": special.erfinv(0.5)},
            {"params": [0.8], "expected": special.erfinv(0.8)},
            {"params": [-0.3], "expected": special.erfinv(-0.3)},
        ],
        "erfc_inv": [
            # y, expected
            {"params": [0.5], "expected": special.erfcinv(0.5)},
            {"params": [1.5], "expected": special.erfcinv(1.5)},
            {"params": [0.2], "expected": special.erfcinv(0.2)},
        ],
    }
    
    # Helper function test data
    helper_functions_data = {
        "lower_incomplete_gamma_regularized": [
            # a, z, expected
            {"params": [2.5, 3.5], "expected": special.gammainc(2.5, 3.5)},
            {"params": [1.0, 2.0], "expected": special.gammainc(1.0, 2.0)},
            {"params": [3.0, 1.5], "expected": special.gammainc(3.0, 1.5)},
        ],
        "incomplete_beta": [
            # x, a, b, expected
            {"params": [0.5, 2.0, 3.0], "expected": special.betainc(2.0, 3.0, 0.5)},
            {"params": [0.3, 1.5, 2.5], "expected": special.betainc(1.5, 2.5, 0.3)},
            {"params": [0.7, 3.0, 2.0], "expected": special.betainc(3.0, 2.0, 0.7)},
        ],
        "beta_function": [
            # a, b, expected
            {"params": [2.0, 3.0], "expected": special.beta(2.0, 3.0)},
            {"params": [1.5, 2.5], "expected": special.beta(1.5, 2.5)},
            {"params": [4.0, 1.0], "expected": special.beta(4.0, 1.0)},
        ],
    }
    
    # PPF test data
    ppf_data = {
        "normal_ppf": [
            # p, mu, sigma, expected
            {"params": [0.025, 0.0, 1.0], "expected": stats.norm.ppf(0.025, 0.0, 1.0)},
            {"params": [0.5, 0.0, 1.0], "expected": stats.norm.ppf(0.5, 0.0, 1.0)},
            {"params": [0.975, 0.0, 1.0], "expected": stats.norm.ppf(0.975, 0.0, 1.0)},
            {"params": [0.5, 10.0, 2.0], "expected": stats.norm.ppf(0.5, 10.0, 2.0)},
            {"params": [0.84134475, 0.0, 1.0], "expected": stats.norm.ppf(0.84134475, 0.0, 1.0)},
        ],
        "exponential_ppf": [
            # p, scale (1/rate), expected 
            {"params": [0.5, 1.0], "expected": stats.expon.ppf(0.5, scale=1.0)},
            {"params": [0.632121, 1.0], "expected": stats.expon.ppf(0.632121, scale=1.0)},
            {"params": [0.95, 0.5], "expected": stats.expon.ppf(0.95, scale=0.5)},
            {"params": [0.1, 2.0], "expected": stats.expon.ppf(0.1, scale=2.0)},
        ],
        "uniform_ppf": [
            # p, a, b, expected
            {"params": [0.25, 0.0, 4.0], "expected": stats.uniform.ppf(0.25, 0.0, 4.0)},
            {"params": [0.5, 1.0, 5.0], "expected": stats.uniform.ppf(0.5, 1.0, 4.0)},  # b-a = 4
            {"params": [0.75, 2.0, 6.0], "expected": stats.uniform.ppf(0.75, 2.0, 4.0)},  # b-a = 4
            {"params": [0.0, 0.0, 1.0], "expected": stats.uniform.ppf(0.0, 0.0, 1.0)},
            {"params": [1.0, 0.0, 1.0], "expected": stats.uniform.ppf(1.0, 0.0, 1.0)},
        ],
        "pareto_ppf": [
            # p, scale, shape, expected (Note: scipy uses b=shape, scale=scale)
            {"params": [0.5, 1.0, 1.0], "expected": stats.pareto.ppf(0.5, b=1.0, scale=1.0)},
            {"params": [0.75, 2.0, 1.0], "expected": stats.pareto.ppf(0.75, b=1.0, scale=2.0)},
            {"params": [0.9, 3.0, 2.0], "expected": stats.pareto.ppf(0.9, b=2.0, scale=3.0)},
        ],
        "weibull_ppf": [
            # p, scale, shape, expected
            {"params": [0.5, 1.0, 1.0], "expected": stats.weibull_min.ppf(0.5, c=1.0, scale=1.0)},
            {"params": [0.632121, 2.0, 2.0], "expected": stats.weibull_min.ppf(0.632121, c=2.0, scale=2.0)},
            {"params": [0.25, 1.0, 2.0], "expected": stats.weibull_min.ppf(0.25, c=2.0, scale=1.0)},
        ],
    }

    # Generate all test data files
    generate_data_file("pmf_pdf_test_data", pmf_pdf_data)
    generate_data_file("cdf_test_data", cdf_data)
    generate_data_file("error_functions_test_data", error_functions_data)
    generate_data_file("helper_functions_test_data", helper_functions_data)
    generate_data_file("ppf_test_data", ppf_data)

def generate_data_file(filename, data):
    """Generate a GDScript test data file"""
    output_path = os.path.join("addons", "godot-stat-math", "tables", f"{filename}.gd")
    
    content = [
        f"# res://addons/godot-stat-math/tables/{filename}.gd",
        "# THIS FILE IS AUTOGENERATED BY generate_test_data.py",
        "# DO NOT EDIT MANUALLY",
        "",
        "const VALUES: Dictionary = {",
    ]

    for func_name, test_cases in data.items():
        content.append(f'\t"{func_name}": [')
        for case in test_cases:
            params_str = ", ".join(map(str, case["params"]))
            expected_val = f'{case["expected"]:.8f}'
            content.append(f'\t\t{{ "params": [{params_str}], "expected": {expected_val} }},')
        content.append("\t],")
    
    content.append("}")
    
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(content))
        
    print(f"Successfully generated test data at: {output_path}")

if __name__ == "__main__":
    generate_test_data() 