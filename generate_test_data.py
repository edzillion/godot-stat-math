import scipy.stats as stats
import scipy.special as special
import numpy as np
import os
import scipy

def generate_basic_stats_test_data():
    """
    Generates comprehensive test data for basic statistics functions including:
    - Non-normal distributions
    - Numerical stability edge cases  
    - Mixed data types
    - Single element arrays
    """
    
    # Non-normal distribution test cases
    basic_stats_data = {
        # Right-skewed data (common in game analytics - player scores, session times)
        "right_skewed_data": {
            "data": [1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0],
            "sorted_data": [1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0],
            "expected_mean": np.mean([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0]),
            "expected_median": np.median([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0]),
            "expected_variance": np.var([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0], ddof=0),
            "expected_std": np.std([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0], ddof=0),
            "expected_sample_variance": np.var([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0], ddof=1),
            "expected_sample_std": np.std([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0], ddof=1),
            "expected_range": 100.0 - 1.0,
            "expected_min": 1.0,
            "expected_max": 100.0,
            "expected_mad": stats.median_abs_deviation([1.0, 1.2, 1.5, 2.0, 2.1, 3.0, 5.0, 10.0, 25.0, 100.0]),
        },
        
        # Left-skewed data (rare high scores with many lower scores)
        "left_skewed_data": {
            "data": [0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9],
            "sorted_data": [0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9],
            "expected_mean": np.mean([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9]),
            "expected_median": np.median([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9]),
            "expected_variance": np.var([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9], ddof=0),
            "expected_std": np.std([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9], ddof=0),
            "expected_sample_variance": np.var([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9], ddof=1),
            "expected_sample_std": np.std([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9], ddof=1),
            "expected_range": 9.9 - 0.1,
            "expected_min": 0.1,
            "expected_max": 9.9,
            "expected_mad": stats.median_abs_deviation([0.1, 1.0, 5.0, 8.0, 9.0, 9.2, 9.5, 9.7, 9.8, 9.9]),
        },
        
        # Heavy-tailed data (damage spikes, network latency)
        "heavy_tailed_data": {
            "data": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0],
            "sorted_data": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 25.0, 150.0],
            "expected_mean": np.mean([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0]),
            "expected_median": np.median([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 25.0, 150.0]),
            "expected_variance": np.var([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0], ddof=0),
            "expected_std": np.std([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0], ddof=0),
            "expected_sample_variance": np.var([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0], ddof=1),
            "expected_sample_std": np.std([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 25.0, 1.6, 1.7, 150.0], ddof=1),
            "expected_range": 150.0 - 1.0,
            "expected_min": 1.0,
            "expected_max": 150.0,
            "expected_mad": stats.median_abs_deviation([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 25.0, 150.0]),
        },
        
        # Bimodal data (two distinct player skill groups)
        "bimodal_data": {
            "data": [1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0],
            "sorted_data": [1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0],
            "expected_mean": np.mean([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0]),
            "expected_median": np.median([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0]),
            "expected_variance": np.var([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0], ddof=0),
            "expected_std": np.std([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0], ddof=0),
            "expected_sample_variance": np.var([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0], ddof=1),
            "expected_sample_std": np.std([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0], ddof=1),
            "expected_range": 9.0 - 1.0,
            "expected_min": 1.0,
            "expected_max": 9.0,
            "expected_mad": stats.median_abs_deviation([1.0, 1.5, 2.0, 2.5, 3.0, 7.0, 7.5, 8.0, 8.5, 9.0]),
        },
        
        # Power-law data (common in gaming analytics)
        "power_law_data": {
            "data": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            "sorted_data": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            "expected_mean": np.mean([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
            "expected_median": np.median([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]),
            "expected_variance": np.var([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], ddof=0),
            "expected_std": np.std([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], ddof=0),
            "expected_sample_variance": np.var([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], ddof=1),
            "expected_sample_std": np.std([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], ddof=1),
            "expected_range": 128.0 - 1.0,
            "expected_min": 1.0,
            "expected_max": 128.0,
        },
        
        # Numerical stability test cases
        "very_large_numbers": {
            "data": [1e10, 2e10, 3e10, 4e10, 5e10],
            "sorted_data": [1e10, 2e10, 3e10, 4e10, 5e10],
            "expected_mean": np.mean([1e10, 2e10, 3e10, 4e10, 5e10]),
            "expected_median": np.median([1e10, 2e10, 3e10, 4e10, 5e10]),
            "expected_variance": np.var([1e10, 2e10, 3e10, 4e10, 5e10], ddof=0),
            "expected_std": np.std([1e10, 2e10, 3e10, 4e10, 5e10], ddof=0),
            "expected_sample_variance": np.var([1e10, 2e10, 3e10, 4e10, 5e10], ddof=1),
            "expected_sample_std": np.std([1e10, 2e10, 3e10, 4e10, 5e10], ddof=1),
            "expected_range": 5e10 - 1e10,
            "expected_min": 1e10,
            "expected_max": 5e10,
        },
        
        "very_small_numbers": {
            "data": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
            "sorted_data": [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
            "expected_mean": np.mean([1e-10, 2e-10, 3e-10, 4e-10, 5e-10]),
            "expected_median": np.median([1e-10, 2e-10, 3e-10, 4e-10, 5e-10]),
            "expected_variance": np.var([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], ddof=0),
            "expected_std": np.std([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], ddof=0),
            "expected_sample_variance": np.var([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], ddof=1),
            "expected_sample_std": np.std([1e-10, 2e-10, 3e-10, 4e-10, 5e-10], ddof=1),
            "expected_range": 5e-10 - 1e-10,
            "expected_min": 1e-10,
            "expected_max": 5e-10,
        },
        
        "mixed_magnitude_data": {
            "data": [1e-8, 1.0, 1e8],
            "sorted_data": [1e-8, 1.0, 1e8],
            "expected_mean": np.mean([1e-8, 1.0, 1e8]),
            "expected_median": np.median([1e-8, 1.0, 1e8]),
            "expected_variance": np.var([1e-8, 1.0, 1e8], ddof=0),
            "expected_std": np.std([1e-8, 1.0, 1e8], ddof=0),
            "expected_sample_variance": np.var([1e-8, 1.0, 1e8], ddof=1),
            "expected_sample_std": np.std([1e-8, 1.0, 1e8], ddof=1),
            "expected_range": 1e8 - 1e-8,
            "expected_min": 1e-8,
            "expected_max": 1e8,
        },
        
        # High precision edge cases
        "close_numbers": {
            "data": [1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005],
            "sorted_data": [1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005],
            "expected_mean": np.mean([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005]),
            "expected_median": np.median([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005]),
            "expected_variance": np.var([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005], ddof=0),
            "expected_std": np.std([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005], ddof=0),
            "expected_sample_variance": np.var([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005], ddof=1),
            "expected_sample_std": np.std([1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004, 1.0000000005], ddof=1),
            "expected_range": 1.0000000005 - 1.0000000001,
            "expected_min": 1.0000000001,
            "expected_max": 1.0000000005,
        },
        
        # Integer-like data as floats (game scores)
        "integer_like_floats": {
            "data": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
            "sorted_data": [1000.0, 1500.0, 2000.0, 2500.0, 3000.0],
            "expected_mean": np.mean([1000.0, 1500.0, 2000.0, 2500.0, 3000.0]),
            "expected_median": np.median([1000.0, 1500.0, 2000.0, 2500.0, 3000.0]),
            "expected_variance": np.var([1000.0, 1500.0, 2000.0, 2500.0, 3000.0], ddof=0),
            "expected_std": np.std([1000.0, 1500.0, 2000.0, 2500.0, 3000.0], ddof=0),
            "expected_sample_variance": np.var([1000.0, 1500.0, 2000.0, 2500.0, 3000.0], ddof=1),
            "expected_sample_std": np.std([1000.0, 1500.0, 2000.0, 2500.0, 3000.0], ddof=1),
            "expected_range": 3000.0 - 1000.0,
            "expected_min": 1000.0,
            "expected_max": 3000.0,
        },
        
        # Single element edge cases
        "single_zero": {
            "data": [0.0],
            "sorted_data": [0.0],
            "expected_mean": 0.0,
            "expected_median": 0.0,
            "expected_variance": 0.0,
            "expected_std": 0.0,
            "expected_range": 0.0,
            "expected_min": 0.0,
            "expected_max": 0.0,
        },
        
        "single_negative": {
            "data": [-42.0],
            "sorted_data": [-42.0],
            "expected_mean": -42.0,
            "expected_median": -42.0,
            "expected_variance": 0.0,
            "expected_std": 0.0,
            "expected_range": 0.0,
            "expected_min": -42.0,
            "expected_max": -42.0,
        },
        
        "single_large": {
            "data": [1e10],
            "sorted_data": [1e10],
            "expected_mean": 1e10,
            "expected_median": 1e10,
            "expected_variance": 0.0,
            "expected_std": 0.0,
            "expected_range": 0.0,
            "expected_min": 1e10,
            "expected_max": 1e10,
        },
        
        "single_small": {
            "data": [1e-10],
            "sorted_data": [1e-10],
            "expected_mean": 1e-10,
            "expected_median": 1e-10,
            "expected_variance": 0.0,
            "expected_std": 0.0,
            "expected_range": 0.0,
            "expected_min": 1e-10,
            "expected_max": 1e-10,
        },
        
        # Identical values for numerical stability
        "identical_values": {
            "data": [42.42424242, 42.42424242, 42.42424242, 42.42424242, 42.42424242],
            "sorted_data": [42.42424242, 42.42424242, 42.42424242, 42.42424242, 42.42424242],
            "expected_mean": 42.42424242,
            "expected_median": 42.42424242,
            "expected_variance": 0.0,
            "expected_std": 0.0,
            "expected_sample_variance": 0.0,
            "expected_sample_std": 0.0,
            "expected_range": 0.0,
            "expected_min": 42.42424242,
            "expected_max": 42.42424242,
        },
    }
    
    # Generate the test data file
    generate_basic_stats_data_file("basic_stats_test_data", basic_stats_data)

def generate_basic_stats_data_file(filename, data):
    """Generate a GDScript test data file for basic statistics"""
    output_path = os.path.join("addons", "godot-stat-math", "tables", f"{filename}.gd")
    
    content = [
        f"# res://addons/godot-stat-math/tables/{filename}.gd",
        "# THIS FILE IS AUTOGENERATED BY generate_test_data.py",
        "# DO NOT EDIT MANUALLY",
        f"# Generated with: scipy {scipy.__version__}, numpy {np.__version__}",
        "",
        "const VALUES: Dictionary = {",
    ]

    for test_name, test_data in data.items():
        content.append(f'\t"{test_name}": {{')
        
        # Add data array
        data_str = ", ".join(map(str, test_data["data"]))
        content.append(f'\t\t"data": [{data_str}],')
        
        # Add sorted data array
        sorted_data_str = ", ".join(map(str, test_data["sorted_data"]))
        content.append(f'\t\t"sorted_data": [{sorted_data_str}],')
        
        # Add expected values
        for key, value in test_data.items():
            if key not in ["data", "sorted_data"]:
                content.append(f'\t\t"{key}": {value:.12f},')
        
        content.append("\t},")
    
    content.append("}")
    
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(content))
        
    print(f"Successfully generated basic stats test data at: {output_path}")

def generate_test_data():
    """
    Generates GDScript files with pre-calculated values for statistical functions
    to be used in unit tests. All values are validated against scipy.
    """
    
    # PMF/PDF test data
    pmf_pdf_data = {
        "binomial_pmf": [
            # k, n, p, expected
            {"params": [2, 5, 0.5], "expected": stats.binom.pmf(2, 5, 0.5)},
            {"params": [0, 5, 0.5], "expected": stats.binom.pmf(0, 5, 0.5)},
            {"params": [5, 5, 0.5], "expected": stats.binom.pmf(5, 5, 0.5)},
            {"params": [6, 5, 0.5], "expected": stats.binom.pmf(6, 5, 0.5)},  # k > n
            {"params": [0, 5, 0.0], "expected": stats.binom.pmf(0, 5, 0.0)},  # p = 0
            {"params": [5, 5, 1.0], "expected": stats.binom.pmf(5, 5, 1.0)},  # p = 1
        ],
        "poisson_pmf": [
            # k, lambda, expected
            {"params": [2, 3.0], "expected": stats.poisson.pmf(2, 3.0)},
            {"params": [0, 3.0], "expected": stats.poisson.pmf(0, 3.0)},
            {"params": [0, 0.0], "expected": stats.poisson.pmf(0, 0.0)},
            {"params": [-1, 3.0], "expected": 0.0},  # k < 0 should return 0
        ],
        "negative_binomial_pmf": [
            # k, r, p, expected (using Godot convention: total trials needed)
            {"params": [5, 2, 0.5], "expected": stats.nbinom.pmf(5-2, 2, 0.5)},  # 3 failures before 2 successes
            {"params": [2, 2, 0.5], "expected": stats.nbinom.pmf(2-2, 2, 0.5)},  # k = r
            {"params": [1, 2, 0.5], "expected": 0.0},  # k < r should return 0
            {"params": [2, 2, 1.0], "expected": stats.nbinom.pmf(2-2, 2, 1.0)},  # p = 1
        ],
        "normal_pdf": [
            # x, mu, sigma, expected
            {"params": [0.0, 0.0, 1.0], "expected": stats.norm.pdf(0.0, 0.0, 1.0)},
            {"params": [1.0, 0.0, 1.0], "expected": stats.norm.pdf(1.0, 0.0, 1.0)},
            {"params": [2.0, 2.0, 1.0], "expected": stats.norm.pdf(2.0, 2.0, 1.0)},
            {"params": [5.0, 3.0, 2.0], "expected": stats.norm.pdf(5.0, 3.0, 2.0)},
        ],
        "exponential_pdf": [
            # x, lambda, expected (Note: scipy uses scale=1/lambda)
            {"params": [0.0, 1.0], "expected": stats.expon.pdf(0.0, scale=1.0)},
            {"params": [1.0, 1.0], "expected": stats.expon.pdf(1.0, scale=1.0)},
            {"params": [0.0, 2.0], "expected": stats.expon.pdf(0.0, scale=0.5)},
            {"params": [2.0, 0.5], "expected": stats.expon.pdf(2.0, scale=2.0)},
            {"params": [-1.0, 1.0], "expected": 0.0},  # Negative x should return 0
        ],
        "uniform_pdf": [
            # x, a, b, expected
            {"params": [2.5, 1.0, 4.0], "expected": stats.uniform.pdf(2.5, 1.0, 3.0)},  # scipy uses (loc, scale)
            {"params": [1.0, 1.0, 4.0], "expected": stats.uniform.pdf(1.0, 1.0, 3.0)},
            {"params": [4.0, 1.0, 4.0], "expected": stats.uniform.pdf(4.0, 1.0, 3.0)},
            {"params": [0.5, 1.0, 4.0], "expected": 0.0},  # Outside range (below)
            {"params": [4.5, 1.0, 4.0], "expected": 0.0},  # Outside range (above)
            {"params": [0.0, -2.0, 2.0], "expected": stats.uniform.pdf(0.0, -2.0, 4.0)},
        ],
        "gamma_pdf": [
            # x, k_shape, theta_scale, expected
            {"params": [1.0, 2.0, 1.0], "expected": stats.gamma.pdf(1.0, 2.0, scale=1.0)},
            {"params": [0.0, 2.0, 1.0], "expected": stats.gamma.pdf(0.0, 2.0, scale=1.0)},
            {"params": [-1.0, 2.0, 1.0], "expected": 0.0},  # Negative x should return 0
        ],
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
    
    # CDF test data - Organized by distribution for direct test consumption
    # Format: {"params": [params...], "expected": result} to match other test data patterns
    cdf_data = {
        "normal_cdf": [
            # Standard normal tests
            {"params": [1.96, 0.0, 1.0], "expected": stats.norm.cdf(1.96, 0.0, 1.0)},
            {"params": [-1.96, 0.0, 1.0], "expected": stats.norm.cdf(-1.96, 0.0, 1.0)},
            {"params": [0.0, 0.0, 1.0], "expected": stats.norm.cdf(0.0, 0.0, 1.0)},
            {"params": [2.0, 2.0, 1.0], "expected": stats.norm.cdf(2.0, 2.0, 1.0)},
            {"params": [0.0, 2.0, 0.5], "expected": stats.norm.cdf(0.0, 2.0, 0.5)},
        ],
        "exponential_cdf": [
            # StatMath expects rate parameter (lambda), scipy uses scale=1/lambda
            {"params": [2.0, 2.0], "expected": stats.expon.cdf(2.0, scale=1.0/2.0)},
            {"params": [1.0, 1.0], "expected": stats.expon.cdf(1.0, scale=1.0/1.0)},
            {"params": [0.693147, 1.0], "expected": stats.expon.cdf(0.693147, scale=1.0/1.0)},
        ],
        "gamma_cdf": [
            # x, shape (k), scale (theta)
            {"params": [2.0, 2.0, 1.0], "expected": stats.gamma.cdf(2.0, a=2.0, scale=1.0)},
            {"params": [1.0, 1.0, 1.0], "expected": stats.gamma.cdf(1.0, a=1.0, scale=1.0)},
            {"params": [3.841, 1.0, 1.0], "expected": stats.gamma.cdf(3.841, a=1.0, scale=1.0)},
        ],
        "beta_cdf": [
            # x, alpha, beta
            {"params": [0.5, 2.0, 2.0], "expected": stats.beta.cdf(0.5, 2.0, 2.0)},
            {"params": [0.25, 2.0, 3.0], "expected": stats.beta.cdf(0.25, 2.0, 3.0)},
            {"params": [0.75, 3.0, 2.0], "expected": stats.beta.cdf(0.75, 3.0, 2.0)},
        ],
        "chi_square_cdf": [
            # x, degrees of freedom
            {"params": [3.841, 1.0], "expected": stats.chi2.cdf(3.841, 1.0)},
            {"params": [5.991, 2.0], "expected": stats.chi2.cdf(5.991, 2.0)},
            {"params": [7.815, 3.0], "expected": stats.chi2.cdf(7.815, 3.0)},
        ],
        "weibull_cdf": [
            # x, scale (lambda), shape (k)
            {"params": [1.5, 1.0, 2.0], "expected": stats.weibull_min.cdf(1.5, c=2.0, scale=1.0)},
            {"params": [2.0, 2.0, 2.0], "expected": stats.weibull_min.cdf(2.0, c=2.0, scale=2.0)},
            {"params": [0.5, 1.0, 1.0], "expected": stats.weibull_min.cdf(0.5, c=1.0, scale=1.0)},
        ],
        "t_cdf": [
            # x, degrees of freedom
            {"params": [1.0, 10.0], "expected": stats.t.cdf(1.0, 10.0)},
            {"params": [0.0, 5.0], "expected": stats.t.cdf(0.0, 5.0)},
            {"params": [2.0, 3.0], "expected": stats.t.cdf(2.0, 3.0)},
        ],
        "f_cdf": [
            # x, dfn, dfd
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
        "gamma_integer": [
            # x, expected - Gamma(n) = (n-1)! for integer n
            {"params": [1.0], "expected": special.gamma(1.0)},  # 0! = 1
            {"params": [4.0], "expected": special.gamma(4.0)},  # 3! = 6  
            {"params": [5.0], "expected": special.gamma(5.0)},  # 4! = 24
        ],
        "gamma_half_integer": [
            # x, expected - Gamma(n+0.5) involving sqrt(PI)
            {"params": [0.5], "expected": special.gamma(0.5)},   # sqrt(PI)
            {"params": [1.5], "expected": special.gamma(1.5)},   # 0.5 * sqrt(PI)
            {"params": [2.5], "expected": special.gamma(2.5)},   # 1.5 * 0.5 * sqrt(PI)
        ],
    }
    
    # Helper function test data
    helper_functions_data = {
        "binomial_coefficient": [
            # n, r, expected
            {"params": [5, 2], "expected": special.comb(5, 2, exact=True)},  # 10
            {"params": [10, 3], "expected": special.comb(10, 3, exact=True)},  # 120
            {"params": [7, 0], "expected": special.comb(7, 0, exact=True)},  # 1
            {"params": [6, 6], "expected": special.comb(6, 6, exact=True)},  # 1
        ],
        "log_factorial": [
            # n, expected
            {"params": [0], "expected": special.gammaln(1)},  # log(0!) = log(1) = 0
            {"params": [5], "expected": special.gammaln(6)},  # log(5!) = log(Gamma(6))
            {"params": [10], "expected": special.gammaln(11)},  # log(10!) = log(Gamma(11))
        ],
        "log_binomial_coef": [
            # n, k, expected
            {"params": [5, 2], "expected": np.log(special.comb(5, 2, exact=True))},  # log(10)
            {"params": [10, 3], "expected": np.log(special.comb(10, 3, exact=True))},  # log(120)
            {"params": [7, 0], "expected": np.log(special.comb(7, 0, exact=True))},  # log(1) = 0
        ],
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
            # p, lambda_param (rate), expected - converted from scipy scale to rate
            {"params": [0.5, 1.0], "expected": stats.expon.ppf(0.5, scale=1.0/1.0)},
            {"params": [0.632121, 1.0], "expected": stats.expon.ppf(0.632121, scale=1.0/1.0)},
            {"params": [0.95, 2.0], "expected": stats.expon.ppf(0.95, scale=1.0/2.0)},
            {"params": [0.1, 0.5], "expected": stats.expon.ppf(0.1, scale=1.0/0.5)},
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
    
    # Generate basic stats test data
    generate_basic_stats_test_data()
    
    # Generate prime numbers data
    generate_prime_numbers_data()

def generate_prime_numbers_data():
    """
    Generates the first 2000 prime numbers for use in Halton sequences and other
    statistical methods that require prime bases.
    """
    def is_prime(n):
        """Check if a number is prime using trial division."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # Generate first 2000 primes
    primes = []
    candidate = 2
    while len(primes) < 2000:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    print(f"Generated {len(primes)} primes, range: {primes[0]} to {primes[-1]}")
    
    # Write the prime numbers data file
    output_path = os.path.join("addons", "godot-stat-math", "tables", "prime_numbers_data.gd")
    
    content = [
        "# res://addons/godot-stat-math/tables/prime_numbers_data.gd",
        "# THIS FILE IS AUTOGENERATED BY generate_test_data.py",
        "# DO NOT EDIT MANUALLY",
        f"# Generated with: scipy {scipy.__version__}, numpy {np.__version__}",
        "",
        "## Prime numbers table for statistical algorithms and quasi-random generation",
        "##",
        "## Contains the first 2000 prime numbers for use in Halton sequences and other",
        "## statistical methods that require prime bases.",
        "##",
        f"## Generated using: Python trial division algorithm up to {primes[-1]}",
        "## This provides sufficient primes for high-dimensional quasi-random sequences.",
        "",
        "const VALUES: Array[int] = [",
    ]
    
    # Format primes as GDScript array with 20 primes per line
    for i in range(0, len(primes), 20):
        chunk = primes[i:i+20]
        if i + 20 >= len(primes):
            # Last chunk - no trailing comma
            line = "\t" + ", ".join(map(str, chunk))
        else:
            line = "\t" + ", ".join(map(str, chunk)) + ","
        content.append(line)
    
    content.append("]")
    content.append("")
    content.append("## Returns the nth prime number (0-indexed)")
    content.append("static func get_nth_prime(n: int) -> int:")
    content.append("\tif n < 0 or n >= VALUES.size():")
    content.append("\t\tpush_error(\"Prime number index %d out of range [0, %d]\" % [n, VALUES.size() - 1])")
    content.append("\t\treturn 2  # Return first prime as fallback")
    content.append("\treturn VALUES[n]")
    
    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(content))
        
    print(f"Successfully generated prime numbers data at: {output_path}")

def generate_data_file(filename, data):
    """Generate a GDScript test data file with scipy function call documentation"""
    output_path = os.path.join("addons", "godot-stat-math", "tables", f"{filename}.gd")
    
    content = [
        f"# res://addons/godot-stat-math/tables/{filename}.gd",
        "# THIS FILE IS AUTOGENERATED BY generate_test_data.py",
        "# DO NOT EDIT MANUALLY",
        f"# Generated with: scipy {scipy.__version__}, numpy {np.__version__}",
        "",
        "const VALUES: Dictionary = {",
    ]

    # Map function names to their scipy call documentation
    scipy_calls = {
        # Error functions
        "erf": "special.erf(x)",
        "erfc": "special.erfc(x)", 
        "erf_inv": "special.erfinv(y)",
        "erfc_inv": "special.erfcinv(y)",
        "gamma_integer": "special.gamma(x)",
        "gamma_half_integer": "special.gamma(x)",
        
        # Helper functions
        "binomial_coefficient": "special.comb(n, r, exact=True)",
        "log_factorial": "special.gammaln(n+1)",
        "log_binomial_coef": "np.log(special.comb(n, k, exact=True))",
        "lower_incomplete_gamma_regularized": "special.gammainc(a, z)",
        "incomplete_beta": "special.betainc(a, b, x)",
        "beta_function": "special.beta(a, b)",
        
        # CDF functions
        "normal_cdf": "stats.norm.cdf(x, mu, sigma)",
        "exponential_cdf": "stats.expon.cdf(x, scale=1.0/lambda)",
        "gamma_cdf": "stats.gamma.cdf(x, a=shape, scale=scale)",
        "beta_cdf": "stats.beta.cdf(x, alpha, beta)",
        "chi_square_cdf": "stats.chi2.cdf(x, df)",
        "weibull_cdf": "stats.weibull_min.cdf(x, c=shape, scale=scale)",
        "t_cdf": "stats.t.cdf(x, df)",
        "f_cdf": "stats.f.cdf(x, dfn, dfd)",
        
        # PPF functions
        "normal_ppf": "stats.norm.ppf(p, mu, sigma)",
        "exponential_ppf": "stats.expon.ppf(p, scale=1.0/lambda)",
        "uniform_ppf": "stats.uniform.ppf(p, a, b-a)",
        "pareto_ppf": "stats.pareto.ppf(p, b=shape, scale=scale)",
        "weibull_ppf": "stats.weibull_min.ppf(p, c=shape, scale=scale)",
        
        # PMF/PDF functions
        "binomial_pmf": "stats.binom.pmf(k, n, p)",
        "poisson_pmf": "stats.poisson.pmf(k, lambda)",
        "negative_binomial_pmf": "stats.nbinom.pmf(k-r, r, p)",
        "normal_pdf": "stats.norm.pdf(x, mu, sigma)",
        "exponential_pdf": "stats.expon.pdf(x, scale=1/lambda)",
        "uniform_pdf": "stats.uniform.pdf(x, a, b-a)",
        "gamma_pdf": "stats.gamma.pdf(x, shape, scale=scale)",
        "geometric_pmf": "stats.geom.pmf(k, p)",
        "lognormal_pdf": "stats.lognorm.pdf(x, s=sigma, scale=exp(mu))",
        "weibull_pdf": "stats.weibull_min.pdf(x, c=shape, scale=scale)",
        "f_pdf": "stats.f.pdf(x, dfn, dfd)",
        "students_t_pdf": "stats.t.pdf(x, df)",
        "beta_pdf": "stats.beta.pdf(x, alpha, beta)",
        "chi_squared_pdf": "stats.chi2.pdf(x, df)",
    }

    for func_name, test_cases in data.items():
        scipy_call = scipy_calls.get(func_name, "# scipy function call not documented")
        content.append(f'\t"{func_name}": [  # Generated using: {scipy_call}')
        for case in test_cases:
            # Standard format: {"params": [...], "expected": ...}
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