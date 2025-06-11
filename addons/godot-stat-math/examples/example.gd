extends Node

# Example of how to use the GDStatsPercentile extension

func _ready():
	# Now you can use both the original functionality and the percentile functions
	
	# Example 1: Normal Distribution
	print("--- Normal Distribution Example ---")
	var mu = 0.0
	var sigma = 1.0
	var percentiles = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
	
	print("Normal distribution (μ=%s, σ=%s)" % [mu, sigma])
	for p in percentiles:
		var x = StatMath.PpfFunctions.normal_ppf(p, mu, sigma)
		# CDF for Normal(mu, sigma) is Φ((x-mu)/sigma), where Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
		var standard_normal_variate = (x - mu) / sigma
		var cdf = 0.5 * (1.0 + StatMath.ErrorFunctions.error_function(standard_normal_variate / sqrt(2.0)))
		print("p=%.3f → x=%.4f, Verification: CDF(%.4f)=%.4f" % [p, x, x, cdf])
	
	# Example 2: Exponential Distribution
	print("\n--- Exponential Distribution Example ---")
	var lambda = 2.0
	print("Exponential distribution (λ=%s)" % [lambda])
	for p in percentiles:
		var x = StatMath.PpfFunctions.exponential_ppf(p, lambda)
		# CDF for Exponential(lambda) is 1 - exp(-lambda * x)
		var cdf = 1.0 - exp(-lambda * x)
		print("p=%.3f → x=%.4f, Verification: CDF(%.4f)=%.4f" % [p, x, x, cdf])
	
	#Example 3: Beta Distribution
	#print(--- Beta Distribution Example ---")
	#var alpha = 2.0
	#var beta = 5.0
	#print("Beta distribution (α=%s, β=%s)" % [alpha, beta])
	#for p in percentiles:
	#  var x = StatMath.PpfFunctions.beta_ppf(p, alpha, beta)
	#  var cdf = StatMath.CdfFunctions.beta_cdf(x, alpha, beta)
	#  print("p=%.3f → x=%.4f, Verification: CDF(%.4f)=%.4f" % [p, x, x, cdf])
	
	# Example 4: Binomial Distribution
	print("\n--- Binomial Distribution Example ---")
	var n = 10
	var prob = 0.3
	print("Binomial distribution (n=%s, p=%s)" % [n, prob])
	for p in percentiles:
		var k = StatMath.PpfFunctions.binomial_ppf(p, n, prob)
		# Calculate cumulative probability up to k
		var cum_prob = 0.0
		for i in range(k+1):
			cum_prob += StatMath.PmfPdfFunctions.binomial_pmf(i, n, prob)
		print("p=%.3f → k=%d, Verification: CDF(%d)=%.4f" % [p, k, k, cum_prob])

	# Practical application example
	print("\n--- Practical Application Example ---")
	# Suppose we have student scores that follow a normal distribution
	var mean_score = 72.0
	var std_dev = 12.0
	
	# Let's determine the cut-off scores for different letter grades
	var grade_A = StatMath.PpfFunctions.normal_ppf(0.9, mean_score, std_dev)
	var grade_B = StatMath.PpfFunctions.normal_ppf(0.7, mean_score, std_dev)
	var grade_C = StatMath.PpfFunctions.normal_ppf(0.5, mean_score, std_dev)
	var grade_D = StatMath.PpfFunctions.normal_ppf(0.3, mean_score, std_dev)
	
	print("Grade cutoffs based on percentiles:")
	print("A: %.1f and above (top 10%%)" % grade_A)
	print("B: %.1f to %.1f (70th-90th percentile)" % [grade_B, grade_A])
	print("C: %.1f to %.1f (50th-70th percentile)" % [grade_C, grade_B])
	print("D: %.1f to %.1f (30th-50th percentile)" % [grade_D, grade_C])
	print("F: Below %.1f (bottom 30%%)" % grade_D)

	# Example 5: Basic Statistics - Analyzing Player Performance Data
	print("\n--- Basic Statistics Example ---")
	# Simulate player performance data (scores, reaction times, etc.)
	var raw_player_scores = [95.5, "invalid_entry", 87.2, null, 92.1, 88.8, 90.0, 94.3, 89.7, 91.2]
	print("Raw player data: %s" % [raw_player_scores])
	
	# Clean and sort the data
	var clean_scores: Array[float] = StatMath.HelperFunctions.sanitize_numeric_array(raw_player_scores)
	print("Cleaned and sorted scores: %s" % [clean_scores])
	
	# Calculate individual statistics
	var avg_score: float = StatMath.BasicStats.mean(clean_scores)
	var median_score: float = StatMath.BasicStats.median(clean_scores)
	var score_std_dev: float = StatMath.BasicStats.standard_deviation(clean_scores)
	var score_range: float = StatMath.BasicStats.range_spread(clean_scores)
	var mad: float = StatMath.BasicStats.median_absolute_deviation(clean_scores)
	
	print("Statistical Analysis:")
	print("  Mean score: %.2f" % avg_score)
	print("  Median score: %.2f" % median_score)
	print("  Standard deviation: %.2f" % score_std_dev)
	print("  Score range: %.2f" % score_range)
	print("  Median absolute deviation: %.2f" % mad)
	
	# Get comprehensive summary
	var summary: Dictionary = StatMath.BasicStats.summary_statistics(clean_scores)
	print("\nComprehensive Summary:")
	for key in summary.keys():
		if key == "count":
			print("  %s: %d" % [key, summary[key]])
		else:
			print("  %s: %.3f" % [key, summary[key]])
	
	# Practical use case: Detecting outliers
	print("\nOutlier Detection (values > 2 standard deviations from mean):")
	var outlier_threshold: float = 2.0 * score_std_dev
	for score in clean_scores:
		var deviation: float = abs(score - avg_score)
		if deviation > outlier_threshold:
			print("  Score %.2f is an outlier (%.2f std devs from mean)" % [score, deviation / score_std_dev])
	
	# Performance classification example
	print("\nPerformance Classification:")
	for score in clean_scores:
		var z_score: float = (score - avg_score) / score_std_dev
		var classification: String
		if z_score > 1.0:
			classification = "Above Average"
		elif z_score > 0.0:
			classification = "Average+"
		elif z_score > -1.0:
			classification = "Average-"
		else:
			classification = "Below Average"
		print("  Score %.2f (z=%.2f): %s" % [score, z_score, classification])

	get_tree().quit()
