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
