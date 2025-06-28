StatMath.CdfFunctions
=====================

Cumulative Distribution Functions (CDF)
This class provides static methods to calculate the cumulative distribution function
for various statistical distributions.

The CDF, ``F(x)``, gives the probability
that a random variable X will take a value less than or equal to x.
Distribution Categories:

* Continuous distributions (Normal, Exponential, Gamma, Beta, etc.)

* Discrete distributions (Binomial, Poisson, Geometric, etc.)

* Special distributions (Chi-Square, F-distribution, Student's t)

* Heavy-tailed distributions (Pareto, Weibull)

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.CdfFunctions.function_name(parameters)

Functions
---------

.. function:: uniform_cdf(x: float, a: float, b: float) -> float:

   Calculates the CDF of a uniform distribution: F(x; a, b).

   Returns the probability that a random variable from a uniform distribution
   on the interval ``[a, b]`` is less than or equal to x.

   Mathematical Note: ``F(x) = (x-a)/(b-a)`` for ``a ≤ x ≤ b``

.. function:: normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:

   Calculates the CDF of a normal distribution: F(x; μ, σ).

   Returns the probability that a random variable from a normal (Gaussian)
   distribution with mean ``μ`` and standard deviation ``σ``
   is less than or equal to x. Uses the `erf() <error_functions.html#erf>`_ for computation.

   Mathematical Note: ``F(x) = (1/2)[1 + erf((x-μ)/(σ√2))]`` See `erf() <error_functions.html#erf>`_ for the error function implementation.

.. function:: exponential_cdf(x: float, lambda_param: float) -> float:

   Calculates the CDF of an exponential distribution: F(x; λ).

   Returns the probability that a random variable from an exponential
   distribution with rate parameter ``λ`` is less than or equal to x.

   Mathematical Note: ``F(x) = 1 - e^(-λx)`` for ``x ≥ 0``

.. function:: beta_cdf(x: float, alpha: float, beta_param: float) -> float:

   Calculates the CDF of a beta distribution: F(x; α, β).

   Returns the probability that a random variable from a beta distribution
   with shape parameters ``α`` and ``β`` is less than or equal to x.
   Uses the `incomplete_beta() <helper_functions.html#incomplete_beta>`_.

   Mathematical Note: ``F(x) = I_x(α, β)`` where I is the incomplete beta function. See `incomplete_beta() <helper_functions.html#incomplete_beta>`_ for implementation details.

.. function:: gamma_cdf(x: float, k_shape: float, theta_scale: float) -> float: # Renamed k, theta

   Calculates the CDF of a gamma distribution: F(x; k, θ).

   Returns the probability that a random variable from a gamma distribution
   with shape parameter ``k`` and scale parameter ``θ`` is less than or equal to x.
   Uses the `lower_incomplete_gamma_regularized() <helper_functions.html#lower_incomplete_gamma_regularized>`_.

   Mathematical Note: ``F(x) = P(k, x/θ)`` where P is the regularized lower incomplete gamma function. See `lower_incomplete_gamma_regularized() <helper_functions.html#lower_incomplete_gamma_regularized>`_ for implementation details.

.. function:: chi_square_cdf(x: float, k_df: float) -> float:

   Calculates the CDF of a chi-square distribution: F(x; k).

   Returns the probability that a random variable from a chi-square distribution
   with ``k`` degrees of freedom is less than or equal to x.
   This is a special case of the gamma distribution.

   Mathematical Note: Chi-square with k df is ``gamma_cdf(k/2, 2)``. See `gamma_cdf() <cdf_functions.html#gamma_cdf>`_ for the gamma CDF implementation.

.. function:: f_cdf(x: float, d1_df: float, d2_df: float) -> float: # Renamed d1, d2

   Calculates the CDF of an F-distribution: F(x; d1, d2).

   Returns the probability that a random variable from an F-distribution
   with ``d1`` and ``d2`` degrees of freedom is less than or equal to x.
   Uses the regularized incomplete beta function.

   Mathematical Note: ``F(x) = I_z(d1/2, d2/2)`` where ``z = d1x/(d1x + d2)``

.. function:: t_cdf(x_val: float, df_nu: float) -> float: # Renamed x, df

   Calculates the CDF of a Student's t-distribution: F(x; ν).

   Returns the probability that a random variable from a Student's t-distribution
   with ``ν`` (nu) degrees of freedom is less than or equal to x.
   Uses the regularized incomplete beta function.

   Mathematical Note: Uses transformation via `incomplete_beta() <helper_functions.html#incomplete_beta>`_

.. function:: binomial_cdf(k_successes: int, n_trials: int, p_prob: float) -> float:

   Calculates the CDF of a binomial distribution: F(k; n, p).

   Returns the probability of observing ``k`` or fewer successes in ``n``
   independent Bernoulli trials, each with success probability ``p``.
   Computed as the sum of `binomial_pmf() <pmf_pdf_functions.html#binomial_pmf>`_ values.

   Mathematical Note: ``F(k) = Σᵢ₌₀ᵏ (n choose i) p^i (1-p)^(n-i)``

.. function:: poisson_cdf(k_events: int, lambda_param: float) -> float:

   Calculates the CDF of a Poisson distribution: F(k; λ).

   Returns the probability of observing ``k`` or fewer events in a fixed interval,
   given an average rate ``λ`` of events. Computed as the sum of Poisson PMF values. See `poisson_pmf() <pmf_pdf_functions.html#poisson_pmf>`_ for the PMF implementation.

   Mathematical Note: ``F(k) = Σᵢ₌₀ᵏ (λ^i e^(-λ))/i!``

.. function:: geometric_cdf(k_trials: int, p_prob: float) -> float:

   Calculates the CDF of a geometric distribution: F(k; p).

   Returns the probability that the first success in independent Bernoulli trials
   occurs on or before the ``k``-th trial. Assumes ``k ≥ 1``.

   Mathematical Note: ``F(k) = 1 - (1-p)^k``

.. function:: negative_binomial_cdf(k_trials: int, r_successes: int, p_prob: float) -> float:

   Calculates the CDF of a negative binomial distribution: F(k; r, p).

   Returns the probability that the ``r``-th success occurs on or before
   the ``k``-th trial in independent Bernoulli trials.
   Computed as the sum of `negative_binomial_pmf() <pmf_pdf_functions.html#negative_binomial_pmf>`_ values.

   Mathematical Note: ``F(k) = Σᵢ₌ᵣᵏ (i-1 choose r-1) p^r (1-p)^(i-r)``

.. function:: pareto_cdf(x: float, scale_param: float, shape_param: float) -> float:

   Calculates the CDF of a Pareto distribution: F(x; scale, shape).

   Returns the probability that a random variable from a Pareto distribution
   with scale parameter (minimum value) and shape parameter is less than or equal to x.
   Uses the closed-form solution.

   Mathematical Note: ``F(x) = 1 - (scale/x)^shape`` for ``x ≥ scale``

.. function:: weibull_cdf(x: float, scale_param: float, shape_param: float) -> float:

   Calculates the CDF of a Weibull distribution: F(x; λ, k).

   Returns the probability that a random variable from a Weibull distribution
   with scale parameter ``λ`` and shape parameter ``k`` is less than or equal to x.
   Uses the closed-form solution. Widely used for reliability analysis.

   Mathematical Note: ``F(x) = 1 - exp(-(x/λ)^k)`` for ``x ≥ 0``

