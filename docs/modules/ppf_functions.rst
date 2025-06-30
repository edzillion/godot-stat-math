StatMath.PpfFunctions
=====================

Inverse Cumulative Distribution Functions (PPF/Quantile Functions)
This class provides static methods to calculate Percentile Point Functions (PPF),
also known as Inverse Cumulative Distribution Functions (ICDF) or quantile functions.

These functions return the value ``x`` such that ``CDF(x) = p``.
Distribution Categories:

* Continuous distributions (Normal, Exponential, Gamma, Beta, etc.)

* Discrete distributions (Binomial, Poisson, Geometric, etc.)

* Special distributions (Chi-Square, F-distribution, Student's t)

* Heavy-tailed distributions (Pareto, Weibull)

* Custom distributions (Discrete Histogram)

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PpfFunctions.function_name(parameters)

Constants
---------

.. data:: ACKLAM_COEFFICIENTS

   Value: ``preload("res://addons/godot-stat-math/tables/acklam_normal_ppf_coefficients.gd")``

Functions
---------

.. function:: uniform_ppf(p: float, a: float, b: float) -> float:

   Calculates the PPF of a uniform distribution: inverse of F(x; a, b).

   Returns the value ``x`` in ``[a, b]`` such that ``P(X ≤ x) = p``.

   Mathematical Note: ``PPF(p) = a + p(b-a)``

.. function:: normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:

   Calculates the PPF of a normal distribution: inverse of F(x; μ, σ).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for a normal distribution
   with mean ``μ`` and standard deviation ``σ``. Uses Acklam's algorithm
   for the standard normal, then transforms the result.

   Mathematical Note: Uses ``x = μ + σ * z`` where ``z`` is the standard normal quantile

.. function:: exponential_ppf(p: float, lambda_param: float) -> float:

   Calculates the PPF of an exponential distribution: inverse of F(x; λ).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for an exponential
   distribution with rate parameter ``λ``. Uses the closed-form solution.

   Mathematical Note: ``PPF(p) = -ln(1-p) / λ``

.. function:: beta_ppf(p: float, alpha_shape: float, beta_shape: float) -> float:

   Calculates the PPF of a beta distribution: inverse of F(x; α, β).

   Returns the value ``x`` in ``[0,1]`` such that ``P(X ≤ x) = p``
   for a beta distribution with shape parameters ``α`` and ``β``.
   Uses numerical methods as no closed-form solution exists.

   Mathematical Note: Uses binary search or Newton's method for ``Beta(2,2)``

.. function:: _beta_2_2_ppf_fast(p: float) -> float:

   Fast analytical PPF for Beta(2,2) distribution
   Solves the cubic equation p = x²(3-2x) using Newton's method
   This is MUCH faster than binary search for the common Beta(2,2) case

.. function:: gamma_ppf(p: float, k_shape: float, theta_scale: float) -> float:

   Calculates the PPF of a gamma distribution: inverse of F(x; k, θ).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for a gamma distribution
   with shape parameter ``k`` and scale parameter ``θ``. Uses binary search
   with Wilson-Hilferty approximation for initial guess when ``k > 1``.

   Mathematical Note: Uses Wilson-Hilferty transformation ``k(1 - 1/(9k) + z/(3√k))³`` for initial guess

.. function:: chi_square_ppf(p: float, k_df: float) -> float:

   Calculates the PPF of a chi-square distribution: inverse of F(x; k).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for a chi-square
   distribution with ``k`` degrees of freedom. Derived from gamma distribution
   as ``χ²(k) = Gamma(k/2, 2)``.

   Mathematical Note: ``χ²(k) ≡ Gamma(shape=k/2, scale=2)``

.. function:: f_ppf(p: float, d1: float, d2: float) -> float:

   Calculates the PPF of an F-distribution: inverse of F(x; d₁, d₂).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for an F-distribution
   with numerator degrees of freedom ``d₁`` and denominator degrees of freedom ``d₂``.
   Uses binary search with beta distribution relationship for initial guess.

   Mathematical Note: Related to Beta distribution via ``F = (d₂Y)/(d₁(1-Y))`` where ``Y ~ Beta(d₁/2, d₂/2)``

.. function:: t_ppf(p: float, df: float) -> float:

   Calculates the PPF of a Student's t-distribution: inverse of F(x; ν).

   Returns the value ``x`` such that ``P(T ≤ x) = p`` for a t-distribution
   with ``ν`` degrees of freedom. Uses symmetry and approximates with normal
   distribution for large degrees of freedom (``ν > 1000``).

   Mathematical Note: Uses symmetry ``t_p(ν) = -t_{1-p}(ν)`` and normal approximation for large ν

.. function:: binomial_ppf(p: float, n: int, prob_success: float) -> int:

   Calculates the PPF of a binomial distribution: inverse of F(k; n, p).

   Returns the smallest integer ``k`` (number of successes) such that ``CDF(k) ≥ p``
   for a binomial distribution with ``n`` trials and success probability ``p``.
   Uses linear search through possible values.

   Mathematical Note: ``P(X ≤ k) = Σᵢ₌₀ᵏ (n choose i) p^i (1-p)^(n-i)``

.. function:: poisson_ppf(p: float, lambda_param: float) -> int:

   Calculates the PPF of a Poisson distribution: inverse of F(k; λ).

   Returns the smallest integer ``k`` such that ``CDF(k) ≥ p`` for a Poisson
   distribution with average rate ``λ``. Uses linear search by summing PMF values
   up to ``λ + 10√λ + 20`` for practical bounds.

   Mathematical Note: ``P(X ≤ k) = Σᵢ₌₀ᵏ e^(-λ)λⁱ/i!``

.. function:: geometric_ppf(p: float, prob_success: float) -> int:

   Calculates the PPF of a geometric distribution: inverse of F(k; p).

   Returns the smallest integer ``k`` (number of trials) such that ``CDF(k) ≥ p``
   for a geometric distribution with success probability ``p``. Uses closed-form
   solution for efficiency.

   Mathematical Note: ``PPF(p) = ⌈ln(1-p) / ln(1-p_success)⌉``

.. function:: negative_binomial_ppf(p: float, r_successes: int, prob_success: float) -> int:

   Calculates the PPF of a negative binomial distribution: inverse of F(k; r, p).

   Returns the smallest integer ``k`` (number of trials) such that ``CDF(k) ≥ p``
   for a negative binomial distribution with ``r`` required successes and success
   probability ``p``. Uses linear search with mean-based bounds.

   Mathematical Note: Mean trials = ``r/p``, searches up to ``mean + 10σ``

.. function:: bernoulli_ppf(p: float, prob_success: float) -> int:

   Calculates the PPF of a Bernoulli distribution: inverse of F(k; p).

   Returns ``0`` (failure) or ``1`` (success) such that ``CDF(k) ≥ p``
   for a Bernoulli distribution with success probability ``p``. Special case
   of binomial distribution with ``n = 1``.

   Mathematical Note: ``PPF(p) = 0`` if ``p ≤ 1-p_success``, otherwise ``1``

.. function:: discrete_histogram_ppf(p: float, values: Array, probabilities: Array[float]) -> Variant:

   Calculates the PPF of a discrete histogram distribution: inverse of F(x; values, probabilities).

   Returns the value from ``values`` array corresponding to the smallest cumulative
   probability ``≥ p``. Creates a user-defined discrete distribution from
   provided values and their associated probabilities.

   Mathematical Note: Implements inverse transform sampling for discrete distributions

.. function:: pareto_ppf(p: float, scale_param: float, shape_param: float) -> float:

   Calculates the PPF of a Pareto distribution: inverse of F(x; scale, shape).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for a Pareto distribution
   with scale parameter (minimum value) and shape parameter. Uses closed-form solution.

   Mathematical Note: ``PPF(p) = scale / (1-p)^(1/shape)``

.. function:: weibull_ppf(p: float, scale_param: float, shape_param: float) -> float:

   Calculates the PPF of a Weibull distribution: inverse of F(x; λ, k).

   Returns the value ``x`` such that ``P(X ≤ x) = p`` for a Weibull distribution
   with scale parameter ``λ`` and shape parameter ``k``. Uses closed-form solution.
   Widely used for reliability analysis and survival modeling.

   Mathematical Note: ``PPF(p) = λ * (-ln(1-p))^(1/k)``

