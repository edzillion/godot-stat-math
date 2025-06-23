StatMath.PmfPdfFunctions
========================

Probability Mass Functions (PMF) and Probability Density Functions (PDF)
This class provides static methods to calculate probability mass functions for discrete
distributions and probability density functions for continuous distributions.

The PMF/PDF
gives the probability (or density) of a random variable taking on a specific value.
Distribution Categories:

* PMF for discrete distributions (Binomial, Poisson, Negative Binomial)

* PDF for continuous distributions (Normal, Exponential, Uniform, Gamma, Beta, Chi-squared, Student's t, F-distribution)

* Uses logarithmic calculations for numerical stability

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PmfPdfFunctions.function_name(parameters)

Functions
---------

.. function:: binomial_pmf(k_successes: int, n_trials: int, p_prob: float) -> float:

   Calculates the PMF of a binomial distribution: P(X = k | n, p).

   Returns the probability of observing exactly ``k`` successes in ``n``
   independent Bernoulli trials, each with success probability ``p``.
   Uses logarithmic calculations for numerical stability.

   Mathematical Note: ``P(X = k) = (n choose k) p^k (1-p)^(n-k)``

.. function:: poisson_pmf(k_events: int, lambda_param: float) -> float:

   Calculates the PMF of a Poisson distribution: P(X = k | λ).

   Returns the probability of observing exactly ``k`` events in a fixed interval,
   given an average rate ``λ`` of events. Uses logarithmic calculations for
   numerical stability.

   Mathematical Note: ``P(X = k) = (λ^k e^(-λ)) / k!``

.. function:: negative_binomial_pmf(k_trials: int, r_successes: int, p_prob: float) -> float:

   Calculates the PMF of a negative binomial distribution: P(X = k | r, p).

   Returns the probability that the ``r``-th success occurs on exactly the
   ``k``-th trial in independent Bernoulli trials with success probability ``p``.
   Uses logarithmic calculations for numerical stability.

   Mathematical Note: ``P(X = k) = (k-1 choose r-1) p^r (1-p)^(k-r)``

.. function:: normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:

   Calculates the PDF of a normal distribution: f(x; μ, σ).

   Returns the probability density at ``x`` for a normal (Gaussian) distribution
   with mean ``μ`` and standard deviation ``σ``.

   Mathematical Note: ``f(x) = (1/σ√(2π)) e^(-(x-μ)²/(2σ²))``

.. function:: exponential_pdf(x: float, lambda_param: float) -> float:

   Calculates the PDF of an exponential distribution: f(x; λ).

   Returns the probability density at ``x`` for an exponential distribution
   with rate parameter ``λ``. Used for modeling waiting times and decay processes.

   Mathematical Note: ``f(x) = λe^(-λx)`` for ``x ≥ 0``, ``0`` otherwise

.. function:: uniform_pdf(x: float, a: float, b: float) -> float:

   Calculates the PDF of a uniform distribution: f(x; a, b).

   Returns the probability density at ``x`` for a uniform distribution
   on the interval ``[a, b]``.

   Mathematical Note: ``f(x) = 1/(b-a)`` for ``a ≤ x ≤ b``, ``0`` otherwise

.. function:: gamma_pdf(x: float, k_shape: float, theta_scale: float) -> float:

   Calculates the PDF of a gamma distribution: f(x; k, θ).

   Returns the probability density at ``x`` for a gamma distribution
   with shape parameter ``k`` and scale parameter ``θ``.

   Mathematical Note: ``f(x) = (1/(Γ(k)θ^k)) x^(k-1) e^(-x/θ)`` for ``x ≥ 0``

.. function:: beta_pdf(x: float, alpha: float, beta_param: float) -> float:

   Calculates the PDF of a beta distribution: f(x; α, β).

   Returns the probability density at ``x`` for a beta distribution
   with shape parameters ``α`` and ``β``. Defined on [0, 1].

   Mathematical Note: ``f(x) = (Γ(α+β)/(Γ(α)Γ(β))) x^(α-1) (1-x)^(β-1)``

.. function:: weibull_pdf(x: float, scale_param: float, shape_param: float) -> float:

   Calculates the PDF of a Weibull distribution: f(x; λ, k).

   Returns the probability density at ``x`` for a Weibull distribution
   with scale parameter ``λ`` and shape parameter ``k``.
   Widely used in reliability analysis, survival analysis, and failure modeling.

   Mathematical Note: ``f(x) = (k/λ)(x/λ)^(k-1) e^(-(x/λ)^k)`` for ``x ≥ 0``

.. function:: lognormal_pdf(x: float, mu: float, sigma: float) -> float:

   Calculates the PDF of a lognormal distribution: f(x; μ, σ).

   Returns the probability density at ``x`` for a lognormal distribution
   with location parameter ``μ`` and scale parameter ``σ``.
   If X ~ Lognormal(μ, σ), then ln(X) ~ Normal(μ, σ).

   Mathematical Note: ``f(x) = (1/(xσ√(2π))) e^(-((ln(x)-μ)²)/(2σ²))`` for ``x > 0``

.. function:: chi_squared_pdf(x: float, k_df: float) -> float:

   Calculates the PDF of a chi-squared distribution: f(x; k).

   Returns the probability density at ``x`` for a chi-squared distribution
   with ``k`` degrees of freedom. This is a special case of the gamma PDF. See `gamma_pdf() <pmf_pdf_functions.html#gamma_pdf>`_ for the general gamma PDF implementation.

   Mathematical Note: ``f(x) = (1/(2^(k/2)Γ(k/2))) x^(k/2-1) e^(-x/2)`` for ``x ≥ 0``

.. function:: t_pdf(x: float, df_nu: float) -> float:

   Calculates the PDF of a Student's t-distribution: f(x; ν).

   Returns the probability density at ``x`` for a Student's t-distribution
   with ``ν`` (nu) degrees of freedom.

   Mathematical Note: ``f(x) = (Γ((ν+1)/2)/(√(νπ)Γ(ν/2))) (1 + x²/ν)^(-(ν+1)/2)``

.. function:: f_pdf(x: float, d1_df: float, d2_df: float) -> float:

   Calculates the PDF of an F-distribution: f(x; d1, d2).

   Returns the probability density at ``x`` for an F-distribution
   with numerator degrees of freedom ``d1`` and denominator degrees of freedom ``d2``.

   Mathematical Note: Uses `beta_function() <helper_functions.html#beta_function>`_ relationship for numerical stability

