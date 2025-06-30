StatMath.Distributions
======================

Random Variate Generation for Statistical Distributions
This class provides methods to generate random numbers (variates) from various
common statistical distributions.

These functions are essential for simulations,
modeling, and statistical analysis in game development.
Distribution Categories:

* Discrete distributions (Bernoulli, Binomial, Geometric, Poisson)

* Continuous distributions (Normal, Exponential, Gamma, Beta, etc.)

* Specialized distributions (Triangular, Pareto, Weibull, Cauchy)

* Custom distributions (Pseudo, Siege, Histogram)
Generates an integer from a Bernoulli distribution.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.Distributions.function_name(parameters)

Functions
---------

.. function:: randi_bernoulli(p: float = 0.5) -> int:

   Generates an integer from a Bernoulli distribution.

   Returns 1 (success) with probability ``p``, and 0 (failure) with probability ``1-p``.
   This is the fundamental building block for many other discrete distributions.

   Mathematical Note: ``E[X] = p``, ``Var(X) = p(1-p)``

.. function:: randi_binomial(p: float, n: int) -> int:

   Generates an integer from a Binomial distribution.

   Returns the number of successes in ``n`` independent Bernoulli trials,
   each with success probability ``p``. Uses repeated geometric distribution sampling.

   Mathematical Note: ``E[X] = np``, ``Var(X) = np(1-p)``

.. function:: randi_geometric(p: float) -> int:

   Generates an integer from a Geometric distribution.

   Returns the number of Bernoulli trials needed to get one success (always ≥ 1).
   Uses fast direct sampling for optimal performance.

   Mathematical Note: ``E[X] = 1/p``, ``Var(X) = (1-p)/p²``

.. function:: randi_poisson(lambda_param: float) -> int:

   Generates an integer from a Poisson distribution.

   Models the number of events occurring in a fixed interval when events occur
   independently at a constant average rate ``lambda_param``. Uses Knuth's algorithm.

   Mathematical Note: ``E[X] = Var(X) = λ``

.. function:: randi_uniform(min_val: int, max_val: int) -> int:

   Generates an integer from a discrete uniform distribution.

   Returns a random integer uniformly distributed in ``[min_val, max_val]`` (both inclusive).
   Wrapper around Godot's randi_range for consistency with other distribution functions.

.. function:: randi_pseudo(c_param: float) -> int:

   Generates an integer from a custom pseudo-random process.

   Uses an iterative Bernoulli process with increasing success probability.
   Success probability starts at ``c_param`` and increases by ``c_param`` each trial.

.. function:: randi_seige(w: float, c_0: float, c_win: float, c_lose: float) -> int:

   Generates an integer from a custom siege scenario model.

   Simulates a scenario where capture probability changes based on win/loss outcomes.
   Attack probability is ``w``, capture probability starts at ``c_0`` and
   changes by ``c_win`` or ``c_lose`` based on attack results.

.. function:: randf_uniform(a: float, b: float) -> float:

   Generates a float from a continuous uniform distribution.

   Returns a random float uniformly distributed in the interval ``[a, b)``.
   All values in the interval have equal probability density.

   Mathematical Note: ``E[X] = (a+b)/2``, ``Var(X) = (b-a)²/12``

.. function:: randf_exponential(lambda_param: float) -> float:

   Generates a float from an Exponential distribution.

   Models the time between events in a Poisson process with rate ``lambda_param``.
   Uses inverse transform sampling: ``-log(1-U)/λ`` where U ~ Uniform(0,1).

   Mathematical Note: ``E[X] = 1/λ``, ``Var(X) = 1/λ²``

.. function:: randf_erlang(k: int, lambda_param: float) -> float:

   Generates a float from an Erlang distribution.

   Special case of `randf_gamma() <distributions.html#randf_gamma>`_ distribution with integer shape parameter ``k``.
   Represents the sum of ``k`` independent Exponential(``lambda_param``) variables.

   Mathematical Note: ``E[X] = k/λ``, ``Var(X) = k/λ²``

.. function:: randf_gamma(shape: float, scale: float = 1.0) -> float:

   Generates a float from a Gamma distribution.

   Uses scale parameterization: ``Gamma(α, θ)`` where mean = ``αθ`` and variance = ``αθ²``.
   Uses Marsaglia and Tsang's method for shape ≥ 1, rejection sampling for shape < 1.

   Mathematical Note: ``E[X] = αθ``, ``Var(X) = αθ²``

.. function:: randf_beta(alpha: float, beta_param: float) -> float:

   Generates a float from a Beta distribution.

   Uses the gamma-to-beta transformation: if ``X ~ randf_gamma(α,1)`` and ``Y ~ randf_gamma(β,1)``,
   then ``X/(X+Y) ~ Beta(α,β)``. Values are always in ``[0,1]``.

   Mathematical Note: ``E[X] = α/(α+β)``, ``Var(X) = αβ/[(α+β)²(α+β+1)]``

.. function:: randf_gaussian() -> float:

   Generates a float from a standard normal distribution N(0,1).

   Uses the Box-Muller transform to convert uniform random variables to normal.
   Returns one of the two generated variates (the other is discarded).

   Mathematical Note: ``E[X] = 0``, ``Var(X) = 1``

.. function:: randf_normal(mu: float = 0.0, sigma: float = 1.0) -> float:

   Generates a float from a normal distribution with specified mean and standard deviation.

   Transforms a standard normal variate: ``Z*σ + μ`` where ``Z ~ N(0,1)``.
   Defaults to ``N(0,1)`` if parameters are not provided.

   Mathematical Note: ``E[X] = μ``, ``Var(X) = σ²``

.. function:: randf_lognormal(mu: float = 0.0, sigma: float = 1.0) -> float:

   Generates a float from a Lognormal distribution.

   Uses the fundamental relationship: if ``X ~ Normal(μ, σ)``, then ``exp(X) ~ Lognormal(μ, σ)``.
   The lognormal distribution models positive values and is commonly used for modeling
   prices, incomes, and other quantities that cannot be negative.

   Mathematical Note: ``E[X] = exp(μ + σ²/2)``, ``Var(X) = [exp(σ²) - 1] × exp(2μ + σ²)``

.. function:: randf_cauchy(location: float = 0.0, scale: float = 1.0) -> float:

   Generates a float from a Cauchy (Lorentzian) distribution.

   Uses the ratio of two independent standard normal variates. The Cauchy distribution
   has undefined mean and variance due to heavy tails, making it useful for modeling
   extreme events and outliers.

   Mathematical Note: Mean and variance are undefined due to heavy tails

.. function:: randf_triangular(min_value: float, max_value: float, mode_value: float) -> float:

   Generates a float from a Triangular distribution.

   Creates values with a triangular probability density function, peaking at ``mode_value``.
   Uses inverse transform sampling for efficiency. Commonly used in game development
   when you know minimum, most likely, and maximum values.

   Mathematical Note: ``E[X] = (a+b+c)/3`` where c is the mode

.. function:: randf_pareto(scale_param: float, shape_param: float) -> float:

   Generates a float from a Pareto distribution (power law).

   Models the famous "80/20 rule" and heavy-tailed distributions. Uses exponential
   transformation method: if ``Y ~ Exponential(shape)``, then
   ``X = scale * exp(Y) ~ Pareto(scale, shape)``.

   Mathematical Note: Mean exists only if ``shape > 1``, variance exists only if ``shape > 2``

.. function:: randf_weibull(scale_param: float, shape_param: float) -> float:

   Generates a float from a Weibull distribution.

   Widely used for reliability analysis, survival analysis, and weather modeling.
   Uses inverse transform sampling: ``λ * (-ln(1-U))^(1/k)`` where U ~ Uniform(0,1).

   Mathematical Note: Mean = ``λ * Γ(1 + 1/k)`` where Γ is the gamma function. See `gamma_function() <helper_functions.html#gamma_function>`_ for the gamma function implementation.

.. function:: randv_histogram(values: Array, probabilities: Array) -> Variant:

   Generates a random value from a discrete histogram distribution.

   Samples from provided values using their associated probabilities. Probabilities
   are automatically normalized, so they don't need to sum to 1. Uses cumulative
   distribution function (CDF) for efficient sampling.

