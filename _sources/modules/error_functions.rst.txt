StatMath.ErrorFunctions
=======================

Error Functions and Related Mathematical Functions
These functions are commonly used in probability, statistics, and partial differential
equations.

They are particularly important for working with normal distributions and
various statistical calculations.
Mathematical Background:
The error function is defined as: ``erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt``
Computes the error function erf(x).
The error function is an odd function (``erf(-x) = -erf(x)``) and is related to

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.ErrorFunctions.function_name(parameters)

Functions
---------

.. function:: erf(x: float) -> float:

   Computes the error function erf(x).

   The error function is an odd function (``erf(-x) = -erf(x)``) and is related to
   the cumulative distribution function (`normal_cdf() <cdf_functions.html#normal_cdf>`_) of the normal distribution.
   Uses the Abramowitz and Stegun formula 7.1.26 approximation with a maximum error of 1.5 × 10⁻⁷.

   Mathematical Note: ``erf(0) = 0``, ``erf(∞) = 1``, ``erf(-∞) = -1``

.. function:: erfc(x: float) -> float:

   Computes the complementary error function erfc(x).

   Defined as ``erfc(x) = 1 - erf(x)``. This function is useful for numerical
   stability when computing values close to 1.

   Mathematical Note: ``erfc(0) = 1``, ``erfc(∞) = 0``, ``erfc(-∞) = 2``

.. function:: erf_inv(x: float) -> float:

   Computes the inverse error function erf⁻¹(x).

   Returns the value y such that ``erf(y) = x``. Uses an iterative approximation
   method with Newton-Raphson refinement for improved accuracy.
   Valid input range: ``-1 < x < 1``

   Mathematical Note: ``erf_inv(0) = 0``, ``erf_inv(1)`` approaches ``∞``,
   ``erf_inv(-1)`` approaches ``-∞``

.. function:: erfc_inv(x: float) -> float:

   Computes the inverse complementary error function erfc⁻¹(x).

   Returns the value y such that ``erfc(y) = x``.
   Valid input range: ``0 < x < 2``

   Mathematical Note: ``erfc_inv(1) = 0``, ``erfc_inv(0)`` approaches ``∞``,
   ``erfc_inv(2)`` approaches ``-∞``

.. function:: log_gamma(x: float) -> float:

   Natural logarithm of the absolute value of the gamma function.

   The gamma function ``Γ(x)`` is a generalization of the factorial function to
   real and complex numbers. For positive integers: ``Γ(n) = (n-1)!``
   This function returns ``ln|Γ(x)|`` to avoid overflow issues.

   Mathematical Note: ``Γ(1) = 1``, ``Γ(n) = (n-1)!`` for positive integers

.. function:: gamma(x: float) -> float:

   The gamma function Γ(x).

   For positive integers: ``Γ(n) = (n-1)!``
   Uses the exponential of `log_gamma() <error_functions.html#log_gamma>`_ to compute the actual gamma value.

   Mathematical Note: ``Γ(1) = 1``, ``Γ(0.5) = √π``

