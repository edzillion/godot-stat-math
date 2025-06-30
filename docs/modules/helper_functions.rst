StatMath.HelperFunctions
========================

Core Mathematical Helper Functions
This class provides essential mathematical utility functions including combinatorial
calculations, special functions (Gamma, Beta), and numerical utilities.

These functions
serve as the mathematical foundation for statistical calculations throughout the StatMath library.
Mathematical Categories:

* Combinatorial functions (binomial coefficients, factorials)

* Gamma and Beta functions with their incomplete variants

* Array sanitization and preprocessing utilities

* Logarithmic versions for numerical stability
Calculates the binomial coefficient C(n, r) or "n choose r".
Computes the number of ways to choose r items from a set of n items without
regard to the order of selection. Uses symmetry optimization and iterative
calculation to maintain numerical precision.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.HelperFunctions.function_name(parameters)

Functions
---------

.. function:: binomial_coefficient(n: int, r: int) -> float:

   Calculates the binomial coefficient C(n, r) or "n choose r".

   Computes the number of ways to choose r items from a set of n items without
   regard to the order of selection. Uses symmetry optimization and iterative
   calculation to maintain numerical precision.

   Formula: ``C(n,r) = n! / (r! × (n-r)!)``

   Uses symmetry ``C(n,r) = C(n, n-r)`` for efficiency.

.. function:: log_factorial(n: int) -> float:

   Calculates the natural logarithm of n factorial: log(n!).

   Computes ``ln(n!)`` directly without calculating the factorial itself,
   avoiding overflow issues with large factorials. Essential for statistical
   calculations involving large numbers.

   Formula: ``log(n!) = Σᵢ₌₂ⁿ log(i)``

   Special Cases: ``log(0!) = log(1!) = 0``

.. function:: log_binomial_coef(n: int, k: int) -> float:

   Calculates the natural logarithm of the binomial coefficient: log(C(n,k)).

   Computes ``ln(C(n,k))`` using logarithmic arithmetic to avoid overflow
   issues with large binomial coefficients. More numerically stable than
   ``log(binomial_coefficient(n,k))`` for large values.

   Formula: ``log(C(n,k)) = Σᵢ₌₁ᵏ [log(n-i+1) - log(i)]``

.. function:: gamma_function(z: float) -> float:

   Computes the Gamma function Γ(z).

   The Gamma function is a generalization of the factorial function to real and complex numbers.
   For positive integers: ``Γ(n) = (n-1)!``
   Uses the Lanczos approximation for positive values and the reflection formula for negative values.

   Mathematical Note: ``Γ(z)Γ(1-z) = π/sin(πz)`` (reflection formula)

.. function:: log_gamma(z: float) -> float:

   Computes the natural logarithm of the Gamma function: log(Γ(z)).

   More numerically stable than ``log(gamma_function(z))`` for large z.
   See `gamma_function() <helper_functions.html#gamma_function>`_ for the gamma function implementation.
   Uses Lanczos approximation directly in logarithmic form to avoid overflow.

   Mathematical Note: Only defined for ``z > 0`` where ``Γ(z) > 0``

.. function:: beta_function(a: float, b: float) -> float:

   Computes the Beta function B(a, b).

   The Beta function is defined as ``B(a,b) = Γ(a)Γ(b) / Γ(a+b)``.
   Uses logarithmic arithmetic for numerical stability with large parameter values.

   Mathematical Note: ``B(a,b) = B(b,a)`` (symmetric property)

.. function:: log_beta_function_direct(a: float, b: float) -> float:

   Computes the natural logarithm of the Beta function: log(B(a,b)).

   More numerically stable than ``log(beta_function(a,b))`` for large parameters.
   See `beta_function() <helper_functions.html#beta_function>`_ for the beta function implementation.
   Formula: ``log(B(a,b)) = log(Γ(a)) + log(Γ(b)) - log(Γ(a+b))``

.. function:: incomplete_beta(x_val: float, a: float, b: float) -> float:

   Computes the regularized incomplete Beta function: I_x(a, b).

   Calculates ``I_x(a,b) = B(x;a,b) / B(a,b)`` where ``B(x;a,b)`` is
   the incomplete Beta function. Uses numerical integration method for basic functionality.

   Mathematical Note: ``I_0(a,b) = 0``, ``I_1(a,b) = 1``

.. function:: lower_incomplete_gamma_regularized(a: float, z: float) -> float:

   Computes the regularized lower incomplete Gamma function: P(a,z).

   Calculates ``P(a,z) = γ(a,z) / Γ(a)`` where ``γ(a,z)`` is the
   lower incomplete Gamma function. Uses different numerical methods based on parameter
   ranges for optimal stability.

   Mathematical Note: ``P(a,0) = 0``, ``P(a,∞) = 1``

.. function:: _gamma_series_expansion(a: float, z: float) -> float:

   Helper function for series expansion method (used when z < a + 1).

   Implements the series expansion form of the incomplete Gamma function for better
   numerical stability in the appropriate parameter range.

   Formula: :math:`P(a,z) = \frac{z^a e^{-z}}{\Gamma(a)} \sum_{n=0}^{\infty} \frac{z^n}{a(a+1)\cdots(a+n)}`

.. function:: _gamma_continued_fraction(a: float, z: float) -> float:

   Helper function for continued fraction method (used when z >= a + 1).

   Implements the continued fraction form of the incomplete Gamma function for better
   numerical stability with larger z values relative to a.

   Uses the continued fraction representation: :math:`Q(a,z) = \frac{z^a e^{-z}}{\Gamma(a)} \frac{1}{z+1-a-\frac{1 \cdot (1-a)}{z+3-a-\frac{2 \cdot (2-a)}{z+5-a-\cdots}}}`

.. function:: sanitize_numeric_array(input_array: Array) -> Array[float]:

   Sanitizes and sorts a mixed-type array into a clean Array[float].

   Accepts an Array with elements of any type, filters out non-numeric values,
   converts remaining elements to float, and returns a sorted array. Essential
   for preprocessing data before statistical calculations.

   Non-numeric values (strings, nulls, objects, etc.) are silently skipped.

.. function:: convert_to_float_array(input_array: Array) -> Array[float]:

   Converts a generic Array to a typed Array[float].

   Essential helper for converting test data arrays (which are generic Array types)
   to the typed Array[float] required by StatMath functions. Each element is explicitly
   cast to float to ensure type safety.

   Use this when working with data from external sources like test data files or
   JSON imports that produce generic arrays.

.. function:: validate_indices(samples: Array[int], population_size: int) -> bool:

   Validates that all indices in a sample are within valid range [0, population_size-1].

   Used by sampling tests to ensure index validity without checking uniqueness.
   Returns true if all indices are valid, false otherwise with error logging.

.. function:: validate_unique_indices(samples: Array[int], population_size: int) -> bool:

   Validates that all indices in a sample are unique and within valid range.

   Used by sampling tests to ensure both validity and uniqueness of indices.
   Returns true if all indices are valid and unique, false otherwise with error logging.

.. function:: get_cdf_value(distribution: Variant, x: float, params: Array) -> float:

   Gets CDF value for any distribution using the appropriate StatMath function.

   Centralized helper that routes CDF calculations to the correct StatMath function
   based on distribution type. Handles both enum and string distribution identifiers.

.. function:: get_ppf_value(distribution: Variant, p: float, params: Array) -> float:

   Gets PPF value for any distribution using the appropriate StatMath function.

   Centralized helper that routes PPF calculations to the correct StatMath function
   based on distribution type. Handles both enum and string distribution identifiers.

.. function:: string_to_distribution_enum(distribution: String) -> StatMath.SupportedDistributions:

   Converts a string distribution name to ``StatMath.SupportedDistributions`` enum.

   Centralized helper for converting string identifiers to proper enum values.
   Supports both uppercase and lowercase string inputs for flexibility.

