StatMath.PrimeNumbersData
=========================

Prime numbers table for statistical algorithms and quasi-random generation
Contains the first 2000 prime numbers for use in Halton sequences and other
statistical methods that require prime bases.

Generated using: Python trial division algorithm up to 17389
This provides sufficient primes for high-dimensional quasi-random sequences.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PrimeNumbersData.function_name(parameters)

Constants
---------

.. data:: VALUES

   Value: ``Array[int] = [``

Functions
---------

.. function:: get_nth_prime(n: int) -> int:

   Returns the nth prime number (0-indexed)

