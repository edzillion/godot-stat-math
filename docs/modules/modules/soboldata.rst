StatMath.SobolData
==================

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.SobolData.function_name(parameters)

Constants
---------

.. data:: DIRECTION_NUMBERS

   Value: ``Array[Array] = [``

.. data:: POLYNOMIAL_DATA

   Value: ``Dictionary[int, Dictionary] = {``

Functions
---------

.. function:: get_direction_numbers(dimension: int) -> Array:

   Get direction numbers for a specific dimension

.. function:: get_max_dimension() -> int:

   Get maximum supported dimension

.. function:: has_dimension(dimension: int) -> bool:

   Check if direction numbers are available for dimension

