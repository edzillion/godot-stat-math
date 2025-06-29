StatMath.SamplingGenPerfTest
============================

Performance Test Suite for StatMath.

SamplingGen Module
Tests random number generation and sampling functions to catch
performance regressions in critical game random systems.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.SamplingGenPerfTest.function_name(parameters)

Constants
---------

.. data:: BATCH_SIZES

   Value: ``Array[int] = [32, 256, 1024] # we only support degrees up to 1024``

.. data:: DIMENSIONS

   Value: ``Array[int] = [1, 3, 10] # better spread``

.. data:: GENERATORS

   Value: ``Array[SamplingGen.SamplingMethod] = [``

Functions
---------

.. function:: get_module_name() -> String:

.. function:: test_generate_samples(

   Test generate_samples performance with parametrized combinations

.. function:: test_coordinated_shuffle_performance() -> void:

.. function:: test_sample_indices_performance() -> void:

