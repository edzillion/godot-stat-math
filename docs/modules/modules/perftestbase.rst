StatMath.PerfTestBase
=====================

Base class for all performance test suites
Each test suite now operates independently, saving its own results immediately
to eliminate shared state conflicts between test suites.

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.PerfTestBase.function_name(parameters)

Constants
---------

.. data:: REGRESSION_THRESHOLD

   Value: ``float = PerfTestManager.REGRESSION_THRESHOLD``

.. data:: WARMUP_ITERATIONS

   Value: ``int = PerfTestManager.WARMUP_ITERATIONS``

.. data:: MEASUREMENT_ITERATIONS

   Value: ``int = PerfTestManager.MEASUREMENT_ITERATIONS``

.. data:: TEST_ITERATIONS

   Value: ``int = 100  # Number of function calls per performance test``

.. data:: DATASET_SIZES

   Value: ``Array[int] = [100, 1000, 5000]  # Standard dataset sizes for stats``

Functions
---------

.. function:: before() -> void:

   Setup performance manager for this specific test suite

.. function:: after() -> void:

   Clean up this test suite independently

.. function:: _measure_test(test_name: String, test_func: Callable) -> Dictionary:

   Measure a test function's performance - delegates to manager

.. function:: _load_baseline() -> Dictionary:

   Load baseline data - delegates to manager

.. function:: _check_performance_regression(module_name: String, test_name: String, current_results: Dictionary, baseline_data: Dictionary) -> void:

   Check performance regression and save immediately - delegates to manager

.. function:: _generate_test_data(size: int, seed: int = 12345) -> Array[float]:

   Generate reproducible test data - delegates to manager

.. function:: get_module_name() -> String:

   Get the module name for this test (override in subclasses)

