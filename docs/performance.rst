Performance Testing
==================

This page displays the current performance metrics for the Godot Stat Math library. The dashboard shows execution times for all performance tests, comparing current results against established baselines.

.. testdirective::

.. performancedashboard::

Test Methodology
----------------

* **Baseline**: Established performance thresholds based on historical data
* **Current**: Latest test execution times in milliseconds  
* **Diff %**: Percentage change from baseline (negative = improvement, positive = regression)
* **Threshold**: Acceptable variance range (typically ±20%)
* **Status**: Pass/fail status based on threshold compliance

Performance tests are run automatically and results are updated with each test execution. Tests marked as "disabled" are not currently being evaluated for performance regressions.

Understanding the Results
------------------------

🟢 **Within threshold**: Performance within acceptable thresholds

🔴 **Above threshold**: Performance degradation beyond threshold

⚪ **Disabled**: Tests currently disabled for performance monitoring

The dashboard is sorted by execution time (slowest functions first) to help identify potential optimization targets. 