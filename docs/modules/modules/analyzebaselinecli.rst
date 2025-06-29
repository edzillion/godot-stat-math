StatMath.AnalyzeBaselineCli
===========================

CLI Script for Baseline Analysis
Usage (run in headless mode by loading the scene):
godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.

tscn
godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --save-json output.json
godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --test-name basic_stats_mean
godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --help

Usage
-----

.. code-block:: gdscript

   # Access via StatMath singleton
   var result = StatMath.AnalyzeBaselineCli.function_name(parameters)

Functions
---------

.. function:: _ready():

.. function:: _show_help():

.. function:: _save_analysis_to_json(data: Dictionary, filepath: String):

