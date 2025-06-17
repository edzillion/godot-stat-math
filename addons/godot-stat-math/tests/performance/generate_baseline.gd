# res://addons/godot-stat-math/tests/performance/generate_baseline.gd
extends Node

## DEPRECATED: Manual Baseline Generator
## 
## ⚠️  This file is no longer needed! ⚠️
##
## The performance testing system now automatically generates baselines from
## successful test runs. Each time you run performance tests and they pass,
## the system:
##
## 1. Saves the results as pass_YYYY-MM-DD_HH-MM-SS.json
## 2. Automatically updates baseline.json using the last 50 successful runs
## 3. Uses robust median statistics for stable baselines
## 4. Manages cleanup of old snapshots automatically
##
## To run performance tests and update baselines automatically:
## - Run GDUnit4 performance tests normally
## - Successful runs will automatically update the baseline
## - Failed runs are saved as fail_ files for debugging but don't affect baselines
##
## This file remains for reference but should be deleted once you confirm
## the automatic system is working correctly.

func _ready() -> void:
	print("======================================")
	print("⚠️  DEPRECATED: Manual Baseline Generator")
	print("======================================")
	print("")
	print("This script is no longer needed!")
	print("")
	print("The performance testing system now automatically:")
	print("✅ Saves successful runs as pass_YYYY-MM-DD.json")
	print("✅ Updates baseline.json from the last 50 successful runs") 
	print("✅ Uses robust median statistics for stable baselines")
	print("✅ Cleans up old snapshots automatically")
	print("")
	print("Just run your GDUnit4 performance tests normally!")
	print("Successful test runs will automatically update baselines.")
	print("")
	print("======================================")
	get_tree().quit()
