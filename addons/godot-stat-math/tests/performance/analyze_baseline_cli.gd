#!/usr/bin/env godot --script
# res://addons/godot-stat-math/tests/performance/analyze_baseline_cli.gd

## CLI Script for Baseline Analysis
##
## Usage (run in headless mode by loading the scene):
##   godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn
##   godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --save-json output.json
##   godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --test-name basic_stats_mean
##   godot --headless addons/godot-stat-math/tests/performance/analyze_baseline_cli.tscn --help

extends Node

func _ready():
	# Parse command line arguments
	var args: PackedStringArray = OS.get_cmdline_args()
	var save_json_file: String = ""
	var specific_test: String = ""
	var show_help: bool = false
	var quiet_mode: bool = false
	
	# Simple argument parsing
	for i in range(args.size()):
		match args[i]:
			"--save-json":
				if i + 1 < args.size():
					save_json_file = args[i + 1]
			"--test-name":
				if i + 1 < args.size():
					specific_test = args[i + 1]
			"--help", "-h":
				show_help = true
			"--quiet", "-q":
				quiet_mode = true
	
	if show_help:
		_show_help()
		get_tree().quit(0)
		return
	
	print("🔍 Godot Stat Math - Baseline Analysis CLI")
	print("==========================================")
	
	# Run the appropriate analysis
	var result: Dictionary = {}
	
	if not specific_test.is_empty():
		# Analyze specific test
		print("📊 Analyzing specific test: %s" % specific_test)
		result = PerfTestManager.analyze_baseline_test(specific_test, not quiet_mode)
		if result.is_empty():
			print("❌ Test analysis failed")
			get_tree().quit(1)
			return
	else:
		# Analyze all baseline tests
		print("📊 Analyzing all baseline tests...")
		result = PerfTestManager.analyze_all_baseline_tests(not quiet_mode)
		if result.is_empty():
			print("❌ Baseline analysis failed")
			get_tree().quit(1)
			return
	
	# Save to JSON file if requested
	if not save_json_file.is_empty():
		_save_analysis_to_json(result, save_json_file)
	
	print("\n✅ Analysis completed successfully!")
	get_tree().quit(0)

func _show_help():
	print("Godot Stat Math - Baseline Analysis CLI")
	print("======================================")
	print("")
	print("USAGE:")
	print("  godot --headless analyze_baseline_cli.tscn [OPTIONS]")
	print("")
	print("OPTIONS:")
	print("  --test-name <name>     Analyze a specific test (e.g., 'basic_stats_mean')")
	print("  --save-json <file>     Save analysis results to JSON file")
	print("  --quiet, -q           Suppress detailed output (only show summary)")
	print("  --help, -h            Show this help message")
	print("")
	print("EXAMPLES:")
	print("  # Analyze all baseline tests")
	print("  godot --headless analyze_baseline_cli.tscn")
	print("")
	print("  # Analyze specific test")
	print("  godot --headless analyze_baseline_cli.tscn --test-name basic_stats_mean")
	print("")
	print("  # Save results to JSON file")
	print("  godot --headless analyze_baseline_cli.tscn --save-json baseline_analysis.json")
	print("")
	print("  # Quiet mode with JSON output")
	print("  godot --headless analyze_baseline_cli.tscn --quiet --save-json results.json")

func _save_analysis_to_json(data: Dictionary, filepath: String):
	var file: FileAccess = FileAccess.open(filepath, FileAccess.WRITE)
	if file == null:
		print("❌ Failed to save JSON file: %s" % filepath)
		return
	
	# Add metadata to the output
	var output: Dictionary = {
		"analysis_data": data,
		"meta": {
			"generated_at": Time.get_datetime_string_from_system(),
			"generated_by": "analyze_baseline_cli.gd",
			"godot_version": Engine.get_version_info()
		}
	}
	
	file.store_string(JSON.stringify(output, "\t"))
	file.close()
	print("💾 Analysis saved to: %s" % filepath) 
