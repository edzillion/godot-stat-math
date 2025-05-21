@tool
extends EditorPlugin

func _enter_tree():
	# Initialization of the plugin goes here.
	add_autoload_singleton("StatMath", "res://addons/godot-stat-math/stat_math.gd")

func _exit_tree():
	# Clean-up of the plugin goes here.
	remove_autoload_singleton("StatMath")
