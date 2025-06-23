
# Documentation collector script for GDScript
extends SceneTree

func _ready():
	print("Starting documentation collection...")
	
	var docs = {}
	
	# Scan all GDScript files in the addon
	var addon_path = "res://addons/godot-stat-math/"
	_collect_docs_recursive(addon_path, docs)
	
	# Save to JSON
	var file = FileAccess.open("res://docs_output.json", FileAccess.WRITE)
	file.store_string(JSON.stringify(docs, "\t"))
	file.close()
	
	print("Documentation collection complete")
	quit()

func _collect_docs_recursive(path: String, docs: Dictionary):
	var dir = DirAccess.open(path)
	if dir:
		dir.list_dir_begin()
		var file_name = dir.get_next()
		while file_name != "":
			var full_path = path + file_name
			if dir.current_is_dir() and file_name != "." and file_name != "..":
				_collect_docs_recursive(full_path + "/", docs)
			elif file_name.ends_with(".gd"):
				_extract_docs_from_file(full_path, docs)
			file_name = dir.get_next()

func _extract_docs_from_file(file_path: String, docs: Dictionary):
	var file = FileAccess.open(file_path, FileAccess.READ)
	if not file:
		return
		
	var content = file.get_as_text()
	file.close()
	
	var lines = content.split("\n")
	var current_docs = []
	var file_docs = {
		"path": file_path,
		"classes": [],
		"functions": [],
		"constants": [],
		"variables": []
	}
	
	# Basic parsing - this is simplified
	for i in range(lines.size()):
		var line = lines[i].strip_edges()
		
		# Collect documentation comments
		if line.begins_with("##"):
			current_docs.append(line.substr(2).strip_edges())
		elif line.begins_with("static func ") or line.begins_with("func "):
			# Extract function info
			var func_info = _parse_function(line, current_docs)
			if func_info:
				file_docs.functions.append(func_info)
			current_docs.clear()
		elif line.begins_with("class_name "):
			# Extract class info  
			var class_info = _parse_class(line, current_docs)
			if class_info:
				file_docs.classes.append(class_info)
			current_docs.clear()
		elif line.begins_with("const "):
			# Extract constant info
			var const_info = _parse_constant(line, current_docs)
			if const_info:
				file_docs.constants.append(const_info)
			current_docs.clear()
		elif not line.begins_with("#") and line.length() > 0:
			# Reset docs if we hit non-comment, non-empty line
			current_docs.clear()
	
	docs[file_path] = file_docs

func _parse_function(line: String, docs: Array) -> Dictionary:
	var parts = line.split("(")
	if parts.size() < 2:
		return {}
		
	var name_part = parts[0]
	var func_name = ""
	
	if name_part.begins_with("static func "):
		func_name = name_part.substr(12).strip_edges()
	elif name_part.begins_with("func "):
		func_name = name_part.substr(5).strip_edges()
	
	var params_part = parts[1].split(")")[0]
	
	return {
		"name": func_name,
		"parameters": params_part,
		"documentation": "\n".join(docs),
		"is_static": name_part.begins_with("static")
	}

func _parse_class(line: String, docs: Array) -> Dictionary:
	var parts = line.split(" ")
	if parts.size() < 2:
		return {}
		
	return {
		"name": parts[1],
		"documentation": "\n".join(docs)
	}

func _parse_constant(line: String, docs: Array) -> Dictionary:
	var parts = line.split(":")
	var name_part = parts[0].replace("const ", "").strip_edges()
	var type_part = ""
	var value_part = ""
	
	if parts.size() > 1:
		var type_and_value = parts[1].split("=")
		type_part = type_and_value[0].strip_edges()
		if type_and_value.size() > 1:
			value_part = type_and_value[1].strip_edges()
	
	return {
		"name": name_part,
		"type": type_part,
		"value": value_part,
		"documentation": "\n".join(docs)
	}
