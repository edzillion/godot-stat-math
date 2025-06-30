#!/usr/bin/env python3
"""
Generate RST API documentation from GDScript docstrings.
Based on Godot's make_rst.py approach but for GDScript files.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class GDScriptFunction:
    def __init__(self, name: str, signature: str, description: str = "", is_static: bool = False):
        self.name = name
        self.signature = signature
        self.description = description
        self.is_static = is_static


class GDScriptClass:
    def __init__(self, name: str, filepath: str):
        self.name = name
        self.filepath = filepath
        self.description = ""
        self.brief_description = ""
        self.extends = ""
        self.functions: List[GDScriptFunction] = []
        self.constants: Dict[str, str] = {}


class GDScriptParser:
    def __init__(self):
        self.classes: Dict[str, GDScriptClass] = {}
    
    def parse_file(self, filepath: str) -> Optional[GDScriptClass]:
        """Parse a single GDScript file and extract documentation."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        lines = content.split('\n')
        
        # Extract class name and extends
        class_name = self._extract_class_name(lines)
        if not class_name:
            # Use filename as class name
            class_name = Path(filepath).stem.replace('_', ' ').title().replace(' ', '')
        
        gdclass = GDScriptClass(class_name, filepath)
        gdclass.extends = self._extract_extends(lines)
        
        # Extract class description
        gdclass.description, gdclass.brief_description = self._extract_class_description(lines)
        
        # Extract functions
        gdclass.functions = self._extract_functions(lines)
        
        # Extract constants
        gdclass.constants = self._extract_constants(lines)
        
        return gdclass
    
    def _extract_class_name(self, lines: List[str]) -> str:
        """Extract class_name from the file."""
        for line in lines:
            line = line.strip()
            if line.startswith('class_name '):
                # Extract class name from "class_name ClassName extends Parent"
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
        return ""
    
    def _extract_extends(self, lines: List[str]) -> str:
        """Extract what this class extends."""
        for line in lines:
            line = line.strip()
            if line.startswith('extends '):
                return line.replace('extends ', '').strip()
            elif 'extends ' in line and line.startswith('class_name'):
                # Handle "class_name MyClass extends Parent"
                parts = line.split('extends')
                if len(parts) > 1:
                    return parts[1].strip()
        return ""
    
    def _extract_class_description(self, lines: List[str]) -> Tuple[str, str]:
        """Extract class-level documentation."""
        description = ""
        brief_description = ""
        
        in_class_doc = False
        current_doc = []
        found_class_name = False
        found_extends = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Mark when we find the class_name line
            if stripped.startswith('class_name '):
                found_class_name = True
                continue
            
            # Also mark when we find extends (for files like stat_math.gd)
            if stripped.startswith('extends '):
                found_extends = True
                continue
            
            # For files with explicit class_name, look for docs after class_name line
            if found_class_name and stripped.startswith('##'):
                # Check if this is really class documentation (not function documentation)
                if not self._is_function_doc(lines, i):
                    if not in_class_doc:
                        in_class_doc = True
                        current_doc = []
                    
                    doc_line = stripped[2:].strip()
                    if doc_line:
                        current_doc.append(doc_line)
                else:
                    # This documentation belongs to a function, stop collecting class docs
                    break
            # For files with extends (like stat_math.gd), look for docs after extends line
            elif found_extends and stripped.startswith('##'):
                # Check if this is really class documentation (not function documentation)
                if not self._is_function_doc(lines, i):
                    if not in_class_doc:
                        in_class_doc = True
                        current_doc = []
                    
                    doc_line = stripped[2:].strip()
                    if doc_line:
                        current_doc.append(doc_line)
                else:
                    # This documentation belongs to a function, stop collecting class docs
                    break
            # For files without explicit class_name or extends, look at the top of the file (with generous limit)
            elif not found_class_name and not found_extends and i < 100 and stripped.startswith('##'):
                # Only collect if this is clearly class documentation
                if not self._is_function_doc(lines, i):
                    if not in_class_doc:
                        in_class_doc = True
                        current_doc = []
                    
                    doc_line = stripped[2:].strip()
                    if doc_line:
                        current_doc.append(doc_line)
                else:
                    # This documentation belongs to a function, stop
                    break
            elif in_class_doc and (stripped.startswith('static func') or 
                                   stripped.startswith('func') or 
                                   stripped.startswith('const ') or
                                   stripped.startswith('var ') or
                                   stripped.startswith('enum ')):  # Stop at actual code, not section comments
                # End of class documentation when we hit actual code
                break
            elif in_class_doc and stripped and not stripped.startswith('#'):
                # End of class documentation when we hit non-comment code
                break
        
        if current_doc:
            description = '\n'.join(current_doc)
            # First sentence as brief description
            sentences = description.split('.')
            if len(sentences) > 1:
                brief_description = sentences[0] + '.'
            else:
                brief_description = description
        
        return description, brief_description
    
    def _is_function_doc(self, lines: List[str], doc_line_index: int) -> bool:
        """Check if this documentation comment belongs to a function."""
        # Look ahead to see if there's a function definition coming up
        for i in range(doc_line_index + 1, min(len(lines), doc_line_index + 5)):
            line = lines[i].strip()
            if line.startswith('static func') or line.startswith('func'):
                return True
            elif line.startswith('##'):
                # More documentation, keep looking
                continue
            elif line.startswith('#') or line == '':
                # Comment or empty line, keep looking
                continue
            else:
                # Found code that's not a function, this isn't function documentation
                return False
        return False
    
    def _extract_functions(self, lines: List[str]) -> List[GDScriptFunction]:
        """Extract all functions and their documentation."""
        functions = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for function definitions first
            if line.startswith('static func') or line.startswith('func'):
                is_static = line.startswith('static func')
                
                # Extract function signature
                func_signature = line
                if '->' not in line and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if '->' in next_line or next_line.endswith(':'):
                        func_signature += ' ' + next_line.split(':')[0].strip()
                
                # Extract function name
                func_name = self._extract_function_name(func_signature)
                
                if func_name:
                    # Look backwards for documentation
                    func_doc = []
                    doc_end = i - 1
                    
                    # Skip empty lines and comments before function
                    while doc_end >= 0 and (lines[doc_end].strip() == '' or lines[doc_end].strip().startswith('#') and not lines[doc_end].strip().startswith('##')):
                        doc_end -= 1
                    
                    # Collect documentation lines (going backwards)
                    if doc_end >= 0 and lines[doc_end].strip().startswith('##'):
                        doc_lines = []
                        doc_start = doc_end
                        
                        # Find the start of the documentation block
                        while doc_start >= 0 and lines[doc_start].strip().startswith('##'):
                            doc_start -= 1
                        doc_start += 1  # Move to first ## line
                        
                        # Collect all documentation lines from start to end
                        for doc_line_idx in range(doc_start, doc_end + 1):
                            if lines[doc_line_idx].strip().startswith('##'):
                                doc_line = lines[doc_line_idx].strip()[2:].strip()
                                doc_lines.append(doc_line)
                        
                        func_doc = doc_lines
                    
                    description = '\n'.join(func_doc)
                    functions.append(GDScriptFunction(
                        func_name, func_signature, description, is_static
                    ))
                
                i += 1
            else:
                i += 1
        
        return functions
    
    def _extract_function_name(self, signature: str) -> str:
        """Extract function name from signature."""
        # Handle both "func name(" and "static func name("
        pattern = r'(?:static\s+)?func\s+(\w+)\s*\('
        match = re.search(pattern, signature)
        return match.group(1) if match else ""
    
    def _extract_constants(self, lines: List[str]) -> Dict[str, str]:
        """Extract constants and their values."""
        constants = {}
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('const '):
                # Extract constant name and value
                const_match = re.match(r'const\s+(\w+)\s*[:=]\s*(.+)', stripped)
                if const_match:
                    name, value = const_match.groups()
                    constants[name] = value.strip()
        
        return constants


class RSTGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_class_rst(self, gdclass: GDScriptClass) -> None:
        """Generate RST file for a single class."""
        # Generate filename
        filename = gdclass.name.lower().replace('stats', '_stats') + '.rst'
        if gdclass.name == 'BasicStats':
            filename = 'basic_stats.rst'
        elif gdclass.name == 'CdfFunctions':
            filename = 'cdf_functions.rst'
        elif gdclass.name == 'PpfFunctions':
            filename = 'ppf_functions.rst'
        elif gdclass.name == 'PmfPdfFunctions':
            filename = 'pmf_pdf_functions.rst'
        elif gdclass.name == 'ErrorFunctions':
            filename = 'error_functions.rst'
        elif gdclass.name == 'HelperFunctions':
            filename = 'helper_functions.rst'
        elif gdclass.name == 'Distributions':
            filename = 'distributions.rst'
        elif gdclass.name == 'SamplingGen':
            filename = 'sampling_gen.rst'
        elif gdclass.name == 'StatMath' or 'stat_math' in gdclass.filepath.lower():
            filename = 'stat_math.rst'
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            self._write_class_rst(f, gdclass)
        
        print(f"Generated: {output_file}")
    
    def _write_class_rst(self, f, gdclass: GDScriptClass) -> None:
        """Write RST content for a class."""
        # Title - handle main StatMath class specially
        if gdclass.name == 'StatMath' or 'stat_math' in gdclass.filepath.lower():
            title = "StatMath"
        else:
            title = f"StatMath.{gdclass.name}"
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        # Brief description
        if gdclass.brief_description:
            processed_brief = self._process_description(gdclass.brief_description)
            f.write(f"{processed_brief}\n\n")
        
        # Full description (only if different from brief)
        if gdclass.description and gdclass.description != gdclass.brief_description:
            # Remove the brief description from the full description to avoid repetition
            full_desc = gdclass.description
            if full_desc.startswith(gdclass.brief_description):
                remaining = full_desc[len(gdclass.brief_description):].strip()
                if remaining:
                    processed_remaining = self._process_description(remaining)
                    f.write(f"{processed_remaining}\n\n")
            else:
                processed_desc = self._process_description(gdclass.description)
                f.write(f"{processed_desc}\n\n")
        
        # Usage
        f.write("Usage\n")
        f.write("-----\n\n")
        f.write(".. code-block:: gdscript\n\n")
        if gdclass.name == 'StatMath' or 'stat_math' in gdclass.filepath.lower():
            f.write(f"   # StatMath is the main singleton - access modules through it\n")
            f.write(f"   var result = StatMath.ModuleName.function_name(parameters)\n")
            f.write(f"   \n")
            f.write(f"   # Or access constants directly\n")
            f.write(f"   var epsilon = StatMath.EPSILON\n\n")
        else:
            f.write(f"   # Access via StatMath singleton\n")
            f.write(f"   var result = StatMath.{gdclass.name}.function_name(parameters)\n\n")
        
        # Constants (if any)
        if gdclass.constants:
            f.write("Constants\n")
            f.write("---------\n\n")
            for name, value in gdclass.constants.items():
                f.write(f".. data:: {name}\n\n")
                f.write(f"   Value: ``{value}``\n\n")
        
        # Functions
        if gdclass.functions:
            f.write("Functions\n")
            f.write("---------\n\n")
            
            for func in gdclass.functions:
                self._write_function_rst(f, func)
    
    def _write_function_rst(self, f, func: GDScriptFunction) -> None:
        """Write RST for a single function."""
        # Function signature as heading
        clean_sig = self._clean_signature(func.signature)
        f.write(f".. function:: {clean_sig}\n\n")
        
        # Function description
        if func.description:
            # Process description to handle special formatting
            description = self._process_description(func.description)
            for line in description.split('\n'):
                if line.strip():
                    f.write(f"   {line}\n")
                else:
                    f.write("\n")
            f.write("\n")
            
    def _extract_function_name(self, signature: str) -> str:
        """Extract function name from signature."""
        # Remove 'static ', 'func ', and extract the function name
        clean_sig = signature.replace('static func ', '').replace('func ', '')
        # Extract function name before the opening parenthesis
        if '(' in clean_sig:
            return clean_sig.split('(')[0].strip()
        return clean_sig.strip()
        

    
    def _clean_signature(self, signature: str) -> str:
        """Clean up function signature for RST."""
        # Remove 'static ' prefix and clean up
        sig = signature.replace('static func ', '').replace('func ', '')
        # Remove extra whitespace
        sig = ' '.join(sig.split())
        return sig
    
    def _process_description(self, description: str) -> str:
        """Process description text to handle GDScript-specific formatting."""
        # Convert [code] tags to RST code
        description = re.sub(r'\[code\](.*?)\[/code\]', r'``\1``', description)
        
        # Handle bullet lists by ensuring proper spacing
        # Look for lines that start with * or - and ensure they have a blank line above them
        lines = description.split('\n')
        processed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            # If this line starts a bullet list and previous line isn't empty
            if stripped.startswith('* ') or stripped.startswith('- '):
                # Check if previous line exists and isn't empty
                if i > 0 and processed_lines and processed_lines[-1].strip():
                    processed_lines.append('')  # Add blank line before bullet list
            processed_lines.append(line)
        
        description = '\n'.join(processed_lines)
        
        # Convert new Godot-style cross-references [method ClassName.method_name]
        def convert_method_ref(match):
            class_name = match.group(1)
            method_name = match.group(2)
            
            # Map class names to their RST document names
            class_to_filename = {
                'BasicStats': 'basic_stats',
                'CdfFunctions': 'cdf_functions', 
                'PpfFunctions': 'ppf_functions',
                'PmfPdfFunctions': 'pmf_pdf_functions',
                'ErrorFunctions': 'error_functions',
                'HelperFunctions': 'helper_functions',
                'Distributions': 'distributions',
                'SamplingGen': 'sampling_gen',
                'StatMath': 'stat_math'
            }
            
            filename = class_to_filename.get(class_name, class_name.lower())
            # Use relative path within the same modules directory
            return f'`{method_name}() <{filename}.html#{method_name}>`_'
        
        description = re.sub(r'\[method (\w+)\.(\w+)\]', convert_method_ref, description)
        
        # Convert old StatMath-style cross-references to functions with parentheses (for backward compatibility)
        description = re.sub(
            r'\[StatMath\.(\w+)\.(\w+)\(\)\]', 
            r':func:`\2() <\1.html#\2>`', 
            description
        )
        
        # Convert old StatMath-style cross-references to functions without parentheses (for backward compatibility)
        description = re.sub(
            r'\[StatMath\.(\w+)\.(\w+)\]', 
            r':func:`\2() <\1.html#\2>`', 
            description
        )
        
        # Convert simple module references like [StatMath.HelperFunctions]
        description = re.sub(
            r'\[StatMath\.(\w+)\]', 
            r':doc:`StatMath.\1 <\1>`', 
            description
        )
        
        return description


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python generate_api_rst.py <gdscript_source_dir> <output_dir>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist")
        sys.exit(1)
    
    parser = GDScriptParser()
    generator = RSTGenerator(output_dir)
    
    # Parse all .gd files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.gd'):
                filepath = os.path.join(root, file)
                print(f"Parsing: {filepath}")
                
                gdclass = parser.parse_file(filepath)
                if gdclass:
                    generator.generate_class_rst(gdclass)
    
    # Also parse the main stat_math.gd file
    main_file = os.path.join(os.path.dirname(source_dir), "stat_math.gd")
    if os.path.exists(main_file):
        print(f"Parsing main file: {main_file}")
        gdclass = parser.parse_file(main_file)
        if gdclass:
            generator.generate_class_rst(gdclass)
    
    print(f"\nRST generation complete! Files written to: {output_dir}")


if __name__ == "__main__":
    main() 