#!/usr/bin/env python3
"""
Convert Joe-Kuo Sobol data to GDScript format
Extracts first 250 dimensions from new-joe-kuo-6.21201 file
"""

def convert_joe_kuo_to_gdscript(input_file, output_file, max_dimensions=1024):
    """Convert Joe-Kuo data to GDScript format"""
    
    # Read and parse the data
    sobol_data = {}
    
    with open(input_file, 'r') as f:
        # Skip header line
        next(f)
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 4:
                continue
                
            try:
                d = int(parts[0])  # dimension
                s = int(parts[1])  # degree
                a = int(parts[2])  # primitive polynomial
                m_values = [int(x) for x in parts[3:]]  # direction numbers
                
                # Only take dimensions up to our limit
                if d <= max_dimensions:
                    sobol_data[d] = {
                        's': s,
                        'a': a, 
                        'm': m_values
                    }
                else:
                    break  # Stop processing once we hit our limit
                    
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: {line}")
                continue
    
    # Generate GDScript file
    with open(output_file, 'w') as f:
        f.write('# res://addons/godot-stat-math/tables/sobol_data.gd\n')
        f.write('class_name SobolData\n\n')
        f.write('# Joe-Kuo Sobol sequence direction numbers\n')
        f.write('# Source: https://web.maths.unsw.edu.au/~fkuo/sobol/\n')
        f.write(f'# Supports up to {max_dimensions} dimensions\n\n')
        
        # Create direction numbers array indexed by dimension
        f.write('# Direction numbers indexed by dimension [0=unused, 1=unused, 2=[1], 3=[1,3], ...]\n')
        f.write('const DIRECTION_NUMBERS: Array[Array] = [\n')
        
        # Dimensions 0 and 1 are special cases (not in Joe-Kuo data)
        f.write('\t[],\t\t# Dimension 0 (unused)\n')
        f.write('\t[],\t\t# Dimension 1 (unused)\n')
        
        # Add the actual data
        for d in range(2, max_dimensions + 1):
            if d in sobol_data:
                m_values = sobol_data[d]['m']
                m_str = ', '.join(map(str, m_values))
                f.write(f'\t[{m_str}],\t# Dimension {d}\n')
            else:
                f.write(f'\t[],\t\t# Dimension {d} (no data available)\n')
        
        f.write(']\n\n')
        
        # Also store the polynomial data for reference
        f.write('# Primitive polynomial data: {dimension: {"s": degree, "a": polynomial}}\n')
        f.write('const POLYNOMIAL_DATA: Dictionary = {\n')
        for d in sorted(sobol_data.keys()):
            s = sobol_data[d]['s']
            a = sobol_data[d]['a']
            f.write(f'\t{d}: {{"s": {s}, "a": {a}}},\n')
        f.write('}\n\n')
        
        # Utility functions
        f.write('## Get direction numbers for a specific dimension\n')
        f.write('static func get_direction_numbers(dimension: int) -> Array:\n')
        f.write('\tif dimension < 0 or dimension >= DIRECTION_NUMBERS.size():\n')
        f.write('\t\treturn []\n')
        f.write('\treturn DIRECTION_NUMBERS[dimension]\n\n')
        
        f.write('## Get maximum supported dimension\n')
        f.write('static func get_max_dimension() -> int:\n')
        f.write(f'\treturn {max_dimensions}\n\n')
        
        f.write('## Check if direction numbers are available for dimension\n')
        f.write('static func has_dimension(dimension: int) -> bool:\n')
        f.write('\tif dimension < 2 or dimension >= DIRECTION_NUMBERS.size():\n')
        f.write('\t\treturn false\n')
        f.write('\treturn not DIRECTION_NUMBERS[dimension].is_empty()\n')
    
    print(f"Converted {len(sobol_data)} dimensions to {output_file}")
    return len(sobol_data)

if __name__ == "__main__":
    input_file = "new-joe-kuo-6.21201"
    output_file = "sobol_data.gd"
    
    try:
        count = convert_joe_kuo_to_gdscript(input_file, output_file, 1024)
        print(f"Successfully converted {count} dimensions")
        print(f"Output written to: {output_file}")
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
    except Exception as e:
        print(f"Error: {e}") 