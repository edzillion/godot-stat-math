#!/usr/bin/env python3
"""
Script to automatically update GDScript test error messages from old format to new format.

OLD: await assert_error(test_call).is_push_error("Error message.")
NEW: await assert_error(test_call).is_push_error("Error message. Received: X")

This script analyzes the test functions to extract the actual test values and 
automatically generates the correct "Received: X" messages.
"""

import re
import os
from typing import Dict, List, Tuple

# Test files that need updating
TEST_FILES = [
    "addons/godot-stat-math/tests/core/distributions_test.gd",
    "addons/godot-stat-math/tests/core/cdf_functions_test.gd",
]

def extract_test_value_from_function(func_content: str, error_line: str) -> str:
    """Extract the test value that would be in the 'Received: X' message."""
    
    # Common patterns to find test values
    patterns = [
        r'\.randi_\w+\(([-+]?\d*\.?\d+)',  # First parameter in randi_* functions
        r'\.randf_\w+\(([-+]?\d*\.?\d+)',  # First parameter in randf_* functions  
        r'\.(\w+)\(([-+]?\d*\.?\d+)',      # First parameter in general functions
        r'\.(\w+)\([^,]+,\s*([-+]?\d*\.?\d+)', # Second parameter 
        r'\.(\w+)\([^,]+,\s*[^,]+,\s*([-+]?\d*\.?\d+)', # Third parameter
    ]
    
    # Try to find the test value
    for pattern in patterns:
        matches = re.findall(pattern, func_content)
        if matches:
            if isinstance(matches[0], tuple):
                return matches[0][-1]  # Get the last element (the value)
            return matches[0]
    
    # Fallback: look for obvious test values
    common_test_values = ['-1.1', '-1.0', '-0.1', '0.0', '1.1', '2.1', '-1', '0']
    for value in common_test_values:
        if value in func_content:
            return value
    
    return "UNKNOWN"

def find_parameter_patterns(func_content: str, param_names: List[str]) -> Dict[str, str]:
    """Find parameter values for common parameter name patterns."""
    params = {}
    
    for param_name in param_names:
        # Look for parameter assignments or direct values
        patterns = [
            rf'{param_name}\s*[=:]\s*([-+]?\d*\.?\d+)',
            rf'\..*\(\s*([-+]?\d*\.?\d+).*{param_name}',
            rf'\..*\({param_name}\s*=\s*([-+]?\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, func_content, re.IGNORECASE)
            if match:
                params[param_name] = match.group(1)
                break
    
    return params

def analyze_and_fix_error_messages(file_path: str) -> List[Tuple[str, str]]:
    """Analyze a test file and return the old->new message replacements needed."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    replacements = []
    
    # Find all error message patterns that need updating (ones without "Received:")
    error_pattern = r'await assert_error\([^)]+\)\.is_push_error\("([^"]+)"\)'
    matches = re.finditer(error_pattern, content)
    
    for match in matches:
        error_msg = match.group(1)
        
        # Skip if already has "Received:" 
        if "Received:" in error_msg:
            continue
            
        # Find the function this error is in
        func_start = content.rfind('func ', 0, match.start())
        func_end = content.find('\nfunc ', func_start + 1)
        if func_end == -1:
            func_end = len(content)
        
        func_content = content[func_start:func_end]
        
        # Determine the expected received value based on error message type
        received_value = determine_received_value(error_msg, func_content)
        
        old_message = f'await assert_error({extract_test_call_var(func_content)}).is_push_error("{error_msg}")'
        new_message = f'await assert_error({extract_test_call_var(func_content)}).is_push_error("{error_msg} Received: {received_value}")'
        
        replacements.append((old_message, new_message))
    
    return replacements

def extract_test_call_var(func_content: str) -> str:
    """Extract the test call variable name (usually 'test_call' or 'test_invalid_input')."""
    patterns = [
        r'await assert_error\((\w+)\)\.is_push_error',
        r'var (\w+): Callable'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, func_content)
        if match:
            return match.group(1)
    
    return "test_call"  # Default fallback

def determine_received_value(error_msg: str, func_content: str) -> str:
    """Determine what the 'Received: X' value should be based on error message and function content."""
    
    # Extract function call to get the actual test values
    func_calls = re.findall(r'StatMath\.\w+\.\w+\([^)]+\)', func_content)
    
    if not func_calls:
        return "UNKNOWN"
    
    # Get the first function call (usually the one being tested)
    call = func_calls[0]
    
    # Extract arguments from the function call
    args_match = re.search(r'\(([^)]+)\)', call)
    if not args_match:
        return "UNKNOWN"
    
    args = [arg.strip() for arg in args_match.group(1).split(',')]
    
    # Determine which argument is being tested based on error message
    if any(phrase in error_msg.lower() for phrase in ['probability', 'p_prob', 'success probability']):
        # Usually the first argument for probability functions
        return args[0] if args else "UNKNOWN"
    elif 'trials' in error_msg.lower() or 'n_trials' in error_msg.lower():
        # Usually the second argument for trial-based functions
        return args[1] if len(args) > 1 else "UNKNOWN"
    elif 'lambda' in error_msg.lower() or 'rate' in error_msg.lower():
        # Usually the first argument for rate/lambda parameters
        return args[0] if args else "UNKNOWN"
    elif 'shape' in error_msg.lower() or 'scale' in error_msg.lower():
        # Shape/scale parameters vary by position
        return args[0] if args else "UNKNOWN"
    else:
        # Default to first argument
        return args[0] if args else "UNKNOWN"

def apply_replacements(file_path: str, replacements: List[Tuple[str, str]]) -> None:
    """Apply the replacements to the file."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Specific mappings for known test patterns
KNOWN_TEST_VALUES = {
    # Distributions test file
    "Success probability (p) must be between 0.0 and 1.0.": {
        "randi_binomial(1.1": "1.1",
        "randi_bernoulli(-0.1": "-0.1", 
        "randi_bernoulli(1.1": "1.1",
        "randi_binomial(-0.1": "-0.1"
    },
    "Number of trials (n) must be non-negative.": {
        "randi_binomial(0.5, -1": "-1"
    },
    "Success probability (p) must be in (0,1].": {
        "randi_geometric(0.0": "0.0",
        "randi_geometric(-0.1": "-0.1", 
        "randi_geometric(1.1": "1.1"
    },
    "Rate parameter (lambda_param) must be positive.": {
        "randi_poisson(0.0": "0.0",
        "randi_poisson(-1.0": "-1.0"
    }
}

def smart_fix_file(file_path: str) -> None:
    """Smart fix using known patterns and heuristics."""
    
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into functions for analysis
    functions = re.split(r'\nfunc ', content)
    
    total_fixes = 0
    
    for i, func in enumerate(functions):
        if i == 0:
            continue  # Skip the header before first function
            
        func = 'func ' + func  # Add back the 'func' keyword
        
        # Find error messages without "Received:"
        error_matches = re.finditer(r'await assert_error\([^)]+\)\.is_push_error\("([^"]+)"\)', func)
        
        for match in error_matches:
            error_msg = match.group(1)
            
            if "Received:" in error_msg:
                continue  # Already fixed
                
            # Find the function call being tested
            call_matches = re.findall(r'StatMath\.\w+\.\w+\([^)]+\)', func)
            if not call_matches:
                continue
                
            call = call_matches[0]
            
            # Extract the test value
            received_value = extract_received_value_smart(error_msg, call)
            
            if received_value != "UNKNOWN":
                old_line = match.group(0)
                new_line = old_line.replace(f'"{error_msg}"', f'"{error_msg} Received: {received_value}"')
                
                content = content.replace(old_line, new_line)
                total_fixes += 1
                print(f"  Fixed: {error_msg} -> added 'Received: {received_value}'")
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  Total fixes applied: {total_fixes}")

def extract_received_value_smart(error_msg: str, function_call: str) -> str:
    """Smart extraction of received value based on error message and function call."""
    
    # Extract all numeric arguments from function call
    args = re.findall(r'([-+]?\d*\.?\d+)', function_call)
    
    if not args:
        return "UNKNOWN"
    
    # Determine which argument based on error message keywords
    if any(keyword in error_msg.lower() for keyword in ['probability', 'p_prob', 'success probability']):
        return args[0] if args else "UNKNOWN"
    elif any(keyword in error_msg.lower() for keyword in ['trials', 'n_trials', 'number of trials']):
        return args[1] if len(args) > 1 else args[0]
    elif any(keyword in error_msg.lower() for keyword in ['lambda', 'rate parameter']):
        return args[0] if args else "UNKNOWN"  
    elif any(keyword in error_msg.lower() for keyword in ['shape parameter', 'scale parameter']):
        return args[0] if args else "UNKNOWN"
    elif any(keyword in error_msg.lower() for keyword in ['parameter a', 'parameter n', 'parameter r']):
        return args[0] if args else "UNKNOWN"
    elif any(keyword in error_msg.lower() for keyword in ['parameter b', 'parameter k']):
        return args[1] if len(args) > 1 else args[0]
    else:
        # Default to first argument
        return args[0] if args else "UNKNOWN"

if __name__ == "__main__":
    print("🔧 Fixing GDScript test error messages...")
    print("=" * 50)
    
    for file_path in TEST_FILES:
        if os.path.exists(file_path):
            smart_fix_file(file_path)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print("=" * 50)
    print("✅ All files processed!")
    print("\n💡 Next steps:")
    print("1. Review the changes made")
    print("2. Run tests to verify fixes") 
    print("3. Commit the changes") 