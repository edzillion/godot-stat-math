#!/usr/bin/env python3
"""
Debug script to find XML test reports created by gdunit4-action.
This helps identify where the action actually saves the XML files.
"""

import os
import glob
from pathlib import Path

def find_xml_files():
    """Search for XML files that might be test reports."""
    print("=== Searching for XML files that might be test reports ===")
    
    # Search patterns and locations
    search_locations = [
        ".",                    # Current directory
        "..",                   # Parent directory  
        "../..",                # Grandparent directory
        "reports",              # Reports folder
        "../reports",           # Parent reports folder
        ".",                    # All subdirectories from current
    ]
    
    xml_patterns = [
        "*.xml",
        "*test*.xml", 
        "*unit*.xml",
        "*performance*.xml",
        "*report*.xml",
        "*gdunit*.xml"
    ]
    
    found_files = []
    
    for location in search_locations:
        print(f"\n--- Searching in: {location} ---")
        
        for pattern in xml_patterns:
            search_path = os.path.join(location, pattern)
            matches = glob.glob(search_path)
            
            if matches:
                print(f"  Pattern '{pattern}' found:")
                for match in matches:
                    abs_path = os.path.abspath(match)
                    size = os.path.getsize(match) if os.path.exists(match) else 0
                    print(f"    ✅ {abs_path} ({size} bytes)")
                    found_files.append(abs_path)
            else:
                print(f"    ❌ Pattern '{pattern}' - no matches")
    
    # Also search recursively in current directory
    print(f"\n--- Recursive search from current directory ---")
    for xml_file in Path(".").rglob("*.xml"):
        abs_path = xml_file.absolute()
        if abs_path not in found_files:
            size = xml_file.stat().st_size
            print(f"    ✅ {abs_path} ({size} bytes)")
            found_files.append(str(abs_path))
    
    print(f"\n=== Summary ===")
    if found_files:
        print(f"Found {len(found_files)} XML files:")
        for file_path in found_files:
            print(f"  - {file_path}")
            
            # Try to peek at content to see if it looks like a test report
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(200)  # First 200 chars
                    if any(keyword in content.lower() for keyword in ['testsuites', 'testsuite', 'testcase', 'gdunit']):
                        print(f"    📋 Looks like a test report!")
                    else:
                        print(f"    📄 Content preview: {content[:50]}...")
            except Exception as e:
                print(f"    ❌ Could not read file: {e}")
    else:
        print("❌ No XML files found!")
    
    # Print current working directory for reference
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")

if __name__ == "__main__":
    find_xml_files() 