#!/usr/bin/env python3
"""
Extract test counts from GDUnit4 XML reports for badge generation.
Generates JSON files that shields.io can consume for dynamic badges.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

def find_test_reports() -> Tuple[Optional[Path], Optional[Path]]:
    """Find unit test and performance test XML reports.
    Returns: (unit_xml_path, performance_xml_path)
    
    GDScript tests using gdUnit4-action save XML files, but the exact location may vary.
    This function searches in multiple possible locations.
    """
    print("Looking for XML reports in multiple locations...")
    
    # Possible locations to search
    search_locations = [
        Path("../unit-tests.xml"),           # Parent directory (original expectation)
        Path("./unit-tests.xml"),            # Current directory
        Path("unit-tests.xml"),              # Current directory (alternative)
        Path("../../unit-tests.xml"),        # Grandparent directory
        Path("reports/unit-tests.xml"),      # Reports subdirectory
    ]
    
    perf_locations = [
        Path("../performance-tests.xml"),    # Parent directory (original expectation)
        Path("./performance-tests.xml"),     # Current directory
        Path("performance-tests.xml"),       # Current directory (alternative)
        Path("../../performance-tests.xml"), # Grandparent directory
        Path("reports/performance-tests.xml"), # Reports subdirectory
    ]
    
    # Search for unit tests XML
    unit_xml = None
    for location in search_locations:
        if location.exists():
            print(f"✅ Found unit tests XML: {location.absolute()}")
            unit_xml = location
            break
        else:
            print(f"❌ Unit tests XML not found: {location.absolute()}")
    
    # Search for performance tests XML
    perf_xml = None
    for location in perf_locations:
        if location.exists():
            print(f"✅ Found performance tests XML: {location.absolute()}")
            perf_xml = location
            break
        else:
            print(f"❌ Performance tests XML not found: {location.absolute()}")
    
    # If still not found, do a broader search
    if unit_xml is None or perf_xml is None:
        print("\n🔍 Performing broader search for XML files...")
        import glob
        
        # Search for any XML files that might be test reports
        xml_files = []
        for pattern in ["*.xml", "../*.xml", "../../*.xml", "reports/*.xml"]:
            xml_files.extend(glob.glob(pattern))
        
        if xml_files:
            print("Found XML files:")
            for xml_file in xml_files:
                abs_path = Path(xml_file).absolute()
                size = abs_path.stat().st_size if abs_path.exists() else 0
                print(f"  - {abs_path} ({size} bytes)")
                
                # Check if it might be a test report by looking at the content
                try:
                    with open(xml_file, 'r', encoding='utf-8') as f:
                        content = f.read(200)
                        if any(keyword in content.lower() for keyword in ['testsuites', 'testsuite', 'testcase']):
                            print(f"    📋 This looks like a test report!")
                            if unit_xml is None and 'unit' in xml_file.lower():
                                unit_xml = Path(xml_file)
                            elif perf_xml is None and 'performance' in xml_file.lower():
                                perf_xml = Path(xml_file)
                except Exception as e:
                    print(f"    ❌ Could not read file: {e}")
        else:
            print("No XML files found in search locations.")
    
    # Print current working directory for debugging
    print(f"\nCurrent working directory: {Path.cwd()}")
    
    return unit_xml, perf_xml

def parse_test_results(xml_file: Path) -> Tuple[int, int, int]:
    """
    Parse GDUnit4 XML results file.
    Returns: (total_tests, failures, skipped)
    """
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get counts from testsuites root element
        total_tests = int(root.get('tests', 0))
        failures = int(root.get('failures', 0))
        skipped = int(root.get('skipped', 0))
        
        print(f"Parsed {xml_file.name}: {total_tests} tests, {failures} failures, {skipped} skipped")
        return total_tests, failures, skipped
    except (ET.ParseError, ValueError) as e:
        print(f"Error parsing {xml_file}: {e}")
        return 0, 0, 0

def generate_badge_data():
    """Generate badge data JSON files for shields.io consumption."""
    print("=== Starting Badge Data Generation ===")
    
    unit_xml, perf_xml = find_test_reports()
    
    # Initialize counts
    unit_tests_total = 0
    unit_tests_passed = 0
    performance_tests_total = 0
    performance_tests_passed = 0
    
    # Process unit tests
    if unit_xml:
        total, failures, skipped = parse_test_results(unit_xml)
        unit_tests_total = total
        unit_tests_passed = total - failures
        print(f"Unit tests from XML: {unit_tests_passed}/{unit_tests_total}")
    else:
        print("No unit tests XML found - using 0 count")
    
    # Process performance tests
    if perf_xml:
        total, failures, skipped = parse_test_results(perf_xml)
        performance_tests_total = total
        performance_tests_passed = total - failures
        print(f"Performance tests from XML: {performance_tests_passed}/{performance_tests_total}")
    else:
        print("No performance tests XML found - using 0 count")
    
    # Create output directory
    output_dir = Path("docs/_static/badges")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating badge data in: {output_dir}")
    
    # Generate unit tests badge data
    unit_badge_data = {
        "schemaVersion": 1,
        "label": "Unit Tests",
        "message": f"{unit_tests_passed}/{unit_tests_total}",
        "color": "brightgreen" if unit_tests_passed == unit_tests_total and unit_tests_total > 0 else "red"
    }
    
    # Generate performance tests badge data  
    perf_badge_data = {
        "schemaVersion": 1,
        "label": "Performance Tests", 
        "message": f"{performance_tests_passed}/{performance_tests_total}" if performance_tests_total > 0 else "0/0",
        "color": "brightgreen" if performance_tests_passed == performance_tests_total and performance_tests_total > 0 else "red"
    }
    
    # Write JSON files
    unit_file = output_dir / "unit_tests.json"
    perf_file = output_dir / "performance_tests.json"
    
    with open(unit_file, 'w') as f:
        json.dump(unit_badge_data, f, indent=2)
    
    with open(perf_file, 'w') as f:
        json.dump(perf_badge_data, f, indent=2)
    
    print(f"=== Badge Data Generated Successfully ===")
    print(f"Unit Tests: {unit_tests_passed}/{unit_tests_total} ({unit_badge_data['color']})")
    print(f"Performance Tests: {performance_tests_passed}/{performance_tests_total} ({perf_badge_data['color']})")
    print(f"Files written:")
    print(f"  - {unit_file}")
    print(f"  - {perf_file}")

if __name__ == "__main__":
    try:
        generate_badge_data()
        print("Badge data generation completed successfully")
    except Exception as e:
        print(f"Error during badge data generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 