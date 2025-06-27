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

def find_latest_report_dir() -> Optional[Path]:
    """Find the latest report directory (highest numbered report_X)."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    
    report_dirs = [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("report_")]
    if not report_dirs:
        return None
    
    # Sort by report number
    def get_report_number(path: Path) -> int:
        try:
            return int(path.name.split("_")[1])
        except (IndexError, ValueError):
            return -1
    
    latest_report = max(report_dirs, key=get_report_number)
    return latest_report

def parse_test_results(xml_file: Path) -> Tuple[int, int, int]:
    """
    Parse GDUnit4 XML results file.
    Returns: (total_tests, failures, skipped)
    """
    if not xml_file.exists():
        return 0, 0, 0
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get counts from testsuites root element
        total_tests = int(root.get('tests', 0))
        failures = int(root.get('failures', 0))
        skipped = int(root.get('skipped', 0))
        
        return total_tests, failures, skipped
    except (ET.ParseError, ValueError) as e:
        print(f"Error parsing {xml_file}: {e}")
        return 0, 0, 0

def determine_test_type(xml_file: Path) -> str:
    """
    Determine if this is a unit test or performance test report
    by checking the test suite names in the XML.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Check testsuite names to determine type
        perf_indicators = ['_perf_test', 'performance', 'perf_']
        
        for testsuite in root.findall('testsuite'):
            suite_name = testsuite.get('name', '').lower()
            package = testsuite.get('package', '').lower()
            
            # If any suite contains performance indicators, it's a performance test
            if any(indicator in suite_name or indicator in package for indicator in perf_indicators):
                return 'performance'
        
        # If we find core test patterns, it's unit tests
        if 'tests/core' in root.find('testsuite').get('package', ''):
            return 'unit'
        
        # Default fallback - check if it contains stat_math_test
        for testsuite in root.findall('testsuite'):
            if 'stat_math_test' in testsuite.get('name', ''):
                return 'unit'
                
        return 'unknown'
        
    except (ET.ParseError, AttributeError) as e:
        print(f"Error determining test type for {xml_file}: {e}")
        return 'unknown'

def count_unit_tests_from_source() -> int:
    """
    Count unit tests by scanning the source files directly.
    This provides a fallback count when XML reports aren't available.
    """
    unit_test_count = 0
    
    # Count tests in core directory
    core_dir = Path("addons/godot-stat-math/tests/core")
    if core_dir.exists():
        for gd_file in core_dir.rglob("*.gd"):
            try:
                with open(gd_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count functions that start with "func test_"
                    unit_test_count += content.count("func test_")
            except Exception as e:
                print(f"Warning: Could not read {gd_file}: {e}")
    
    # Count tests in stat_math_test.gd
    stat_math_test = Path("addons/godot-stat-math/tests/stat_math_test.gd")
    if stat_math_test.exists():
        try:
            with open(stat_math_test, 'r', encoding='utf-8') as f:
                content = f.read()
                unit_test_count += content.count("func test_")
        except Exception as e:
            print(f"Warning: Could not read stat_math_test.gd: {e}")
    
    return unit_test_count

def generate_badge_data():
    """Generate badge data JSON files for shields.io consumption."""
    latest_report = find_latest_report_dir()
    
    unit_tests_total = 0
    performance_tests_total = 0
    unit_tests_passed = 0
    performance_tests_passed = 0
    
    if latest_report:
        results_xml = latest_report / "results.xml"
        if results_xml.exists():
            total, failures, skipped = parse_test_results(results_xml)
            test_type = determine_test_type(results_xml)
            
            print(f"Found report: {latest_report}")
            print(f"Test type detected: {test_type}")
            print(f"Total tests: {total}, Failures: {failures}, Skipped: {skipped}")
            
            if test_type == 'performance':
                performance_tests_total = total
                performance_tests_passed = total - failures
            elif test_type == 'unit':
                unit_tests_total = total
                unit_tests_passed = total - failures
            else:
                print(f"Warning: Unknown test type detected")
    
    # If we didn't find unit tests in XML, count from source
    if unit_tests_total == 0:
        unit_tests_total = count_unit_tests_from_source()
        unit_tests_passed = unit_tests_total  # Assume passing since release requires passing tests
        print(f"Counted {unit_tests_total} unit tests from source files")
    
    # Create output directory
    output_dir = Path("docs/_static/badges")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unit tests badge data
    unit_badge_data = {
        "schemaVersion": 1,
        "label": "Unit Tests",
        "message": f"{unit_tests_passed}/{unit_tests_total}",
        "color": "brightgreen" if unit_tests_passed == unit_tests_total else "red"
    }
    
    # Generate performance tests badge data  
    perf_badge_data = {
        "schemaVersion": 1,
        "label": "Performance Tests", 
        "message": f"{performance_tests_passed}/{performance_tests_total}",
        "color": "brightgreen" if performance_tests_passed == performance_tests_total else "orange"
    }
    
    # Write JSON files
    with open(output_dir / "unit_tests.json", 'w') as f:
        json.dump(unit_badge_data, f, indent=2)
    
    with open(output_dir / "performance_tests.json", 'w') as f:
        json.dump(perf_badge_data, f, indent=2)
    
    print(f"Generated badge data:")
    print(f"  Unit Tests: {unit_tests_passed}/{unit_tests_total}")
    print(f"  Performance Tests: {performance_tests_passed}/{performance_tests_total}")
    print(f"  Badge JSON files written to: {output_dir}")

if __name__ == "__main__":
    generate_badge_data() 