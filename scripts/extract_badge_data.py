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
    
    Hardcoded paths - gdUnit4 creates reports in predictable locations:
    - reports/report_1/results.xml (unit tests, ~648 tests)
    - reports/report_2/results.xml (performance tests, ~114 tests)
    """
    print("=== Looking for test reports in known locations ===")
    
    unit_xml_path = Path("reports/report_1/results.xml")
    perf_xml_path = Path("reports/report_2/results.xml")
    
    unit_xml = unit_xml_path if unit_xml_path.exists() else None
    perf_xml = perf_xml_path if perf_xml_path.exists() else None
    
    # Report findings
    if unit_xml:
        print(f"✅ Unit tests XML: {unit_xml}")
    else:
        print(f"❌ Unit tests XML not found: {unit_xml_path}")
        
    if perf_xml:
        print(f"✅ Performance tests XML: {perf_xml}")  
    else:
        print(f"❌ Performance tests XML not found: {perf_xml_path}")
        
    return unit_xml, perf_xml

def parse_test_results(xml_file: Path) -> Tuple[int, int, int]:
    """
    Parse GDUnit4 XML results file.
    Returns: (total_tests, failures, skipped)
    """
    
    try:
        with open(xml_file, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)
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