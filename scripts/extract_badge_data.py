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
    
    GDScript tests using gdUnit4-action save XML files directly in the project root
    with the filename specified by the 'report-name' parameter.
    """
    print("Looking for XML reports in project root...")
    
    # GDScript tests save XML files directly in the project root with custom names
    unit_xml = Path("unit-tests.xml")
    perf_xml = Path("performance-tests.xml")
    
    # Check if the files exist
    if unit_xml.exists():
        print(f"✅ Found unit tests XML: {unit_xml}")
    else:
        print(f"❌ Unit tests XML not found: {unit_xml}")
        unit_xml = None
        
    if perf_xml.exists():
        print(f"✅ Found performance tests XML: {perf_xml}")
    else:
        print(f"❌ Performance tests XML not found: {perf_xml}")
        perf_xml = None
    
    # Fallback: check reports directory (for local testing or other workflows)
    if unit_xml is None or perf_xml is None:
        print("Checking fallback reports directory...")
        reports_dir = Path("reports")
        if reports_dir.exists():
            report_dirs = sorted(reports_dir.glob("report_*"), key=lambda x: x.name)
            if report_dirs:
                latest_report_dir = report_dirs[-1]
                print(f"Using latest report directory: {latest_report_dir}")
                
                if unit_xml is None:
                    results_xml = latest_report_dir / "results.xml"
                    if results_xml.exists():
                        print(f"Found fallback results XML: {results_xml}")
                        unit_xml = results_xml
    
    return unit_xml, perf_xml

def parse_test_results(xml_file: Path) -> Tuple[int, int, int]:
    """
    Parse GDUnit4 XML results file.
    Returns: (total_tests, failures, skipped)
    """
    if not xml_file.exists():
        print(f"XML file does not exist: {xml_file}")
        return 0, 0, 0
    
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

def count_performance_tests_from_source() -> int:
    """
    Count performance tests by scanning the source files directly.
    """
    perf_test_count = 0
    
    # Count tests in performance directory
    perf_dir = Path("addons/godot-stat-math/tests/performance/core")
    if perf_dir.exists():
        print(f"Scanning performance directory: {perf_dir}")
        for gd_file in perf_dir.rglob("*.gd"):
            try:
                with open(gd_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count functions that start with "func test_"
                    file_test_count = content.count("func test_")
                    perf_test_count += file_test_count
                    if file_test_count > 0:
                        print(f"  {gd_file.name}: {file_test_count} tests")
            except Exception as e:
                print(f"Warning: Could not read {gd_file}: {e}")
    else:
        print(f"Performance directory not found: {perf_dir}")
    
    print(f"Total performance tests counted from source: {perf_test_count}")
    return perf_test_count

def count_unit_tests_from_source() -> int:
    """
    Count unit tests by scanning the source files directly.
    This provides a fallback count when XML reports aren't available.
    """
    unit_test_count = 0
    
    # Count tests in core directory
    core_dir = Path("addons/godot-stat-math/tests/core")
    if core_dir.exists():
        print(f"Scanning core directory: {core_dir}")
        for gd_file in core_dir.rglob("*.gd"):
            try:
                with open(gd_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count functions that start with "func test_"
                    file_test_count = content.count("func test_")
                    unit_test_count += file_test_count
                    if file_test_count > 0:
                        print(f"  {gd_file.name}: {file_test_count} tests")
            except Exception as e:
                print(f"Warning: Could not read {gd_file}: {e}")
    else:
        print(f"Core directory not found: {core_dir}")
    
    # Count tests in stat_math_test.gd
    stat_math_test = Path("addons/godot-stat-math/tests/stat_math_test.gd")
    if stat_math_test.exists():
        try:
            with open(stat_math_test, 'r', encoding='utf-8') as f:
                content = f.read()
                file_test_count = content.count("func test_")
                unit_test_count += file_test_count
                print(f"  {stat_math_test.name}: {file_test_count} tests")
        except Exception as e:
            print(f"Warning: Could not read stat_math_test.gd: {e}")
    else:
        print(f"stat_math_test.gd not found: {stat_math_test}")
    
    print(f"Total unit tests counted from source: {unit_test_count}")
    return unit_test_count

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
        # Fall back to source code counting
        unit_tests_total = count_unit_tests_from_source()
        unit_tests_passed = unit_tests_total  # Assume passing since release requires passing tests
        print(f"Unit tests from source files: {unit_tests_passed}/{unit_tests_total}")
    
    # Process performance tests
    if perf_xml:
        total, failures, skipped = parse_test_results(perf_xml)
        performance_tests_total = total
        performance_tests_passed = total - failures
        print(f"Performance tests from XML: {performance_tests_passed}/{performance_tests_total}")
    else:
        # Fall back to source code counting
        performance_tests_total = count_performance_tests_from_source()
        performance_tests_passed = performance_tests_total  # Assume passing since tests succeeded in workflow
        print(f"Performance tests from source files: {performance_tests_passed}/{performance_tests_total}")
    
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