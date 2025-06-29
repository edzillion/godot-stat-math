"""
Test Results Extension for Sphinx
Embeds the latest GDUnit4 test report HTML into documentation
"""

import os
import shutil
import re
from pathlib import Path
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


class TestResultsDirective(SphinxDirective):
    """Directive to embed latest test report"""
    
    has_content = False
    optional_arguments = 0
    option_spec = {}

    def run(self):
        env = self.state.document.settings.env
        
        # Get the unit test report directory - hardcoded path
        source_dir = Path(env.srcdir)
        unit_test_report = source_dir.parent / 'reports' / 'report_1'
        
        if not unit_test_report.exists():
            # Return a message if no reports found
            warning_html = '''
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <strong>⚠️ No Unit Test Report Found</strong><br>
                Unit test report not found at reports/report_1/.
                Run tests to generate a report.
            </div>
            '''
            return [nodes.raw('', warning_html, format='html')]
        
        # Copy the report to the build directory and create an iframe
        self._copy_report_to_build(unit_test_report, env)
        
        # Generate iframe HTML to embed the report
        iframe_html = self._generate_iframe_html(unit_test_report.name)
        
        return [nodes.raw('', iframe_html, format='html')]
    

    
    def _copy_report_to_build(self, report_dir: Path, env):
        """Copy the report directory to the build output and modify for compact layout"""
        # Get the build directory
        build_dir = Path(env.app.outdir)
        
        # Create a reports directory in the build output
        build_reports_dir = build_dir / 'test_reports' 
        build_reports_dir.mkdir(exist_ok=True)
        
        # Copy the entire report directory
        dest_dir = build_reports_dir / report_dir.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(report_dir, dest_dir)
        
        # Modify the index.html to have a more compact header
        self._modify_report_for_compact_layout(dest_dir)
    
    def _modify_report_for_compact_layout(self, report_dir: Path):
        """Modify the report HTML files for a more compact layout"""
        # Find all HTML files to modify
        html_files = []
        
        # Add the main index file
        index_file = report_dir / 'index.html'
        if index_file.exists():
            html_files.append(index_file)
        
        # Add all test suite HTML files
        test_suites_dir = report_dir / 'test_suites'
        if test_suites_dir.exists():
            html_files.extend(test_suites_dir.glob('*.html'))
        
        # Add all path aggregation HTML files
        path_dir = report_dir / 'path'
        if path_dir.exists():
            html_files.extend(path_dir.glob('*.html'))
        
        # Apply compact CSS to all HTML files
        for html_file in html_files:
            self._apply_compact_css_to_file(html_file)
    
    def _apply_compact_css_to_file(self, html_file: Path):
        """Apply compact CSS to a single HTML file"""
        try:
            # Read the original HTML
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Skip if CSS already applied
            if 'Compact header styles for embedding' in html_content:
                return
            
            # Add custom CSS to make the layout more compact
            compact_css = '''
            <style>
            /* Compact header styles for embedding */
            header {
                padding: 5px 10px !important;
                min-height: auto !important;
                height: auto !important;
            }
            
            header .logo {
                font-size: 12px !important;
                margin-bottom: 0 !important;
                top: 5px !important;
                left: 10px !important;
            }
            
            header .logo img {
                width: 20px !important;
                height: 20px !important;
            }
            
            .report-container {
                margin: 0 10px !important;
                margin-top: 15px !important;
            }
            
            .report-container h1 {
                font-size: 18px !important;
                margin: 0 !important;
            }
            
            /* Force horizontal layout for summary */
            .summary {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                justify-content: space-between !important;
                align-items: center !important;
                margin: 0 !important;
                padding: 5px 10px !important;
                gap: 0 !important;
                width: 100% !important;
                max-width: none !important;
                box-sizing: border-box !important;
            }
            
            .summary-item {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                text-align: center !important;
                flex: 0 0 auto !important;
                max-width: 72px !important;
                min-width: 54px !important;
            }
            
            .summary-item .label {
                font-size: 13px !important;
                color: white !important;
                margin-bottom: 1px !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            
            .summary-item .value {
                font-size: 16px !important;
                font-weight: bold !important;
                color: lightgray !important;
                display: block !important;
                padding-top: 0 !important;
            }
            
            .success-rate {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
                text-align: center !important;
                flex: 0 0 auto !important;
                max-width: 84px !important;
                min-width: 72px !important;
                padding-left: 0 !important;
            }
            
            .success-rate .check-icon {
                font-size: 16px !important;
                width: 24px !important;
                height: 24px !important;
                margin-bottom: 1px !important;
            }
            
            .success-rate .rate-text {
                display: flex !important;
                flex-direction: column !important;
                align-items: center !important;
            }
            
            .success-rate .rate-text .label {
                font-size: 8px !important;
                color: white !important;
                margin-bottom: 1px !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
            }
            
            .success-rate .rate-text .value {
                font-size: 14px !important;
                font-weight: bold !important;
                color: lightgray !important;
            }
            
            main {
                margin-top: 0 !important;
                margin-left: 1em !important;
                margin-right: 1em !important;
                overflow-y: hidden !important;
            }
            
            /* Ensure content area takes up more space */
            #content {
                min-height: 600px !important;
                height: calc(100vh - 150px) !important;
            }
            
            /* Adjust navigation and breadcrumbs */
            nav, .breadcrumb {
                padding: 5px 0px !important;
            }
            
            /* Remove grid item padding */
            .grid-item {
                padding-left: 0px !important;
            }
            
            /* Add bottom padding to grid-item tbody for better scroller positioning */
            .grid-item tbody {
                padding-bottom: 10px !important;
            }
            
            /* Override footer paragraph padding */
            footer p {
                padding-left: 1em !important;
            }
            
            /* Compact table styling for test suite pages */
            #report-table {
                margin-top: 0 !important;
                table-layout: fixed !important;
                width: 100% !important;
            }
            
            #report-table th, #report-table td {
                padding: 4px 8px !important;
                font-size: 14px !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                white-space: nowrap !important;
            }
            
            /* Set specific widths for table columns to prevent overflow */
            #report-table th:first-child, #report-table td:first-child {
                width: 40% !important;
                max-width: 300px !important;
            }
            
            #report-table th:nth-child(2), #report-table td:nth-child(2) {
                width: 80px !important;
                min-width: 80px !important;
            }
            
            /* Compact breadcrumb styling */
            .breadcrumb {
                margin: 0 !important;
                padding: 5px 10px !important;
            }
            
            .breadcrumb a {
                font-size: 12px !important;
                padding: 2px 6px !important;
            }
            
            /* Report area styling for individual test pages */
            .tab-report-grid {
                margin-top: 10px !important;
                display: flex !important;
                flex-direction: column !important;
                gap: 15px !important;
            }
            
            .tab-report-grid .grid-item {
                width: 100% !important;
                flex: none !important;
            }
            
            #report_area {
                margin-top: 15px !important;
                order: 2 !important;
            }
            
            #report_area h4 {
                font-size: 16px !important;
                margin-top: 0 !important;
                margin-bottom: 10px !important;
            }
            </style>
            '''
            
            # Insert the CSS before the closing </head> tag
            html_content = html_content.replace('</head>', f'{compact_css}</head>')
            
            # Add tooltips to truncated table cells
            html_content = self._add_tooltips_to_truncated_cells(html_content)
            
            # Write the modified HTML back
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            # If modification fails, just use the original
            pass
    
    def _add_tooltips_to_truncated_cells(self, html_content: str) -> str:
        """Add title attributes (tooltips) to table cells that will be truncated"""
        try:
            # Simple and reliable approach: find all <td>content</td> patterns
            # and add title attributes to the first one in each row
            
            def add_title_if_long(match):
                full_td = match.group(0)
                td_content = match.group(1)
                
                # Skip if already has title attribute
                if 'title=' in full_td:
                    return full_td
                
                # Get plain text content (no HTML tags)
                text_content = re.sub(r'<[^>]*>', '', td_content).strip()
                
                # Add title if text is long (likely to be truncated)
                if len(text_content) > 50:
                    # Escape quotes for HTML
                    escaped_text = text_content.replace('"', '&quot;').replace("'", '&#39;')
                    # Insert title attribute into the opening tag
                    return full_td.replace('<td>', f'<td title="{escaped_text}">', 1)
                
                return full_td
            
            # Process each line separately to handle indentation properly
            lines = html_content.split('\n')
            result_lines = []
            in_table_row = False
            
            for line in lines:
                # Check if we're starting a new table row
                if '<tr' in line:
                    in_table_row = True
                elif '</tr>' in line:
                    in_table_row = False
                
                # If we're in a table row and this line has the first <td>
                if in_table_row and '<td>' in line and not any(x in line for x in ['<td><span', '<td><a', '<td><div']):
                    # This should be the first column (testcase name)
                    # Use the simplest regex pattern that works
                    line = re.sub(r'<td>(.*?)</td>', add_title_if_long, line, count=1)
                
                result_lines.append(line)
            
            return '\n'.join(result_lines)
            
        except Exception as e:
            # If tooltip addition fails, return original content
            print(f"Tooltip addition failed: {e}")  # Debug output
            return html_content
    
    def _generate_iframe_html(self, report_name: str):
        """Generate HTML for iframe to display the report"""
        iframe_html = f'''
        <style>
        .test-report-container iframe {{
            width: 100%; 
            height: 900px; 
            border: 1px solid #ddd; 
            border-radius: 5px;
        }}
        </style>
        <div class="test-report-container" style="width: 100%; margin: 20px 0;">
            <iframe 
                src="test_reports/{report_name}/index.html" 
                frameborder="0">
                <p>Your browser does not support iframes. 
                <a href="test_reports/{report_name}/index.html" target="_blank">
                View the test report in a new window</a>.</p>
            </iframe>
            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                <a href="test_reports/{report_name}/index.html" target="_blank">
                    📊 Open test report in new window
                </a>
            </p>
        </div>
        '''
        return iframe_html


def setup(app: Sphinx):
    """Setup function for the Sphinx extension"""
    app.add_directive('testresults', TestResultsDirective)
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 