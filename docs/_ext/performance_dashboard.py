"""
Performance Dashboard Extension for Sphinx
Generates a performance dashboard page from JSON test results
"""

import json
import os
import re
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective


class PerformanceDashboardDirective(SphinxDirective):
    """Directive to generate performance dashboard"""
    
    has_content = False
    optional_arguments = 1
    option_spec = {
        'results_path': str,
        'chart_points': int,
    }

    def run(self):
        print("DEBUG: Performance dashboard directive run() called")
        env = self.state.document.settings.env
        
        # Get results path from option or default
        results_path = self.options.get('results_path', 
            '../addons/godot-stat-math/tests/performance/results')
        
        # Get number of chart points to show (default 5)
        chart_points = int(self.options.get('chart_points', 5))
        
        # Resolve relative to source directory
        source_dir = Path(env.srcdir)
        results_dir = source_dir / results_path
        
        print(f"DEBUG: Looking for performance data in: {results_dir}")
        
        # Load performance data
        performance_data = self._load_performance_data(results_dir, chart_points)
        
        print(f"DEBUG: Loaded {len(performance_data.get('tests', []))} tests")
        
        # Create HTML node
        dashboard_html = self._generate_dashboard_html(performance_data)
        print(f"DEBUG: Generated HTML length: {len(dashboard_html)}")
        raw_node = nodes.raw('', dashboard_html, format='html')
        
        return [raw_node]
    
    def _load_historical_data(self, results_dir: Path, max_points: int = 10):
        """Load historical performance data from pass/fail files"""
        historical_data = {}
        
        # Find all pass_* and fail_* files, sort by timestamp
        pattern = re.compile(r'(pass|fail)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json')
        files = []
        
        for file_path in results_dir.glob('*.json'):
            match = pattern.match(file_path.name)
            if match:
                status, timestamp = match.groups()
                files.append((file_path, status, timestamp))
        
        # Sort by timestamp (newest first, we'll reverse later for chart)
        files.sort(key=lambda x: x[2], reverse=True)
        
        # Take only the most recent max_points files
        files = files[:max_points]
        
        # Load data from each file
        for file_path, status, timestamp in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    tests = data.get('tests', {})
                    
                    for test_name, test_data in tests.items():
                        if test_name not in historical_data:
                            historical_data[test_name] = []
                        
                        historical_data[test_name].append({
                            'timestamp': timestamp,
                            'result_ms': test_data.get('result_ms', 0),
                            'baseline_ms': test_data.get('baseline_ms', 0),
                            'status': status,
                            'diff_percent': test_data.get('diff_percent', 0)
                        })
            except Exception as e:
                continue
        
        # Reverse to get oldest first (for chart display)
        for test_name in historical_data:
            historical_data[test_name].reverse()
        
        return historical_data
    
    def _load_performance_data(self, results_dir: Path, chart_points: int = 10):
        """Load performance data from JSON files"""
        try:
            latest_file = results_dir / 'latest.json'
            baseline_file = results_dir / 'baseline.json'
            
            latest_data = {}
            baseline_data = {}
            
            if latest_file.exists():
                with open(latest_file, 'r') as f:
                    latest_data = json.load(f)
            
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline_data = json.load(f)
            
            # Load historical data for charts
            historical_data = self._load_historical_data(results_dir, chart_points)
            
            # Combine data for dashboard
            combined_data = []
            tests = latest_data.get('tests', {})
            baseline_tests = baseline_data.get('tests', {})
            
            for test_name, test_data in tests.items():
                baseline_info = baseline_tests.get(test_name, {})
                
                combined_entry = {
                    'name': test_name,
                    'current_ms': test_data.get('result_ms', 0),
                    'baseline_ms': baseline_info.get('baseline_ms', test_data.get('baseline_ms', 0)),
                    'diff_percent': test_data.get('diff_percent', 0),
                    'threshold_percent': baseline_info.get('threshold_percent', 0.2),
                    'status': test_data.get('status', 'unknown'),
                    'coefficient_of_variation': baseline_info.get('coefficient_of_variation', 0),
                    'historical': historical_data.get(test_name, [])
                }
                combined_data.append(combined_entry)
            
            # Sort by current performance (slowest first)
            combined_data.sort(key=lambda x: x['current_ms'], reverse=True)
            
            return {
                'tests': combined_data,
                'meta': latest_data.get('meta', {}),
                'baseline_meta': baseline_data.get('meta', {})
            }
            
        except Exception as e:
            # Use env logger or print for debugging
            if hasattr(self.state.document.settings.env, 'app'):
                self.state.document.settings.env.app.warn(f"Could not load performance data: {e}")
            return {'tests': [], 'meta': {}, 'baseline_meta': {}}
    
    def _group_tests_by_module(self, tests):
        """Group tests by module name"""
        modules = {}
        for test in tests:
            # Extract module name (first two parts before underscore, e.g., "ppf_functions")
            parts = test['name'].split('_')
            if len(parts) >= 2:
                module_name = f"{parts[0]}_{parts[1]}"
            elif len(parts) == 1:
                module_name = parts[0]
            else:
                module_name = 'other'
            
            if module_name not in modules:
                modules[module_name] = []
            modules[module_name].append(test)
        return modules
    
    def _calculate_module_stats(self, module_tests):
        """Calculate statistics for a module"""
        total = len(module_tests)
        passed = sum(1 for test in module_tests if 'pass' in test['status'].lower())
        failed = sum(1 for test in module_tests if 'fail' in test['status'].lower())
        disabled = sum(1 for test in module_tests if 'disabled' in test['status'].lower())
        
        # Determine overall module status - prioritize pass/fail over disabled status
        if failed > 0:
            status = 'fail'
        elif passed > 0:
            status = 'pass'
        elif disabled == total:
            status = 'disabled'
        else:
            status = 'pass'  # Default to pass if unclear
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'disabled': disabled,
            'status': status
        }

    def _generate_dashboard_html(self, data):
        """Generate HTML for the performance dashboard"""
        tests = data['tests']
        meta = data['meta']
        
        # Group tests by module
        modules = self._group_tests_by_module(tests)
        
        html = f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        
        <div class="performance-dashboard">
            <div class="dashboard-header">
                <h2>📊 Performance Dashboard</h2>
                <div class="dashboard-meta">
                    <span class="meta-item">Generated: {meta.get('generated_at', 'Unknown')}</span>
                    <span class="meta-item">Total Tests: {len(tests)}</span>
                    <span class="meta-item">Passed: {meta.get('passed_tests', 0)}</span>
                    <span class="meta-item">Failed: {meta.get('failed_tests', 0)}</span>
                </div>
            </div>
            
            <div class="performance-legend">
                <div class="legend-item">
                    <span class="status-indicator status-pass"></span> Within threshold
                </div>
                <div class="legend-item">
                    <span class="status-indicator status-fail"></span> Above threshold
                </div>
                <div class="legend-item">
                    <span class="status-indicator status-disabled"></span> Disabled
                </div>
            </div>
            
            <div class="performance-modules">
        """
        
        chart_counter = 0
        for module_name, module_tests in modules.items():
            module_stats = self._calculate_module_stats(module_tests)
            module_status_class = self._get_status_class(module_stats['status'])
            
            # Format module name for display
            display_module_name = module_name.replace('_', ' ').title()
            
            html += f"""
                <div class="module-section">
                    <div class="module-header {module_status_class}" onclick="toggleModule('{module_name}')">
                        <div class="module-info">
                            <span class="module-name">{display_module_name}</span>
                            <span class="module-stats">
                                <span class="status-indicator {module_status_class}"></span>
                                ({module_stats['passed']}/{module_stats['total']} passed)
                            </span>
                        </div>
                        <div class="module-toggle">
                            <span class="arrow" id="arrow_{module_name}">▼</span>
                        </div>
                    </div>
                    
                    <div class="module-content" id="content_{module_name}">
                        <div class="performance-grid">
                            <div class="grid-header">
                                <div class="col-function">Function</div>
                                <div class="col-current">Current (ms)</div>
                                <div class="col-baseline">Baseline (ms)</div>
                                <div class="col-diff">Diff %</div>
                                <div class="col-threshold">Threshold</div>
                                <div class="col-status">Status</div>
                            </div>
            """
            
            for test in module_tests:
                status_class = self._get_status_class(test['status'])
                diff_class = 'positive' if test['diff_percent'] > 0 else 'negative' if test['diff_percent'] < 0 else 'neutral'
                
                # Format function name for display (remove module prefix)
                if test['name'].startswith(module_name + '_'):
                    function_name = test['name'][len(module_name)+1:]
                else:
                    function_name = test['name']
                display_name = function_name.replace('_', ' ').title()
                
                # Generate chart data
                chart_id = f"chart_{chart_counter}_{test['name'].replace(' ', '_').replace('-', '_')}"
                chart_data = self._generate_chart_data(test, chart_id)
                chart_counter += 1
                
                html += f"""
                            <div class="grid-row {status_class}">
                                <div class="col-function" title="{test['name']}">{display_name}</div>
                                <div class="col-current">{test['current_ms']:.3f}</div>
                                <div class="col-baseline">{test['baseline_ms']:.3f}</div>
                                <div class="col-diff {diff_class}">
                                    {test['diff_percent']:+.1f}%
                                </div>
                                <div class="col-threshold">±{test['threshold_percent']*100:.0f}%</div>
                                <div class="col-status">
                                    <span class="status-indicator {status_class}"></span>
                                    {test['status']}
                                </div>
                            </div>
                            <div class="chart-row {status_class}">
                                <div class="chart-container">
                                    <canvas id="{chart_id}"></canvas>
                                </div>
                            </div>
                            <script>
                            {chart_data}
                            </script>
                """
            
            html += """
                        </div>
                    </div>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <script>
        function toggleModule(moduleName) {
            const content = document.getElementById('content_' + moduleName);
            const arrow = document.getElementById('arrow_' + moduleName);
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                arrow.textContent = '▼';
            } else {
                content.style.display = 'none';
                arrow.textContent = '▶';
            }
        }
        
        // Initialize all modules as expanded
        document.addEventListener('DOMContentLoaded', function() {
            const moduleContents = document.querySelectorAll('.module-content');
            moduleContents.forEach(content => {
                content.style.display = 'block';
            });
        });
        </script>
        
        <style>
        .performance-dashboard {
            margin: 2em 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        .module-section {
            margin-bottom: 1.5em;
            border: 1px solid #e1e4e5;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .module-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1em;
            background: #f8f8f8;
            cursor: pointer;
            border-bottom: 1px solid #e1e4e5;
            transition: background-color 0.2s;
        }
        
        .module-header:hover {
            background: #f0f0f0;
        }
        
        .module-header.status-fail {
            background: #ffe6e6;
        }
        
        .module-header.status-disabled {
            background: #f5f5f5;
            opacity: 0.8;
        }
        
        .module-info {
            display: flex;
            align-items: center;
            gap: 1em;
        }
        
        .module-name {
            font-weight: bold;
            font-size: 1.1em;
            color: #2980b9;
        }
        
        .module-stats {
            display: flex;
            align-items: center;
            gap: 0.5em;
            font-size: 0.9em;
            color: #666;
        }
        
        .module-toggle .arrow {
            font-size: 0.8em;
            transition: transform 0.2s;
        }
        
        .module-content {
            display: block;
        }
        
        .dashboard-header h2 {
            margin-bottom: 0.5em;
            color: #2980b9;
            border-bottom: 2px solid #2980b9;
            padding-bottom: 0.3em;
        }
        
        .dashboard-meta {
            display: flex;
            gap: 1.5em;
            margin-bottom: 1.5em;
            font-size: 0.9em;
            color: #666;
        }
        
        .meta-item {
            background: #f8f8f8;
            padding: 0.3em 0.6em;
            border-radius: 4px;
            border: 1px solid #e1e4e5;
        }
        
        .performance-legend {
            display: flex;
            gap: 1.5em;
            margin-bottom: 1em;
            font-size: 0.9em;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5em;
        }
        
        .performance-grid {
            border: 1px solid #e1e4e5;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .grid-header {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 0.8fr 1fr 1.5fr;
            background: #f8f8f8;
            font-weight: bold;
            border-bottom: 2px solid #e1e4e5;
        }
        
        .grid-row {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 0.8fr 1fr 1.5fr;
            background: #f8f8f8;
            border-bottom: none;
            align-items: center;
        }
        
        .chart-row {
            width: 100%;
            border-bottom: 1px solid #e1e4e5;
            padding: 0.5em 0;
        }
        
        .chart-container {
            width: 100%;
            height: 80px;
            padding: 0.5em;
        }
        
        .chart-container canvas {
            width: 100% !important;
            height: 100% !important;
        }
        
        .grid-row:nth-child(even) {
            background: #f9f9f9;
        }
        
        .grid-row.status-fail {
            background: #ffe6e6;
        }
        
        .grid-row.status-disabled {
            opacity: 0.6;
        }
        
        .grid-header > div, .grid-row > div {
            padding: 0.8em;
            border-right: 1px solid #e1e4e5;
        }
        
        .grid-header > div:last-child, .grid-row > div:last-child {
            border-right: none;
        }
        
        .col-current, .col-baseline {
            font-family: 'Courier New', monospace;
            text-align: right;
        }
        
        .col-diff {
            text-align: right;
            font-weight: bold;
        }
        
        .col-diff.positive {
            color: #d32f2f;
        }
        
        .col-diff.negative {
            color: #388e3c;
        }
        
        .col-diff.neutral {
            color: #666;
        }
        
        .col-threshold {
            text-align: center;
            font-size: 0.9em;
        }
        
        .col-status {
            display: flex;
            align-items: center;
            gap: 0.5em;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .status-indicator.status-pass {
            background: #4caf50;
        }
        
        .status-indicator.status-fail {
            background: #f44336;
        }
        
        .status-indicator.status-disabled {
            background: #9e9e9e;
        }
        
        /* Make sure module header status indicators work */
        .module-header .status-indicator.status-pass {
            background: #4caf50;
        }
        
        .module-header .status-indicator.status-fail {
            background: #f44336;
        }
        
        .module-header .status-indicator.status-disabled {
            background: #9e9e9e;
        }
        
        .col-function {
            font-weight: 500;
        }
        

        
        @media (max-width: 768px) {
            .grid-header, .grid-row {
                grid-template-columns: 1fr;
            }
            
            .grid-header > div, .grid-row > div {
                border-right: none;
                border-bottom: 1px solid #e1e4e5;
            }
            

            
            .chart-container {
                height: 60px;
            }
        }
        </style>
        """
        
        return html
    
    def _generate_chart_data(self, test, chart_id):
        """Generate Chart.js configuration for a test's historical data"""
        historical = test.get('historical', [])
        if not historical:
            return "// No historical data available"
        
        # Prepare data points (reverse so newest is on left)
        reversed_historical = list(reversed(historical))
        labels = [f"T-{i}" for i in range(len(reversed_historical)-1, -1, -1)]
        values = [point['result_ms'] for point in reversed_historical]
        baseline_value = test['baseline_ms']
        
        # Determine point colors based on pass/fail status
        point_colors = []
        for point in reversed_historical:
            if point['status'] == 'pass':
                point_colors.append('#4caf50')  # Green
            else:
                point_colors.append('#f44336')  # Red
        
        # Get scale for Y-axis
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        padding = (max_val - min_val) * 0.1
        y_min = max(0, min_val - padding)
        y_max = max_val + padding
        
        return f"""
        (function() {{
            const ctx = document.getElementById('{chart_id}');
            if (!ctx) return;
            
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        data: {values},
                        borderColor: '#2196f3',
                        backgroundColor: 'transparent',
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointBackgroundColor: {point_colors},
                        pointBorderColor: {point_colors},
                        pointRadius: 3,
                        pointHoverRadius: 4,
                        tension: 0.2
                    }}, {{
                        label: 'Baseline',
                        data: Array({len(values)}).fill({baseline_value}),
                        borderColor: '#ff9800',
                        backgroundColor: 'transparent',
                        borderWidth: 1,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        borderDash: [2, 2]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {{
                        padding: {{
                            left: 5,
                            right: 5,
                            top: 5,
                            bottom: 5
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                title: function(context) {{
                                    return 'Run ' + context[0].label;
                                }},
                                label: function(context) {{
                                    return context.parsed.y.toFixed(3) + ' ms';
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            display: false,
                            grid: {{ display: false }}
                        }},
                        y: {{
                            display: true,
                            position: 'left',
                            min: {y_min},
                            max: {y_max},
                            grid: {{ display: false }},
                            ticks: {{
                                maxTicksLimit: 3,
                                font: {{ size: 10 }},
                                callback: function(value) {{
                                    return value.toFixed(1);
                                }}
                            }}
                        }}
                    }},
                    elements: {{
                        point: {{
                            hoverBorderWidth: 2
                        }}
                    }}
                }}
            }});
        }})();
        """
    
    def _get_status_class(self, status):
        """Convert status string to CSS class"""
        if 'pass' in status.lower():
            return 'status-pass'
        elif 'fail' in status.lower():
            return 'status-fail'
        elif 'disabled' in status.lower():
            return 'status-disabled'
        else:
            return 'status-unknown'


def setup(app: Sphinx):
    """Setup function for the Sphinx extension"""
    print("DEBUG: Setting up performance dashboard extension")
    app.add_directive('performancedashboard', PerformanceDashboardDirective)
    print("DEBUG: Performance dashboard directive added")
    
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    } 