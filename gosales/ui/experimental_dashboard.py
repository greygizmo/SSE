"""
Experimental Dashboard - Pure HTML/CSS/JS
A beautiful, modern dashboard with dark theme and #BAD532 highlights
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GoSales Engine - Experimental Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom HTML/CSS/JS Dashboard
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GoSales Engine - Experimental Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            color: #e5e5e5;
            overflow-x: hidden;
            min-height: 100vh;
        }

        .dashboard {
            padding: 2rem;
            max-width: 1800px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(186, 213, 50, 0.1);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: linear-gradient(135deg, #BAD532 0%, #9BB828 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
            color: #0a0a0a;
        }

        .logo-text h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #BAD532 0%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .logo-text p {
            font-size: 0.75rem;
            color: #888;
            margin-top: 0.25rem;
        }

        .header-actions {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: rgba(186, 213, 50, 0.1);
            border: 1px solid rgba(186, 213, 50, 0.2);
            border-radius: 8px;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #BAD532;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }

        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .grid-large {
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        }

        /* Cards */
        .card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, #BAD532, transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }

        .card:hover {
            transform: translateY(-4px);
            border-color: rgba(186, 213, 50, 0.2);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.4), 0 0 40px rgba(186, 213, 50, 0.1);
        }

        .card:hover::before {
            opacity: 1;
        }

        /* Metric Cards */
        .metric-card {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }

        .metric-icon {
            width: 40px;
            height: 40px;
            background: rgba(186, 213, 50, 0.1);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .metric-label {
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 600;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #fff;
            line-height: 1;
        }

        .metric-change {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 600;
        }

        .metric-change.positive {
            background: rgba(186, 213, 50, 0.15);
            color: #BAD532;
        }

        .metric-change.negative {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }

        /* Chart Cards */
        .chart-card {
            min-height: 300px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }

        .chart-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: #fff;
        }

        .chart-controls {
            display: flex;
            gap: 0.5rem;
        }

        .chart-button {
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #888;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .chart-button:hover {
            background: rgba(186, 213, 50, 0.1);
            border-color: rgba(186, 213, 50, 0.3);
            color: #BAD532;
        }

        .chart-button.active {
            background: rgba(186, 213, 50, 0.15);
            border-color: rgba(186, 213, 50, 0.4);
            color: #BAD532;
        }

        /* Simple Bar Chart */
        .bar-chart {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            height: 200px;
            gap: 0.5rem;
        }

        .bar {
            flex: 1;
            background: linear-gradient(180deg, #BAD532 0%, rgba(186, 213, 50, 0.5) 100%);
            border-radius: 6px 6px 0 0;
            position: relative;
            transition: all 0.3s;
            cursor: pointer;
        }

        .bar:hover {
            background: linear-gradient(180deg, #C9E05C 0%, #BAD532 100%);
            transform: scaleY(1.05);
        }

        .bar-label {
            position: absolute;
            bottom: -24px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.75rem;
            color: #666;
            white-space: nowrap;
        }

        /* Table */
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .data-table thead {
            background: rgba(255, 255, 255, 0.03);
        }

        .data-table th {
            padding: 1rem;
            text-align: left;
            font-size: 0.75rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        .data-table td {
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            color: #e5e5e5;
        }

        .data-table tbody tr {
            transition: background 0.2s;
        }

        .data-table tbody tr:hover {
            background: rgba(186, 213, 50, 0.05);
        }

        .table-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .table-badge.success {
            background: rgba(186, 213, 50, 0.15);
            color: #BAD532;
        }

        .table-badge.warning {
            background: rgba(251, 191, 36, 0.15);
            color: #fbbf24;
        }

        .table-badge.error {
            background: rgba(239, 68, 68, 0.15);
            color: #ef4444;
        }

        /* Progress Bar */
        .progress-container {
            margin: 1rem 0;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #BAD532 0%, #9BB828 100%);
            border-radius: 4px;
            transition: width 1s ease-out;
        }

        /* Activity Feed */
        .activity-feed {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .activity-item {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
            border-left: 3px solid #BAD532;
            transition: all 0.2s;
        }

        .activity-item:hover {
            background: rgba(186, 213, 50, 0.05);
            transform: translateX(4px);
        }

        .activity-icon {
            width: 36px;
            height: 36px;
            background: rgba(186, 213, 50, 0.1);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .activity-content {
            flex: 1;
        }

        .activity-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.25rem;
        }

        .activity-time {
            font-size: 0.75rem;
            color: #666;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .dashboard {
                padding: 1rem;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .header {
                flex-direction: column;
                gap: 1rem;
                text-align: center;
            }
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .card {
            animation: fadeInUp 0.6s ease-out;
            animation-fill-mode: both;
        }

        .card:nth-child(1) { animation-delay: 0.1s; }
        .card:nth-child(2) { animation-delay: 0.2s; }
        .card:nth-child(3) { animation-delay: 0.3s; }
        .card:nth-child(4) { animation-delay: 0.4s; }
        .card:nth-child(5) { animation-delay: 0.5s; }
        .card:nth-child(6) { animation-delay: 0.6s; }
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <div class="logo-icon">GS</div>
                <div class="logo-text">
                    <h1>GoSales Engine</h1>
                    <p>Experimental Dashboard v3.0</p>
                </div>
            </div>
            <div class="header-actions">
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>All Systems Operational</span>
                </div>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="grid">
            <div class="card metric-card">
                <div class="metric-header">
                    <div class="metric-icon">📊</div>
                </div>
                <div class="metric-label">Total Revenue</div>
                <div class="metric-value">$2.4M</div>
                <div class="metric-change positive">↑ 18.2%</div>
            </div>

            <div class="card metric-card">
                <div class="metric-header">
                    <div class="metric-icon">🤖</div>
                </div>
                <div class="metric-label">Active Models</div>
                <div class="metric-value">14</div>
                <div class="metric-change positive">↑ 2</div>
            </div>

            <div class="card metric-card">
                <div class="metric-header">
                    <div class="metric-icon">🎯</div>
                </div>
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">94.8%</div>
                <div class="metric-change positive">↑ 1.3%</div>
            </div>

            <div class="card metric-card">
                <div class="metric-header">
                    <div class="metric-icon">👥</div>
                </div>
                <div class="metric-label">Predictions</div>
                <div class="metric-value">28.5K</div>
                <div class="metric-change positive">↑ 12.4%</div>
            </div>
        </div>

        <!-- Charts Section -->
        <div class="grid grid-large">
            <div class="card chart-card">
                <div class="chart-header">
                    <h3 class="chart-title">Revenue Trend</h3>
                    <div class="chart-controls">
                        <button class="chart-button active">7D</button>
                        <button class="chart-button">30D</button>
                        <button class="chart-button">90D</button>
                    </div>
                </div>
                <div class="bar-chart">
                    <div class="bar" style="height: 60%;">
                        <div class="bar-label">Mon</div>
                    </div>
                    <div class="bar" style="height: 75%;">
                        <div class="bar-label">Tue</div>
                    </div>
                    <div class="bar" style="height: 85%;">
                        <div class="bar-label">Wed</div>
                    </div>
                    <div class="bar" style="height: 70%;">
                        <div class="bar-label">Thu</div>
                    </div>
                    <div class="bar" style="height: 95%;">
                        <div class="bar-label">Fri</div>
                    </div>
                    <div class="bar" style="height: 50%;">
                        <div class="bar-label">Sat</div>
                    </div>
                    <div class="bar" style="height: 45%;">
                        <div class="bar-label">Sun</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 class="chart-title" style="margin-bottom: 1.5rem;">Model Performance</h3>
                <div class="progress-container">
                    <div class="progress-header">
                        <span>Solidworks</span>
                        <span style="color: #BAD532;">96.2%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 96.2%;"></div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-header">
                        <span>Services</span>
                        <span style="color: #BAD532;">94.8%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 94.8%;"></div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-header">
                        <span>Hardware</span>
                        <span style="color: #BAD532;">92.5%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 92.5%;"></div>
                    </div>
                </div>
                <div class="progress-container">
                    <div class="progress-header">
                        <span>Training</span>
                        <span style="color: #BAD532;">89.3%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 89.3%;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Activity and Table -->
        <div class="grid grid-large">
            <div class="card">
                <h3 class="chart-title" style="margin-bottom: 1.5rem;">Recent Activity</h3>
                <div class="activity-feed">
                    <div class="activity-item">
                        <div class="activity-icon">✅</div>
                        <div class="activity-content">
                            <div class="activity-title">Model training completed</div>
                            <div class="activity-time">2 minutes ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon">🔄</div>
                        <div class="activity-content">
                            <div class="activity-title">Pipeline execution started</div>
                            <div class="activity-time">15 minutes ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon">📊</div>
                        <div class="activity-content">
                            <div class="activity-title">Generated 1,247 new predictions</div>
                            <div class="activity-time">1 hour ago</div>
                        </div>
                    </div>
                    <div class="activity-item">
                        <div class="activity-icon">⚡</div>
                        <div class="activity-content">
                            <div class="activity-title">Feature engineering completed</div>
                            <div class="activity-time">2 hours ago</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 class="chart-title" style="margin-bottom: 1.5rem;">Top Opportunities</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Customer</th>
                            <th>Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Acme Corp</td>
                            <td>0.96</td>
                            <td><span class="table-badge success">Hot</span></td>
                        </tr>
                        <tr>
                            <td>TechStart Inc</td>
                            <td>0.92</td>
                            <td><span class="table-badge success">Hot</span></td>
                        </tr>
                        <tr>
                            <td>Global Systems</td>
                            <td>0.88</td>
                            <td><span class="table-badge warning">Warm</span></td>
                        </tr>
                        <tr>
                            <td>Design Studio</td>
                            <td>0.84</td>
                            <td><span class="table-badge warning">Warm</span></td>
                        </tr>
                        <tr>
                            <td>Innovate Ltd</td>
                            <td>0.79</td>
                            <td><span class="table-badge warning">Warm</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // Add some interactivity
        document.querySelectorAll('.chart-button').forEach(button => {
            button.addEventListener('click', function() {
                document.querySelectorAll('.chart-button').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                // Animate bars
                const bars = document.querySelectorAll('.bar');
                bars.forEach((bar, index) => {
                    const randomHeight = Math.random() * 60 + 40;
                    setTimeout(() => {
                        bar.style.height = randomHeight + '%';
                    }, index * 50);
                });
            });
        });

        // Animate progress bars on load
        window.addEventListener('load', () => {
            document.querySelectorAll('.progress-fill').forEach((fill, index) => {
                const targetWidth = fill.style.width;
                fill.style.width = '0%';
                setTimeout(() => {
                    fill.style.width = targetWidth;
                }, index * 200);
            });
        });
    </script>
</body>
</html>
"""

# Render the custom dashboard
components.html(dashboard_html, height=1200, scrolling=True)

