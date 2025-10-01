// GoSales Engine - Chart Initializations

let revenueChart, rocChart, featureChart, validationChart;

// Chart.js default config
Chart.defaults.color = '#a0a0a0';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.font.family = 'Inter, sans-serif';

const chartColors = {
    primary: '#BAD532',
    primaryDark: '#9BB828',
    primaryLight: '#C9E05C',
    success: '#BAD532',
    warning: '#fbbf24',
    error: '#ef4444',
    info: '#3b82f6'
};

// Revenue Trend Chart
function initializeRevenueChart() {
    const ctx = document.getElementById('revenueChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (revenueChart) {
        revenueChart.destroy();
    }

    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(186, 213, 50, 0.3)');
    gradient.addColorStop(1, 'rgba(186, 213, 50, 0)');

    revenueChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: getDateLabels(7),
            datasets: [{
                label: 'Revenue',
                data: [320000, 420000, 380000, 450000, 520000, 410000, 490000],
                borderColor: chartColors.primary,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: chartColors.primary,
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e5e5e5',
                    bodyColor: '#a0a0a0',
                    borderColor: 'rgba(186, 213, 50, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toLocaleString();
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000) + 'K';
                        }
                    }
                }
            }
        }
    });
}

// ROC Curve Chart
function initializeROCChart() {
    const ctx = document.getElementById('rocChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (rocChart) {
        rocChart.destroy();
    }

    rocChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 11}, (_, i) => (i / 10).toFixed(1)),
            datasets: [{
                label: 'ROC Curve',
                data: [0, 0.15, 0.35, 0.55, 0.70, 0.82, 0.90, 0.95, 0.98, 0.995, 1.0],
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(186, 213, 50, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: chartColors.primary
            }, {
                label: 'Random Classifier',
                data: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                borderColor: '#666666',
                borderDash: [5, 5],
                fill: false,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#a0a0a0',
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e5e5e5',
                    bodyColor: '#a0a0a0',
                    borderColor: 'rgba(186, 213, 50, 0.3)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'False Positive Rate',
                        color: '#a0a0a0'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'True Positive Rate',
                        color: '#a0a0a0'
                    }
                }
            }
        }
    });
}

// Feature Importance Chart
function initializeFeatureChart() {
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (featureChart) {
        featureChart.destroy();
    }

    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [
                'RFM Score',
                'Recency',
                'Frequency',
                'Monetary',
                'Tenure',
                'Territory',
                'Industry',
                'Account Age'
            ],
            datasets: [{
                label: 'Importance',
                data: [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05],
                backgroundColor: chartColors.primary,
                borderRadius: 6,
                hoverBackgroundColor: chartColors.primaryLight
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e5e5e5',
                    bodyColor: '#a0a0a0',
                    borderColor: 'rgba(186, 213, 50, 0.3)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return 'Importance: ' + (context.parsed.x * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                },
                y: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Validation Metrics History Chart
function initializeValidationChart() {
    const ctx = document.getElementById('validationChart');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (validationChart) {
        validationChart.destroy();
    }

    validationChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: getDateLabels(30),
            datasets: [{
                label: 'Calibration MAE',
                data: generateTrendData(30, 0.02, 0.03),
                borderColor: chartColors.primary,
                backgroundColor: 'transparent',
                tension: 0.4,
                pointRadius: 2,
                yAxisID: 'y'
            }, {
                label: 'PSI',
                data: generateTrendData(30, 0.10, 0.20),
                borderColor: chartColors.info,
                backgroundColor: 'transparent',
                tension: 0.4,
                pointRadius: 2,
                yAxisID: 'y'
            }, {
                label: 'KS Statistic',
                data: generateTrendData(30, 0.15, 0.25),
                borderColor: chartColors.warning,
                backgroundColor: 'transparent',
                tension: 0.4,
                pointRadius: 2,
                yAxisID: 'y'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#a0a0a0',
                        usePointStyle: true
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e5e5e5',
                    bodyColor: '#a0a0a0',
                    borderColor: 'rgba(186, 213, 50, 0.3)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    }
                }
            }
        }
    });
}

// Helper function to generate trend data
function generateTrendData(count, min, max) {
    const data = [];
    let current = (min + max) / 2;
    for (let i = 0; i < count; i++) {
        current += (Math.random() - 0.5) * (max - min) * 0.2;
        current = Math.max(min, Math.min(max, current));
        data.push(parseFloat(current.toFixed(4)));
    }
    return data;
}

// Update revenue chart based on time range
function updateRevenueChart(timeRange) {
    if (!revenueChart) return;

    let labels, data;
    switch(timeRange) {
        case '7d':
            labels = getDateLabels(7);
            data = [320000, 420000, 380000, 450000, 520000, 410000, 490000];
            break;
        case '30d':
            labels = getDateLabels(30);
            data = Array.from({length: 30}, () => Math.floor(Math.random() * 300000) + 300000);
            break;
        case '90d':
            labels = getDateLabels(90);
            data = Array.from({length: 90}, () => Math.floor(Math.random() * 300000) + 300000);
            break;
    }

    revenueChart.data.labels = labels;
    revenueChart.data.datasets[0].data = data;
    revenueChart.update();
}

