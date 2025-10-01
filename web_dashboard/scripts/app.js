// GoSales Engine - Main Application Logic

function dashboardApp() {
    return {
        // State
        currentView: 'overview',
        darkMode: true,
        timeRange: '7d',
        selectedModel: 'solidworks',
        
        // Data
        stats: {
            revenue: 2400000,
            activeModels: 14,
            accuracy: 94.8,
            predictions: 28500
        },
        
        models: [
            'solidworks',
            'services',
            'hardware',
            'training',
            'simulation',
            'camworks'
        ],
        
        modelPerformance: [
            { name: 'Solidworks', accuracy: 96.2 },
            { name: 'Services', accuracy: 94.8 },
            { name: 'Hardware', accuracy: 92.5 },
            { name: 'Training', accuracy: 89.3 }
        ],
        
        recentActivity: [
            { id: 1, icon: '✅', title: 'Model training completed', time: '2 minutes ago' },
            { id: 2, icon: '🔄', title: 'Pipeline execution started', time: '15 minutes ago' },
            { id: 3, icon: '📊', title: 'Generated 1,247 new predictions', time: '1 hour ago' },
            { id: 4, icon: '⚡', title: 'Feature engineering completed', time: '2 hours ago' }
        ],
        
        topOpportunities: [
            { id: 1, customer: 'Acme Corp', score: '0.96', status: 'Hot' },
            { id: 2, customer: 'TechStart Inc', score: '0.92', status: 'Hot' },
            { id: 3, customer: 'Global Systems', score: '0.88', status: 'Warm' },
            { id: 4, customer: 'Design Studio', score: '0.84', status: 'Warm' },
            { id: 5, customer: 'Innovate Ltd', score: '0.79', status: 'Warm' }
        ],
        
        // Methods
        init() {
            console.log('Dashboard initialized');
            this.loadRealData();
            this.initializeCharts();
            this.animateProgressBars();
        },
        
        async loadRealData() {
            try {
                // Load stats
                const statsRes = await fetch('/api/stats');
                if (statsRes.ok) {
                    const stats = await statsRes.json();
                    this.stats = stats;
                }
                
                // Load opportunities
                const oppsRes = await fetch('/api/opportunities');
                if (oppsRes.ok) {
                    const opps = await oppsRes.json();
                    this.topOpportunities = opps.slice(0, 10).map((opp, idx) => ({
                        id: idx + 1,
                        customer: opp.customer,
                        division: opp.division || '',
                        score: opp.score.toFixed(2),
                        status: opp.status
                    }));
                }
            } catch (error) {
                console.error('Error loading real data:', error);
            }
        },
        
        getViewTitle() {
            const titles = {
                'overview': 'Dashboard Overview',
                'metrics': 'Model Performance Metrics',
                'whitespace': 'Whitespace Analysis',
                'prospects': 'Prospect Management',
                'validation': 'Model Validation',
                'explainability': 'Model Explainability',
                'monitoring': 'System Monitoring',
                'architecture': 'System Architecture',
                'qa': 'Quality Assurance'
            };
            return titles[this.currentView] || this.currentView;
        },
        
        formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(value);
        },
        
        formatNumber(value) {
            if (value >= 1000) {
                return (value / 1000).toFixed(1) + 'K';
            }
            return value.toString();
        },
        
        refreshData() {
            console.log('Refreshing data...');
            // Simulate API call
            setTimeout(() => {
                this.stats.predictions += Math.floor(Math.random() * 100);
                console.log('Data refreshed');
            }, 500);
        },
        
        toggleTheme() {
            this.darkMode = !this.darkMode;
            document.body.classList.toggle('light-mode');
        },
        
        setTimeRange(range) {
            this.timeRange = range;
            this.updateRevenueChart(range);
        },
        
        initializeCharts() {
            // Charts will be initialized in charts.js
            setTimeout(() => {
                if (typeof initializeRevenueChart === 'function') {
                    initializeRevenueChart();
                }
                if (typeof initializeROCChart === 'function') {
                    initializeROCChart();
                }
                if (typeof initializeFeatureChart === 'function') {
                    initializeFeatureChart();
                }
                if (typeof initializeValidationChart === 'function') {
                    initializeValidationChart();
                }
            }, 100);
        },
        
        updateRevenueChart(range) {
            console.log('Updating revenue chart for range:', range);
            // Will be implemented in charts.js
        },
        
        animateProgressBars() {
            setTimeout(() => {
                const progressBars = document.querySelectorAll('.progress-fill');
                progressBars.forEach(bar => {
                    const width = bar.style.width;
                    bar.style.width = '0%';
                    setTimeout(() => {
                        bar.style.width = width;
                    }, 100);
                });
            }, 200);
        }
    };
}

// Utility Functions
function getRandomData(count, min, max) {
    const data = [];
    for (let i = 0; i < count; i++) {
        data.push(Math.floor(Math.random() * (max - min + 1)) + min);
    }
    return data;
}

function getDateLabels(days) {
    const labels = [];
    const today = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    return labels;
}

