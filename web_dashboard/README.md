# 🚀 GoSales Engine - Standalone Web Dashboard

A modern, interactive web-based alternative to the Streamlit dashboard with enhanced visualizations and user experience.

## ✨ Features

### 🎨 Modern Design
- Dark theme with #BAD532 brand highlights
- Glassmorphism effects and smooth animations
- Responsive design (mobile, tablet, desktop)
- Interactive charts and visualizations

### 📊 Full Feature Set
- **Overview Dashboard** - Key metrics, revenue trends, activity feed
- **Model Metrics** - Performance metrics, ROC curves, feature importance
- **Whitespace Analysis** - Top opportunities with scores and rankings
- **Model Validation** - Validation status, drift monitoring, history
- **Explainability** - SHAP values, feature importance (coming soon)
- **Monitoring** - System health, data quality (coming soon)
- **Architecture** - System diagrams (coming soon)
- **Quality Assurance** - Leakage tests, QA reports (coming soon)

### 🔧 Technologies Used
- **Frontend:** Pure HTML, CSS, JavaScript
- **State Management:** Alpine.js (lightweight reactive framework)
- **Charts:** Chart.js for beautiful visualizations
- **Data Viz:** D3.js for advanced visualizations
- **Backend:** Flask (Python) for API server
- **Data:** Connects to GoSales SQLite databases and CSV outputs

## 🚀 Quick Start

### Prerequisites
```bash
pip install flask flask-cors pandas
```

### Start the Server
```powershell
# From the web_dashboard directory
python server.py
```

### Access the Dashboard
Open your browser to: **http://localhost:5000**

## 📂 Project Structure

```
web_dashboard/
├── index.html              # Main HTML file
├── styles/
│   ├── main.css           # Core styles, layout
│   ├── components.css     # UI component styles
│   └── charts.css         # Chart-specific styles
├── scripts/
│   ├── app.js            # Main application logic
│   └── charts.js         # Chart initializations
├── server.py             # Flask API server
└── README.md             # This file
```

## 🎯 API Endpoints

The Flask server provides these endpoints:

| Endpoint | Description |
|----------|-------------|
| `/api/models` | List of available models |
| `/api/metrics/<model>` | Metrics for specific model |
| `/api/whitespace` | Whitespace analysis data |
| `/api/validation` | Validation run results |
| `/api/stats` | Dashboard statistics |
| `/api/opportunities` | Top scoring opportunities |

## 🎨 Customization

### Change Brand Color
Edit `styles/main.css`:
```css
:root {
    --primary: #BAD532;  /* Change to your color */
}
```

### Modify Charts
Edit `scripts/charts.js` to customize chart configurations.

### Add New Views
1. Add navigation item in `index.html`
2. Create view section with `x-show="currentView === 'yourview'"`
3. Add view title in `app.js` `getViewTitle()`

## 🔄 Connecting Real Data

The dashboard can connect to real GoSales data:

### From Outputs Directory
Place this in the `gosales/outputs/` directory:
- `metrics_*.json` - Model metrics
- `whitespace_*.csv` - Whitespace analysis
- `validation/` - Validation runs

### From Database
Point to `gosales/gosales_curated.db` for:
- Transaction counts
- Customer data
- Real-time stats

## 📊 vs. Streamlit Dashboard

| Feature | Web Dashboard | Streamlit |
|---------|--------------|-----------|
| **Load Speed** | ⚡ Instant | ✅ Fast |
| **Visual Design** | 🔥 Premium | ✅ Professional |
| **Customization** | 🔥 Total control | ⚠️ Limited |
| **Interactivity** | 🔥 Highly interactive | ✅ Good |
| **Data Binding** | ⚠️ Manual | 🔥 Automatic |
| **Deployment** | 🔥 Simple static | ⚠️ Needs Python |
| **Mobile** | 🔥 Fully responsive | ⚠️ Limited |

## 🚀 Deployment Options

### Option 1: Static Hosting
Deploy to Netlify, Vercel, or GitHub Pages:
```bash
# Build doesn't need compilation - just upload files!
netlify deploy --prod --dir=web_dashboard
```

### Option 2: With API Server
Deploy Flask backend + static frontend:
```bash
# Heroku
heroku create gosales-dashboard
git push heroku main

# AWS, Azure, GCP
# Deploy Flask app with static file serving
```

### Option 3: Embedded
Use the dashboard in an iframe or embed in existing sites.

## 🎓 Usage Guide

### Navigation
- Click sidebar items to switch views
- Use breadcrumbs to track location
- 🔄 button refreshes data
- 🌙 button toggles dark/light mode

### Charts
- Hover for tooltips
- Click time range buttons (7D, 30D, 90D)
- Charts auto-update based on selections

### Tables
- Search functionality
- Sortable columns
- Export to CSV
- Clickable rows for details

## 🛠️ Development

### Run in Development Mode
```bash
# Backend
python server.py

# Frontend (live reload)
# Use any static file server or just open index.html
```

### Add New Chart
```javascript
// In scripts/charts.js
function initializeYourChart() {
    const ctx = document.getElementById('yourChart');
    const chart = new Chart(ctx, {
        // Chart configuration
    });
}
```

### Add New API Endpoint
```python
# In server.py
@app.route('/api/your-endpoint')
def your_endpoint():
    # Fetch and process data
    return jsonify(data)
```

## 🎯 Roadmap

### Phase 1: Core Features (✅ Complete)
- [x] Dashboard overview
- [x] Model metrics
- [x] Whitespace analysis
- [x] Validation status
- [x] API server

### Phase 2: Advanced Features (🚧 In Progress)
- [ ] Real-time data updates
- [ ] SHAP visualizations
- [ ] Architecture diagrams
- [ ] QA test results
- [ ] User authentication

### Phase 3: Enhancements
- [ ] Collaborative features
- [ ] Custom dashboards
- [ ] Export to PowerPoint/PDF
- [ ] Scheduled reports
- [ ] Mobile app

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port 5000 is available
netstat -an | findstr :5000

# Use different port
python server.py --port 8080
```

### Charts Not Loading
- Check browser console for errors
- Ensure Chart.js CDN is accessible
- Verify canvas elements have IDs

### Data Not Showing
- Check API endpoints return data: `http://localhost:5000/api/stats`
- Verify paths to gosales directory
- Check file permissions

## 📞 Support

- Check logs in terminal where server is running
- Browser console for frontend errors
- Verify data files exist in `gosales/outputs/`

## 🎉 Benefits

### For Users
- ✨ Beautiful, modern interface
- ⚡ Fast, responsive experience
- 📱 Works on all devices
- 🎯 Intuitive navigation

### For Developers
- 🔧 Easy to customize
- 📦 Simple deployment
- 🎨 Full design control
- 🚀 Standard web technologies

### For Business
- 💼 Professional presentation
- 🌐 Easy to share (just a URL)
- 📊 Better data insights
- 🎓 Lower learning curve

## 📝 License

Same as GoSales Engine main project.

---

**Built with ❤️ for better data visualization and user experience**

Access at: http://localhost:5000

