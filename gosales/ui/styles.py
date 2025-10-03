"""
ShadCN-inspired CSS styles for GoSales Engine Streamlit app
"""


def get_shadcn_styles() -> str:
    """
    Returns comprehensive ShadCN-inspired CSS for the Streamlit app
    """
    return """
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ==================== ROOT VARIABLES (Grayscale + Professional Blue) ==================== */
    :root {
        /* Colors - Light Mode - Grayscale with Professional Blue highlights */
        --background: 0 0% 100%;
        --foreground: 0 0% 10%;
        --card: 0 0% 98%;
        --card-foreground: 0 0% 10%;
        --popover: 0 0% 100%;
        --popover-foreground: 0 0% 10%;
        --primary: 217 91% 60%;  /* #2a5298 */
        --primary-foreground: 0 0% 10%;
        --secondary: 0 0% 85%;
        --secondary-foreground: 0 0% 20%;
        --muted: 0 0% 94%;
        --muted-foreground: 0 0% 45%;
        --accent: 0 0% 92%;
        --accent-foreground: 0 0% 10%;
        --destructive: 0 84% 50%;
        --destructive-foreground: 0 0% 100%;
        --border: 0 0% 88%;
        --input: 0 0% 88%;
        --ring: 217 91% 60%;  /* #2a5298 */
        --radius: 0.375rem;  /* More compact */
        
        /* Success/Warning/Info colors - Grayscale friendly */
        --success: 217 91% 60%;  /* #2a5298 for success */
        --success-foreground: 0 0% 10%;
        --warning: 45 93% 47%;  /* Warm gray-yellow */
        --warning-foreground: 0 0% 10%;
        --info: 0 0% 40%;  /* Dark gray */
        --info-foreground: 0 0% 100%;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }
    
    /* Dark Mode Variables - Grayscale with Professional Blue */
    [data-theme="dark"] {
        --background: 0 0% 8%;
        --foreground: 0 0% 95%;
        --card: 0 0% 12%;
        --card-foreground: 0 0% 95%;
        --popover: 0 0% 10%;
        --popover-foreground: 0 0% 95%;
        --primary: 217 91% 60%;  /* #2a5298 */
        --primary-foreground: 0 0% 10%;
        --secondary: 0 0% 18%;
        --secondary-foreground: 0 0% 90%;
        --muted: 0 0% 15%;
        --muted-foreground: 0 0% 60%;
        --accent: 0 0% 20%;
        --accent-foreground: 0 0% 95%;
        --border: 0 0% 22%;
        --input: 0 0% 22%;
        --success: 217 91% 60%;  /* #2a5298 */
        --success-foreground: 0 0% 10%;
    }
    
    /* ==================== GLOBAL RESETS ==================== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: hsl(var(--background));
        color: hsl(var(--foreground));
    }
    
    /* ==================== TYPOGRAPHY ==================== */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    h1 { font-size: 2.25rem; margin-bottom: 1rem; }
    h2 { font-size: 1.875rem; margin-bottom: 0.875rem; }
    h3 { font-size: 1.5rem; margin-bottom: 0.75rem; }
    
    /* ==================== SIDEBAR STYLING ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, hsl(var(--card)) 0%, hsl(var(--muted)) 100%);
        border-right: 1px solid hsl(var(--border));
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1rem;
    }
    
    /* ==================== NAVIGATION TABS ==================== */
    .stTabs {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: hsl(var(--muted));
        padding: 4px;
        border-radius: var(--radius);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: calc(var(--radius) - 4px);
        padding: 8px 16px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s;
        border: none;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: hsl(var(--accent));
    }
    
    .stTabs [aria-selected="true"] {
        background-color: hsl(var(--background)) !important;
        box-shadow: var(--shadow-sm);
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        border: none;
        border-radius: var(--radius);
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        background: hsl(var(--primary) / 0.9);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary button variant */
    .stButton.secondary > button {
        background: hsl(var(--secondary));
        color: hsl(var(--secondary-foreground));
    }
    
    /* ==================== INPUT FIELDS ==================== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > input {
        border: 1px solid hsl(var(--input));
        border-radius: var(--radius);
        padding: 0.5rem 0.75rem;
        background-color: hsl(var(--background));
        color: hsl(var(--foreground));
        transition: all 0.2s;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        outline: none;
        border-color: hsl(var(--ring));
        box-shadow: 0 0 0 3px hsl(var(--ring) / 0.2);
    }
    
    /* ==================== CARDS ==================== */
    .element-container {
        transition: all 0.2s;
    }
    
    /* Custom card class */
    .shadcn-card {
        background: hsl(var(--card));
        border: 1px solid hsl(var(--border));
        border-radius: var(--radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s;
    }
    
    .shadcn-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    /* ==================== METRICS (Compact & Modern) ==================== */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;  /* More compact */
        font-weight: 600;
        color: hsl(var(--foreground));
        line-height: 1.2;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem;  /* Smaller label */
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: hsl(var(--muted-foreground));
        margin-bottom: 0.25rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.75rem;  /* Smaller delta */
        font-weight: 500;
    }
    
    /* Metric container - more compact */
    [data-testid="metric-container"] {
        padding: 0.75rem 1rem !important;  /* Reduced padding */
    }
    
    /* ==================== DATA TABLES ==================== */
    .dataframe {
        border: 1px solid hsl(var(--border)) !important;
        border-radius: var(--radius);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    
    .dataframe thead tr th {
        background: hsl(var(--muted)) !important;
        color: hsl(var(--muted-foreground)) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        padding: 12px !important;
    }
    
    .dataframe tbody tr:hover {
        background: hsl(var(--accent)) !important;
    }
    
    .dataframe tbody tr td {
        padding: 12px !important;
        border-bottom: 1px solid hsl(var(--border)) !important;
    }
    
    /* ==================== ALERTS ==================== */
    .stAlert {
        border-radius: var(--radius);
        border: 1px solid hsl(var(--border));
        padding: 1rem;
    }
    
    /* Info alert */
    [data-baseweb="notification"][kind="info"] {
        background-color: hsl(var(--info) / 0.1);
        border-left: 4px solid hsl(var(--info));
    }
    
    /* Success alert */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background-color: hsl(var(--success) / 0.1);
        border-left: 4px solid hsl(var(--success));
    }
    
    /* Warning alert */
    .stWarning, [data-baseweb="notification"][kind="warning"] {
        background-color: hsl(var(--warning) / 0.1);
        border-left: 4px solid hsl(var(--warning));
    }
    
    /* Error alert */
    .stError, [data-baseweb="notification"][kind="negative"] {
        background-color: hsl(var(--destructive) / 0.1);
        border-left: 4px solid hsl(var(--destructive));
    }
    
    /* ==================== EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background-color: hsl(var(--muted));
        border-radius: var(--radius);
        border: 1px solid hsl(var(--border));
        font-weight: 500;
        padding: 0.75rem 1rem;
        transition: all 0.2s;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: hsl(var(--accent));
    }
    
    .streamlit-expanderContent {
        border: 1px solid hsl(var(--border));
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 1rem;
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, hsl(var(--primary)), hsl(var(--primary) / 0.8));
        border-radius: var(--radius);
    }
    
    /* ==================== CHARTS ==================== */
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        border: 1px solid hsl(var(--border));
        border-radius: var(--radius);
        padding: 1rem;
        background: hsl(var(--card));
        box-shadow: var(--shadow-sm);
    }
    
    /* ==================== LOADING SPINNER ==================== */
    .stSpinner > div {
        border-top-color: hsl(var(--primary)) !important;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        border: 2px dashed hsl(var(--border));
        border-radius: var(--radius);
        padding: 2rem;
        background: hsl(var(--muted) / 0.3);
        transition: all 0.2s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: hsl(var(--primary));
        background: hsl(var(--muted) / 0.5);
    }
    
    /* ==================== DOWNLOAD BUTTON ==================== */
    .stDownloadButton > button {
        background: hsl(var(--secondary));
        color: hsl(var(--secondary-foreground));
        border: 1px solid hsl(var(--border));
    }
    
    .stDownloadButton > button:hover {
        background: hsl(var(--accent));
        border-color: hsl(var(--primary));
    }
    
    /* ==================== CUSTOM ANIMATIONS ==================== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: hsl(var(--muted));
        border-radius: var(--radius);
    }
    
    ::-webkit-scrollbar-thumb {
        background: hsl(var(--muted-foreground) / 0.3);
        border-radius: var(--radius);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: hsl(var(--muted-foreground) / 0.5);
    }
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {
        h1 { font-size: 1.75rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.25rem; }
        
        .shadcn-card {
            padding: 1rem;
        }
    }
    
    /* ==================== ACCESSIBILITY ==================== */
    *:focus-visible {
        outline: 2px solid hsl(var(--ring));
        outline-offset: 2px;
    }
    
    /* Reduce motion for users who prefer it */
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* ==================== CUSTOM UTILITIES ==================== */
    .text-muted {
        color: hsl(var(--muted-foreground));
    }
    
    .text-primary {
        color: hsl(var(--primary));
    }
    
    .bg-card {
        background-color: hsl(var(--card));
    }
    
    .border-default {
        border: 1px solid hsl(var(--border));
    }
    
    .rounded {
        border-radius: var(--radius);
    }
    
    .shadow-sm {
        box-shadow: var(--shadow-sm);
    }
    
    .shadow-md {
        box-shadow: var(--shadow-md);
    }
    
    /* ==================== BADGE COMPONENT ==================== */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.625rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: calc(var(--radius) - 2px);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-primary {
        background: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
    }
    
    .badge-secondary {
        background: hsl(var(--secondary));
        color: hsl(var(--secondary-foreground));
    }
    
    .badge-success {
        background: hsl(var(--success));
        color: hsl(var(--success-foreground));
    }
    
    .badge-warning {
        background: hsl(var(--warning));
        color: hsl(var(--warning-foreground));
    }
    
    .badge-error {
        background: hsl(var(--destructive));
        color: hsl(var(--destructive-foreground));
    }
    </style>
    """


def get_dark_mode_toggle() -> str:
    """
    Returns JavaScript code for dark mode toggle functionality
    """
    return """
    <script>
    // Dark mode toggle functionality
    function toggleDarkMode() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }
    
    // Load saved theme preference
    function loadThemePreference() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }
    
    // Initialize theme on page load
    loadThemePreference();
    </script>
    """

