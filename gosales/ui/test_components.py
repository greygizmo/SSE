"""
Quick test script to verify component functionality
Run this to test components without launching full Streamlit app
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gosales.ui.components import metric_card, card, alert, badge, stat_grid


def test_metric_card():
    """Test metric_card with various delta values"""
    print("Testing metric_card component...")
    
    # Test numeric delta
    try:
        result = metric_card(
            label="Revenue",
            value="$1.2M",
            delta="+12.5%",
            icon="💰"
        )
        print("✅ Numeric delta (+12.5%): PASS")
    except Exception as e:
        print(f"❌ Numeric delta failed: {e}")
    
    # Test text delta
    try:
        result = metric_card(
            label="Status",
            value="Excellent",
            delta="All Clear",
            icon="✅"
        )
        print("✅ Text delta (All Clear): PASS")
    except Exception as e:
        print(f"❌ Text delta failed: {e}")
    
    # Test negative delta
    try:
        result = metric_card(
            label="Errors",
            value="5",
            delta="-3",
            icon="⚠️"
        )
        print("✅ Negative delta (-3): PASS")
    except Exception as e:
        print(f"❌ Negative delta failed: {e}")


def test_stat_grid():
    """Test stat_grid with mixed delta types"""
    print("\nTesting stat_grid component...")
    
    try:
        result = stat_grid([
            {"label": "Numeric", "value": "100", "delta": "+12%", "icon": "📊"},
            {"label": "Text", "value": "Good", "delta": "All systems go", "icon": "✅"},
            {"label": "Negative", "value": "50", "delta": "-5%", "icon": "📉"},
            {"label": "No delta", "value": "75", "icon": "🎯"}
        ])
        print("✅ stat_grid with mixed deltas: PASS")
    except Exception as e:
        print(f"❌ stat_grid failed: {e}")


def test_alert():
    """Test alert component"""
    print("\nTesting alert component...")
    
    variants = ["info", "success", "warning", "error"]
    for variant in variants:
        try:
            result = alert(
                message=f"This is a {variant} message",
                variant=variant,
                title=f"{variant.title()} Test"
            )
            print(f"✅ Alert variant '{variant}': PASS")
        except Exception as e:
            print(f"❌ Alert variant '{variant}' failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("GoSales UI Components Test Suite")
    print("=" * 60)
    
    test_metric_card()
    test_stat_grid()
    test_alert()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


