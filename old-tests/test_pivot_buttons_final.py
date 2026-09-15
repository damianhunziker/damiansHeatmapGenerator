#!/usr/bin/env python3
"""
Test script for pivot point buttons functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chart_analysis import create_interactive_chart
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from strategy_utils import fetch_data
import pandas as pd

def test_pivot_buttons():
    """Test the pivot point buttons functionality"""
    print("🔍 Testing Pivot Point Buttons Functionality")
    print("=" * 60)
    
    # Strategy parameters
    strategy_params = {
        'debug_mode': True,
        'show_pivot_points': True,
        'trade_direction': 'both',
        'calculate_divergences': True,
        'start_date': "2024-01-01",
        'end_date': "2024-02-01",
        'asset': "BTCUSDT",
        'initial_equity': 10000,
        'fee_pct': 0.04
    }
    
    print("\n📊 Fetching data...")
    
    # Fetch data
    raw_data = fetch_data(
        asset="BTCUSDT",
        interval="4h"
    )
    
    # Format data as expected by create_interactive_chart
    timeframe_data = {
        'primary': {
            'data': raw_data
        }
    }
    
    print("\n📈 Creating interactive chart...")
    
    # Create the chart
    fig = create_interactive_chart(
        timeframe_data=timeframe_data,
        strategy_class=LiveKAMASSLStrategy,
        strategy_params=strategy_params,
        last_n_candles_analyze=1000,
        last_n_candles_display=200
    )
    
    if fig:
        # Save the chart as HTML file
        html_filename = "pivot_buttons_test_final.html"
        fig.write_html(html_filename)
        print(f"\n💾 Chart saved as {html_filename}")
        
        # Print button information
        if hasattr(fig.layout, 'updatemenus') and fig.layout.updatemenus:
            print(f"\n🎛️ Button Information:")
            print(f"   Number of button menus: {len(fig.layout.updatemenus)}")
            
            for i, menu in enumerate(fig.layout.updatemenus):
                print(f"\n   Menu {i+1}:")
                if hasattr(menu, 'buttons'):
                    print(f"      Number of buttons: {len(menu.buttons)}")
                    for j, button in enumerate(menu.buttons):
                        if hasattr(button, 'label'):
                            print(f"         Button {j+1}: {button.label}")
        
        # Count pivot point traces
        long_pivot_count = 0
        short_pivot_count = 0
        total_traces = len(fig.data)
        
        for trace in fig.data:
            if hasattr(trace, 'legendgroup'):
                if trace.legendgroup == 'long_pivots':
                    long_pivot_count += 1
                elif trace.legendgroup == 'short_pivots':
                    short_pivot_count += 1
        
        print(f"\n📈 Chart Statistics:")
        print(f"   Total traces: {total_traces}")
        print(f"   Long pivot traces: {long_pivot_count}")
        print(f"   Short pivot traces: {short_pivot_count}")
        print(f"   Other traces: {total_traces - long_pivot_count - short_pivot_count}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Open {html_filename} in your browser to test the buttons")
        print(f"   Available buttons:")
        print(f"      - Hide All Pivots")
        print(f"      - Show All Pivots") 
        print(f"      - Hide Long Pivots")
        print(f"      - Show Long Pivots")
        print(f"      - Hide Short Pivots")
        print(f"      - Show Short Pivots")
        
        return True
    else:
        print("❌ Failed to create chart")
        return False

if __name__ == "__main__":
    success = test_pivot_buttons()
    if success:
        print("\n🎉 Pivot button test completed successfully!")
    else:
        print("\n💥 Pivot button test failed!")
        sys.exit(1) 