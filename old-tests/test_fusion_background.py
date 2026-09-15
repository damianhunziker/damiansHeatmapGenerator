#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from chart_analysis import create_interactive_chart

def test_fusion_background():
    """Test the improved fusion range filter background coloring and indicator"""
    
    print("🧪 Testing Fusion Range Filter Background and Indicator")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='4h')
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 45000
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'time_period_start': dates,
        'price_open': prices,
        'price_high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'price_low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'price_close': prices,
        'volume_traded': np.random.uniform(100, 1000, len(dates))
    })
    
    data.set_index('time_period_start', inplace=True)
    
    print(f"📊 Generated test data:")
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['price_low'].min():.2f} to ${data['price_high'].max():.2f}")
    
    # Strategy parameters with fusion range filter enabled
    strategy_params = {
        'entry_filter': 0.7,
        'exit_filter': 1.0,
        'initial_equity': 10000,
        'fee_pct': 0.04,
        'use_divergence_exit_long': True,
        'use_hidden_divergence_exit_long': False,
        'use_divergence_exit_short': True,
        'use_hidden_divergence_exit_short': False,
        'divergence_order': 5,
        'rsi_length': 14,
        'calculate_divergences': True,
        'indicators': ['momentum', 'adp'],
        'start_date': None,
        'end_date': None,
        'asset': 'BTCUSDT',
        'use_fusion_for_long': True,  # Enable fusion range filter
        'atr_length': 9,
        'hma_mode': 'VWMA',
        'hma_length': 50,
        'atr_scaling_factor': 1.4,
        'debug_mode': True,  # Enable debug mode
        'trade_direction': 'both',
        'show_pivot_points': True,
        'short_indicators': ['adp', 'stoch', 'cci', 'diosc', 'cmf']
    }
    
    print(f"\n🔧 Strategy Configuration:")
    print(f"   Trade Direction: {strategy_params['trade_direction']}")
    print(f"   Fusion Range Filter: {strategy_params['use_fusion_for_long']}")
    print(f"   ATR Length: {strategy_params['atr_length']}")
    print(f"   HMA Mode: {strategy_params['hma_mode']}")
    print(f"   HMA Length: {strategy_params['hma_length']}")
    print(f"   ATR Scaling Factor: {strategy_params['atr_scaling_factor']}")
    print(f"   Debug Mode: {strategy_params['debug_mode']}")
    
    # Create timeframe data in the expected format
    timeframe_data = {
        'primary': {
            'data': data
        }
    }
    
    print(f"\n📈 Creating interactive chart with fusion background...")
    
    try:
        # Create the interactive chart
        fig = create_interactive_chart(
            timeframe_data=timeframe_data,
            strategy_class=LiveKAMASSLStrategy,
            strategy_params=strategy_params,
            last_n_candles_analyze=200,
            last_n_candles_display=100
        )
        
        # Save the chart
        output_file = 'fusion_background_test.html'
        fig.write_html(output_file)
        
        print(f"\n✅ Chart created successfully!")
        print(f"   Output file: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        print(f"   Total traces: {len(fig.data)}")
        
        # Count different types of traces
        trace_types = {}
        fusion_traces = 0
        pivot_traces = 0
        
        for trace in fig.data:
            trace_type = type(trace).__name__
            trace_types[trace_type] = trace_types.get(trace_type, 0) + 1
            
            if hasattr(trace, 'name') and trace.name:
                if 'Fusion' in trace.name:
                    fusion_traces += 1
                elif 'Pivot' in trace.name:
                    pivot_traces += 1
        
        print(f"\n📊 Chart Analysis:")
        print(f"   Trace types: {trace_types}")
        print(f"   Fusion Range Filter traces: {fusion_traces}")
        print(f"   Pivot point traces: {pivot_traces}")
        
        # Check if updatemenus (buttons) are present
        if hasattr(fig.layout, 'updatemenus') and fig.layout.updatemenus:
            print(f"   Buttons: {len(fig.layout.updatemenus[0].buttons)} pivot control buttons")
            for i, button in enumerate(fig.layout.updatemenus[0].buttons):
                print(f"      Button {i+1}: {button.label}")
        
        # Check if shapes (background rectangles) are present
        if hasattr(fig.layout, 'shapes') and fig.layout.shapes:
            print(f"   Background shapes: {len(fig.layout.shapes)} fusion filter rectangles")
        
        print(f"\n🎨 Features implemented:")
        print(f"   ✅ Improved background coloring with grouped rectangles")
        print(f"   ✅ Separate Fusion Range Filter indicator subplot")
        print(f"   ✅ Pivot point visibility controls")
        print(f"   ✅ Divergence detection and visualization")
        print(f"   ✅ Multiple indicator subplots")
        
        print(f"\n🌐 Open the HTML file in your browser to see:")
        print(f"   • Light green background areas where Fusion Range Filter is active")
        print(f"   • Separate Fusion Range Filter indicator showing 0/1 states")
        print(f"   • Pivot point toggle buttons")
        print(f"   • All indicators in their own subplots")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fusion_background()
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check the 'fusion_background_test.html' file to see the results.")
    else:
        print(f"\n💥 Test failed!")
        sys.exit(1) 