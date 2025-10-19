#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from chart_analysis import create_interactive_chart

def test_chart_analysis_fusion_debug():
    """Test chart_analysis.py with Fusion Range Filter debugging"""
    
    print("\n🔍 DEBUG: Testing chart_analysis.py with Fusion Range Filter")
    print("="*70)
    
    # Load cached data
    print("\n📊 Loading cached data...")
    df = pd.read_csv('ohlc_cache/BTCUSDT_4h_ohlc.csv')
    
    # Convert timestamp to datetime index
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    
    # Filter to a specific date range
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    df = df[start_date:end_date]
    
    print(f"   ✅ Data loaded:")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Data shape: {df.shape}")
    
    # Create timeframe data structure as expected by chart_analysis.py
    timeframe_data = {
        'primary': {
            'interval': '4h',
            'data': df
        }
    }
    
    print(f"\n🔧 Timeframe data structure created:")
    print(f"   Keys: {list(timeframe_data.keys())}")
    print(f"   Primary data shape: {timeframe_data['primary']['data'].shape}")
    
    # Strategy class
    strategy_class = LiveKAMASSLStrategy
    
    # Strategy parameters with Fusion Range Filter enabled and debug mode
    strategy_params = {
        'debug_mode': True,
        'use_fusion_for_long': True,
        'atr_length': 9,
        'hma_mode': 'VWMA',
        'hma_length': 50,
        'atr_scaling_factor': 1.4,
        'trade_direction': 'long',
        'entry_filter': 0.7,
        'exit_filter': 1.0,
        'calculate_divergences': True,
        'use_divergence_exit_long': True,
        'use_hidden_divergence_exit_long': False,
        'use_divergence_exit_short': True,
        'use_hidden_divergence_exit_short': False,
        'divergence_order': 5,
        'rsi_length': 14,
        'show_pivot_points': True
    }
    
    print(f"\n⚙️ Strategy parameters:")
    for key, value in strategy_params.items():
        print(f"   {key}: {value}")
    
    # Test parameters
    last_n_candles_analyze = 500
    last_n_candles_display = 200
    
    print(f"\n📈 Chart parameters:")
    print(f"   Analyze last {last_n_candles_analyze} candles")
    print(f"   Display last {last_n_candles_display} candles")
    
    # Add extensive debugging to chart_analysis call
    print(f"\n🚀 Calling create_interactive_chart...")
    print(f"   Strategy class: {strategy_class.__name__}")
    print(f"   Fusion filter enabled: {strategy_params['use_fusion_for_long']}")
    print(f"   Debug mode: {strategy_params['debug_mode']}")
    
    try:
        # Call chart_analysis.py
        fig = create_interactive_chart(
            timeframe_data=timeframe_data,
            strategy_class=strategy_class,
            strategy_params=strategy_params,
            last_n_candles_analyze=last_n_candles_analyze,
            last_n_candles_display=last_n_candles_display
        )
        
        print(f"\n✅ Chart creation successful!")
        print(f"   Figure type: {type(fig)}")
        print(f"   Number of traces: {len(fig.data)}")
        # Count subplots by checking layout
        subplot_count = 1  # At least one main plot
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'yaxis2'):
            subplot_count = 2
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'yaxis3'):
            subplot_count = 3
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'yaxis4'):
            subplot_count = 4
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'yaxis5'):
            subplot_count = 5
        print(f"   Number of subplots: {subplot_count}")
        
        # Analyze the figure structure
        print(f"\n🔍 Analyzing figure structure:")
        subplot_titles = []
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'annotations'):
            for annotation in fig.layout.annotations:
                if hasattr(annotation, 'text') and 'subplot' not in annotation.text.lower():
                    subplot_titles.append(annotation.text)
        
        print(f"   Subplot titles found: {subplot_titles}")
        
        # Check for Fusion Range Filter traces
        fusion_traces = []
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'name') and trace.name and 'fusion' in trace.name.lower():
                fusion_traces.append((i, trace.name))
        
        print(f"   Fusion-related traces: {fusion_traces}")
        
        # Check for background shapes (fusion filter rectangles)
        fusion_shapes = 0
        if hasattr(fig, 'layout') and hasattr(fig.layout, 'shapes'):
            for shape in fig.layout.shapes:
                if hasattr(shape, 'fillcolor') and 'green' in str(shape.fillcolor).lower():
                    fusion_shapes += 1
        
        print(f"   Fusion background shapes: {fusion_shapes}")
        
        # Save the chart
        output_file = 'chart_analysis_fusion_debug.html'
        fig.write_html(output_file)
        print(f"\n💾 Chart saved to: {output_file}")
        
        # Get file size
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")
        
        # Final analysis
        print(f"\n📊 Final Analysis:")
        if len(fusion_traces) > 0:
            print(f"   ✅ Fusion Range Filter traces found: {len(fusion_traces)}")
        else:
            print(f"   ❌ No Fusion Range Filter traces found!")
            
        if fusion_shapes > 0:
            print(f"   ✅ Fusion background shapes found: {fusion_shapes}")
        else:
            print(f"   ❌ No Fusion background shapes found!")
            
        if 'Fusion Range Filter' in subplot_titles:
            print(f"   ✅ Fusion Range Filter subplot title found!")
        else:
            print(f"   ❌ Fusion Range Filter subplot title NOT found!")
            print(f"   Available titles: {subplot_titles}")
        
        return fig
        
    except Exception as e:
        print(f"\n❌ Error during chart creation:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_chart_analysis_fusion_debug() 