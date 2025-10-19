import pandas as pd
import numpy as np
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
from datetime import datetime, timedelta

def debug_may_3_trade():
    """Debug the specific trade on May 3, 2019 at 12:00 PM"""
    print("\n🔍 DEBUGGING MAY 3, 2019 TRADE")
    print("="*60)
    
    # Load cached data
    df = pd.read_csv('ohlc_cache/BTCUSDT_4h_ohlc.csv')
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    df.set_index('time_period_start', inplace=True)
    
    # Filter to period around May 3, 2019
    start_date = '2019-05-01'
    end_date = '2019-05-10'
    df_debug = df[start_date:end_date].copy()
    
    print(f"\nLoaded data from {start_date} to {end_date}")
    print(f"Data shape: {df_debug.shape}")
    print(f"Date range: {df_debug.index.min()} to {df_debug.index.max()}")
    
    # Initialize strategy with exact same parameters
    strategy = LiveKAMASSLStrategy(
        debug_mode=True,
        use_fusion_for_long=True,
        atr_length=9,
        hma_mode='VWMA',
        hma_length=50,
        atr_scaling_factor=1.4,
        trade_direction='long',
        initial_equity=10000,
        fee_pct=0.04
    )
    
    print("\n📊 STRATEGY PARAMETERS:")
    print(f"- Debug mode: True")
    print(f"- Use fusion for long: True")
    print(f"- ATR length: 9")
    print(f"- HMA mode: VWMA")
    print(f"- HMA length: 50")
    print(f"- ATR scaling factor: 1.4")
    print(f"- Trade direction: long")
    
    # Prepare signals 
    print("\n🔧 PREPARING SIGNALS...")
    prepared_data = strategy.prepare_signals(df_debug)
    
    # Find the specific timestamp
    target_time = pd.Timestamp('2019-05-03 12:00:00')
    
    if target_time not in prepared_data.index:
        print(f"❌ Target time {target_time} not found in data!")
        print("Available times around May 3:")
        may_3_times = prepared_data[prepared_data.index.date == pd.Timestamp('2019-05-03').date()]
        for time in may_3_times.index:
            print(f"  {time}")
        return
        
    # Get the index of target time
    target_idx = prepared_data.index.get_loc(target_time)
    
    print(f"\n🎯 ANALYZING TIMESTAMP: {target_time}")
    print(f"Index position: {target_idx}")
    
    # Get candle data
    current_candle = prepared_data.iloc[target_idx]
    prev_candle = prepared_data.iloc[target_idx-1] if target_idx > 0 else None
    
    print(f"\n📈 CANDLE DATA:")
    print(f"Open: ${current_candle['price_open']:.2f}")
    print(f"High: ${current_candle['price_high']:.2f}")
    print(f"Low: ${current_candle['price_low']:.2f}")
    print(f"Close: ${current_candle['price_close']:.2f}")
    print(f"Volume: {current_candle['volume_traded']:.2f}")
    
    # Check all entry conditions manually
    print(f"\n🔍 ENTRY CONDITIONS ANALYSIS:")
    
    # 1. Position check
    position_ok = not strategy.in_long_position
    print(f"1. Not in position: {position_ok}")
    
    # 2. Trend condition
    trend_condition = current_candle['exit_kama'] > current_candle['kama2']
    print(f"2. Trend condition (exit_kama > kama2): {trend_condition}")
    print(f"   Exit KAMA: {current_candle['exit_kama']:.8f}")
    print(f"   KAMA2: {current_candle['kama2']:.8f}")
    print(f"   Difference: {current_candle['exit_kama'] - current_candle['kama2']:.8f}")
    
    # 3. Fusion Range Filter
    fusion_ma, fusion_atr, fusion_cond = strategy.fusion_range_filter.calculate(prepared_data)
    fusion_condition = not fusion_cond.iloc[target_idx]  # Inverted: allow when MA <= ATR
    print(f"3. Fusion condition (MA <= ATR): {fusion_condition}")
    print(f"   Fusion MA: {fusion_ma.iloc[target_idx]:.8f}")
    print(f"   Fusion ATR: {fusion_atr.iloc[target_idx]:.8f}")
    print(f"   Original fusion_cond (MA > ATR): {fusion_cond.iloc[target_idx]}")
    print(f"   Inverted condition (allow entry): {fusion_condition}")
    
    # 4. KAMA delta condition
    kama_delta_condition = current_candle['entry_kama_delta'] > current_candle['entry_kama_delta_limit']
    print(f"4. KAMA delta condition: {kama_delta_condition}")
    print(f"   Entry KAMA Delta: {current_candle['entry_kama_delta']:.8f}")
    print(f"   Entry KAMA Delta Limit: {current_candle['entry_kama_delta_limit']:.8f}")
    print(f"   Difference: {current_candle['entry_kama_delta'] - current_candle['entry_kama_delta_limit']:.8f}")
    
    # 5. Overall entry condition
    entry_condition = position_ok and trend_condition and fusion_condition and kama_delta_condition
    print(f"\n🎯 OVERALL ENTRY CONDITION: {entry_condition}")
    print(f"   All conditions met: {all([position_ok, trend_condition, fusion_condition, kama_delta_condition])}")
    
    # 6. Check actual signal generated
    actual_signal = current_candle['long_entry']
    print(f"6. Actual long_entry signal: {actual_signal}")
    
    if entry_condition and not actual_signal:
        print("❌ BUG: All conditions met but no signal generated!")
    elif not entry_condition and actual_signal:
        print("❌ BUG: Signal generated but conditions not met!")
    elif entry_condition and actual_signal:
        print("✅ CORRECT: All conditions met and signal generated!")
    else:
        print("✅ CORRECT: No signal expected and none generated")
    
    # Show previous few candles for context
    print(f"\n📊 CONTEXT - PREVIOUS CANDLES:")
    for i in range(max(0, target_idx-3), target_idx):
        candle_time = prepared_data.index[i]
        candle = prepared_data.iloc[i]
        trend_ok = candle['exit_kama'] > candle['kama2']
        fusion_ok = not fusion_cond.iloc[i]
        kama_ok = candle['entry_kama_delta'] > candle['entry_kama_delta_limit']
        signal = candle['long_entry']
        
        print(f"  {candle_time}: T:{trend_ok} F:{fusion_ok} K:{kama_ok} S:{signal} | Price: ${candle['price_close']:.2f}")
    
    # Show next few candles for context
    print(f"\n📊 CONTEXT - NEXT CANDLES:")
    for i in range(target_idx, min(len(prepared_data), target_idx+4)):
        candle_time = prepared_data.index[i]
        candle = prepared_data.iloc[i]
        trend_ok = candle['exit_kama'] > candle['kama2']
        fusion_ok = not fusion_cond.iloc[i]
        kama_ok = candle['entry_kama_delta'] > candle['entry_kama_delta_limit']
        signal = candle['long_entry']
        
        print(f"  {candle_time}: T:{trend_ok} F:{fusion_ok} K:{kama_ok} S:{signal} | Price: ${candle['price_close']:.2f}")
    
    # Process trades to verify
    print(f"\n🔄 PROCESSING TRADES...")
    trades = strategy.process_chunk(prepared_data)
    
    print(f"\n📈 TRADES FOUND:")
    for i, trade in enumerate(trades):
        entry_time, exit_time, entry_price, exit_price, net_profit, gross_profit, total_fee, direction, exit_reason = trade
        
        # Check if this is our target trade
        if entry_time.date() == pd.Timestamp('2019-05-03').date():
            print(f"  🎯 TARGET TRADE #{i+1}:")
            print(f"     Entry: {entry_time} at ${entry_price:.2f}")
            print(f"     Exit: {exit_time} at ${exit_price:.2f}")
            print(f"     Direction: {direction}")
            if gross_profit is not None:
                print(f"     Gross Profit: ${gross_profit:.2f}")
                print(f"     Net Profit: ${net_profit:.2f}")
            else:
                print(f"     Position still open")
            print(f"     Exit Reason: {exit_reason}")
        else:
            print(f"  Trade #{i+1}: {entry_time} -> {exit_time} ({direction}) Net: ${net_profit:.2f}")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    
if __name__ == "__main__":
    debug_may_3_trade() 