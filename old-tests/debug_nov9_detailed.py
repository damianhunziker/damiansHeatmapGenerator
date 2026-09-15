#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_utils import fetch_data
from classes.strategies.live_kama_ssl_strategy import LiveKAMASSLStrategy
import pandas as pd
from datetime import datetime

def debug_nov9_signal():
    """
    Detailliertes Debugging für den 9.11.2018 16:00 und 20:00 Zeitpunkt
    """
    
    # Daten mit genügend Vorlauf holen
    print("=== Debugging 9.11.2018 16:00 und 20:00 Short Signal ===")
    
    # Verwende die gleiche fetch_data Funktion wie test.py
    timeframe_data = fetch_data(
        asset="BTCUSDT",
        interval="4h", 
        start_date="2018-01-01",  # Genügend Vorlauf für Aufwärmung
        end_date="2018-12-01"
    )
    
    data = timeframe_data['primary']['data']
    
    print(f"\nDaten geladen: {len(data)} Kerzen")
    print(f"Datumsbereich: {data.index[0]} bis {data.index[-1]}")
    
    # Strategy initialisieren mit den korrekten Parametern
    strategy = LiveKAMASSLStrategy(
        debug_mode=True,
        trade_direction='short',
        calculate_divergences=True,
        use_divergence_exit_long=True,
        use_divergence_exit_short=True
    )
    
    # Indikatoren berechnen
    print("\n=== Berechne Indikatoren ===")
    prepared_data = strategy.calculate_indicators(data)
    
    # Finde die exakten Indizes für 9.11.2018 16:00 und 20:00
    target_dates = [
        pd.Timestamp("2018-11-09 16:00:00"),
        pd.Timestamp("2018-11-09 20:00:00")
    ]
    
    target_indices = []
    for target_date in target_dates:
        try:
            target_idx = prepared_data.index.get_loc(target_date)
            target_indices.append((target_date, target_idx))
            print(f"\nIndex für {target_date}: {target_idx}")
        except KeyError:
            print(f"\nDatum {target_date} nicht gefunden in den Daten")
            # Finde das nächstliegende Datum
            nearest_date = min(prepared_data.index, key=lambda x: abs(x - target_date))
            target_idx = prepared_data.index.get_loc(nearest_date)
            target_indices.append((nearest_date, target_idx))
            print(f"Nächstliegendes Datum: {nearest_date} (Index: {target_idx})")
    
    # Untersuche mehrere Kerzen um beide Zeitpunkte herum
    all_indices = set()
    for target_date, target_idx in target_indices:
        start_idx = max(0, target_idx - 3)
        end_idx = min(len(prepared_data), target_idx + 4)
        all_indices.update(range(start_idx, end_idx))
    
    # Sortiere alle Indizes
    sorted_indices = sorted(all_indices)
    
    print(f"\n=== Untersuchung von Index {min(sorted_indices)} bis {max(sorted_indices)} ===")
    
    for idx in sorted_indices:
        row = prepared_data.iloc[idx]
        timestamp = prepared_data.index[idx]
        
        # Berechne SSL Crossover für diese Kerze
        if idx > 0:
            prev_row = prepared_data.iloc[idx-1]
            
            # SSL Crossover Down: ssl_down kreuzt über ssl_up (für Short Entry)
            ssl_crossover_down = (
                prev_row['ssl_up'] > prev_row['ssl_down'] and  # Vorher: ssl_up war über ssl_down (bullisch)
                row['ssl_down'] >= row['ssl_up']  # Jetzt: ssl_down ist über/gleich ssl_up (bärisch)
            )
            
            # SSL Crossover Up: ssl_up kreuzt über ssl_down (für Short Exit) 
            ssl_crossover_up = (
                prev_row['ssl_down'] > prev_row['ssl_up'] and  # Vorher: ssl_down war über ssl_up (bärisch)
                row['ssl_up'] >= row['ssl_down']  # Jetzt: ssl_up ist über/gleich ssl_down (bullisch)
            )
            
            # Preis unter DEMA
            price_below_dema = row['price_close'] < row['dema']
            
            # KAMA Short Entry Condition
            kama_short_entry_condition = (
                row['entry_kama_delta'] < 0 and 
                abs(row['entry_kama_delta']) > abs(row['exit_kama_delta_limit'])
            )
            
            # Komplette Short Entry Bedingung
            short_entry_condition = ssl_crossover_down and price_below_dema and kama_short_entry_condition
            
        else:
            ssl_crossover_down = False
            ssl_crossover_up = False
            price_below_dema = row['price_close'] < row['dema']
            kama_short_entry_condition = False
            short_entry_condition = False
        
        # Markiere die Ziel-Zeitpunkte
        marker = ""
        for target_date, target_idx in target_indices:
            if timestamp == target_date or idx == target_idx:
                marker = "🎯"
                break
        if not marker:
            marker = "  "
        
        print(f"\n{marker} {timestamp} (Index: {idx})")
        print(f"    Preis: {row['price_close']:.2f}")
        print(f"    SSL Down: {row['ssl_down']:.2f}")
        print(f"    SSL Up: {row['ssl_up']:.2f}")
        print(f"    SSL Status: {'BEARISH' if row['ssl_down'] > row['ssl_up'] else 'BULLISH'}")
        print(f"    DEMA: {row['dema']:.2f}")
        print(f"    Preis unter DEMA: {price_below_dema}")
        print(f"    Entry KAMA Delta: {row['entry_kama_delta']:.6f}")
        print(f"    Exit KAMA Delta Limit: {row['exit_kama_delta_limit']:.6f}")
        print(f"    KAMA Entry Bedingung: {kama_short_entry_condition}")
        
        if idx > 0:
            print(f"    SSL Crossover Down: {ssl_crossover_down}")
            print(f"    SSL Crossover Up: {ssl_crossover_up}")
            print(f"    🎯 SHORT ENTRY SIGNAL: {short_entry_condition}")
            
            if ssl_crossover_down:
                print(f"        ➡️ SSL DOWN CROSSOVER ERKANNT!")
                print(f"           Vorher: SSL_Up={prev_row['ssl_up']:.2f} > SSL_Down={prev_row['ssl_down']:.2f}")
                print(f"           Jetzt:  SSL_Down={row['ssl_down']:.2f} >= SSL_Up={row['ssl_up']:.2f}")
            
            if ssl_crossover_up:
                print(f"        ➡️ SSL UP CROSSOVER ERKANNT!")
                print(f"           Vorher: SSL_Down={prev_row['ssl_down']:.2f} > SSL_Up={prev_row['ssl_up']:.2f}")
                print(f"           Jetzt:  SSL_Up={row['ssl_up']:.2f} >= SSL_Down={row['ssl_down']:.2f}")
        
        print("    " + "="*60)

if __name__ == "__main__":
    debug_nov9_signal() 