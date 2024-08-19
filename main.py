import sys
import ccxt.async_support as ccxt
import asyncio
import numpy as np
import pandas as pd
from ta.trend import MACD, EMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from datetime import datetime


async def fetch_ohlcv(exchange, symbol, timeframe='1h', limit=100):
    ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def calculate_indicators(df):
    # Existing indicators
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()

    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()

    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi.rsi()

    ema_short = EMAIndicator(close=df['close'], window=10)
    ema_long = EMAIndicator(close=df['close'], window=50)
    df['ema_short'] = ema_short.ema_indicator()
    df['ema_long'] = ema_long.ema_indicator()

    ichimoku = IchimokuIndicator(high=df['high'], low=df['low'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = obv.on_balance_volume()

    # New indicators
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20, window_atr=10)
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_mid'] = kc.keltner_channel_mband()
    df['kc_low'] = kc.keltner_channel_lband()

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    df['di_plus'] = adx.adx_pos()
    df['di_minus'] = adx.adx_neg()

    williams_r = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14)
    df['williams_r'] = williams_r.williams_r()

    cmf = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=20)
    df['cmf'] = cmf.chaikin_money_flow()

    return df


async def find_profitable_scenarios(exchange, symbols):
    profitable_scenarios = []

    for symbol in symbols:
        df = await fetch_ohlcv(exchange, symbol)
        df = calculate_indicators(df)

        last_row = df.iloc[-1]
        current_price = last_row['close']
        opportunity_time = last_row.name.strftime('%Y-%m-%d %H:%M:%S')

        signal_score = 0

        # Existing strategies
        if current_price <= last_row['bb_lower'] * 1.01:
            signal_score += 1
        elif current_price >= last_row['bb_upper'] * 0.99:
            signal_score -= 1

        atr_average = df['atr'].rolling(window=20).mean().iloc[-1]
        if last_row['atr'] > 1.5 * atr_average:
            signal_score += 0.5

        if df['macd'].iloc[-2] < df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
            signal_score += 1
        elif df['macd'].iloc[-2] > df['macd_signal'].iloc[-2] and df['macd'].iloc[-1] < df['macd_signal'].iloc[-1]:
            signal_score -= 1

        if last_row['rsi'] < 30:
            signal_score += 1
        elif last_row['rsi'] > 70:
            signal_score -= 1

        if df['ema_short'].iloc[-2] < df['ema_long'].iloc[-2] and df['ema_short'].iloc[-1] > df['ema_long'].iloc[-1]:
            signal_score += 1
        elif df['ema_short'].iloc[-2] > df['ema_long'].iloc[-2] and df['ema_short'].iloc[-1] < df['ema_long'].iloc[-1]:
            signal_score -= 1

        if current_price > last_row['ichimoku_a'] and current_price > last_row['ichimoku_b']:
            signal_score += 1
        elif current_price < last_row['ichimoku_a'] and current_price < last_row['ichimoku_b']:
            signal_score -= 1

        if last_row['stoch_k'] < 20 and last_row['stoch_d'] < 20:
            signal_score += 1
        elif last_row['stoch_k'] > 80 and last_row['stoch_d'] > 80:
            signal_score -= 1

        obv_sma = df['obv'].rolling(window=20).mean()
        if df['obv'].iloc[-1] > obv_sma.iloc[-1] and df['obv'].iloc[-2] <= obv_sma.iloc[-2]:
            signal_score += 1
        elif df['obv'].iloc[-1] < obv_sma.iloc[-1] and df['obv'].iloc[-2] >= obv_sma.iloc[-2]:
            signal_score -= 1

        # New strategies
        if current_price < last_row['kc_low']:
            signal_score += 1
        elif current_price > last_row['kc_high']:
            signal_score -= 1

        if last_row['adx'] > 25:
            if last_row['di_plus'] > last_row['di_minus']:
                signal_score += 1
            elif last_row['di_minus'] > last_row['di_plus']:
                signal_score -= 1

        if last_row['williams_r'] < -80:
            signal_score += 1
        elif last_row['williams_r'] > -20:
            signal_score -= 1

        if last_row['cmf'] > 0.05:
            signal_score += 1
        elif last_row['cmf'] < -0.05:
            signal_score -= 1

        # Determine overall signal
        if signal_score > 3:
            signal = "Strong Buy"
        elif signal_score > 1:
            signal = "Buy"
        elif signal_score < -3:
            signal = "Strong Sell"
        elif signal_score < -1:
            signal = "Sell"
        else:
            signal = "Neutral"

        profitable_scenarios.append(f"{symbol:<10} {opportunity_time} - Signal: {signal} (Score: {signal_score:.2f})")

    return profitable_scenarios


async def main():
    exchange = ccxt.mexc({'enableRateLimit': True})
    await exchange.load_markets()
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'XTZ/USDT', 'BXX/USDT', 'PEPE/USDT']

    print("Monitoring cryptocurrency markets for profitable scenarios...")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            try:
                scenarios = await find_profitable_scenarios(exchange, symbols)

                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Scan Results:")
                if scenarios:
                    print("Profitable scenarios found:")
                    for scenario in scenarios:
                        print(f"  {scenario}")
                else:
                    print("No profitable scenarios found at the moment.")

                print("\nWaiting for 5 minutes before next scan...")
                await asyncio.sleep(300)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Retrying in 1 minute...")
                await asyncio.sleep(60)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
