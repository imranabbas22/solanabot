import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_historical_data(symbol, exchange_id='kraken', timeframe='1h', days=730):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    # Calculate start time
    start_time = datetime.now() - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    all_ohlcv = []
    limit = 300 # Coinbase has lower limits often
    if exchange_id == 'kraken':
        limit = 720

    print(f"Fetching {symbol} from {exchange_id} since {start_time} (timestamp: {since})...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                print("No data returned.")
                break

            first_date = datetime.fromtimestamp(ohlcv[0][0]/1000)
            last_date = datetime.fromtimestamp(ohlcv[-1][0]/1000)
            print(f"Fetched {len(ohlcv)} candles. From {first_date} to {last_date}")

            # Check if we are getting new data
            if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                 # Overlap or same data, we need to advance since carefully
                 # Find the index where new data starts
                 last_ts = all_ohlcv[-1][0]
                 new_data = [x for x in ohlcv if x[0] > last_ts]
                 if not new_data:
                     print("No new data found in this batch.")
                     break
                 all_ohlcv.extend(new_data)
                 since = new_data[-1][0] + 1
            else:
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1

            # Safety break
            if len(all_ohlcv) > 20000:
                 break

            # If the last fetched candle is close to now, we are done
            if (datetime.now() - last_date).total_seconds() < 3600 * 2:
                print("Reached present time.")
                break

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    symbols_map = {
        'SOL/USDT': 'SOL/USD',
        'ADA/USDT': 'ADA/USD',
        'DASH/USDT': 'DASH/USD',
        'BTC/USDT': 'BTC/USD'
    }

    exchange_id = 'coinbase'

    for usdt_symbol, pair_symbol in symbols_map.items():
        # Coinbase uses different symbols sometimes, e.g. SOL-USD
        # But CCXT standardizes to SOL/USD usually.

        df = fetch_historical_data(pair_symbol, exchange_id=exchange_id, days=730)

        if df.empty or len(df) < 100:
             print(f"Retrying with Kraken for {pair_symbol}...")
             df = fetch_historical_data(pair_symbol, exchange_id='kraken', days=730)

        filename = f"data_{usdt_symbol.replace('/', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} rows to {filename}")
