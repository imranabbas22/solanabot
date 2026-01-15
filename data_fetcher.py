import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_historical_data(symbol, exchange_id='coinbase', timeframe='1h', days=730):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    start_time = datetime.now() - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)

    all_ohlcv = []
    limit = 300

    print(f"Fetching {symbol} ({timeframe}) from {exchange_id} since {start_time}...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            if all_ohlcv and ohlcv[0][0] <= all_ohlcv[-1][0]:
                 last_ts = all_ohlcv[-1][0]
                 new_data = [x for x in ohlcv if x[0] > last_ts]
                 if not new_data:
                     break
                 all_ohlcv.extend(new_data)
                 since = new_data[-1][0] + 1
            else:
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1

            last_date = datetime.fromtimestamp(ohlcv[-1][0]/1000)
            print(f"Fetched {len(ohlcv)} candles. Last: {last_date}")

            if (datetime.now() - last_date).total_seconds() < 3600:
                break

            if len(all_ohlcv) > 20000:
                 break

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            if "rate limit" in str(e).lower():
                time.sleep(5)
                continue
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

    for usdt_symbol, pair_symbol in symbols_map.items():
        df = fetch_historical_data(pair_symbol, exchange_id='coinbase', timeframe='1h', days=730)
        filename = f"data_{usdt_symbol.replace('/', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} rows to {filename}")
