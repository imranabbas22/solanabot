"""
Solana Spot Trading Bot v5 (Swing Trend)

- Strategy: Long-Term Trend Following (Swing)
- Asset: SOL/USDT (Configurable)
- Logic:
    1. Trend: EMA 800 (approx 33 days on 1h timeframe).
    2. Buy: Close > EMA 800.
    3. Sell: Close < EMA 800.
- Risk: No Stop Loss (Trend Reversal Exit).
- Sizing: 100% Equity (Compounding).

Requirements:
- pip install ccxt pandas numpy requests
"""

import time
import logging
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import json
import os
import requests
import sys

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("solana_bot_v5.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG START ---
API_KEY = "API_KEY_HERE"
API_SECRET = "API_SECRET_HERE"

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "TELEGRAM_TOKEN_HERE"
TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID_HERE"

EXCHANGE_ID = 'binance'
SYMBOL = 'SOL/USDT'
TIMEFRAME = '1h'
FETCH_LIMIT = 1000 # Need > 800 for EMA

DRY_RUN = False

# --- STRATEGY SETTINGS ---
EMA_LONG_PERIOD = 800
SLEEP_INTERVAL = 60 # Check every minute
DEFAULT_STATUS_INTERVAL = 6 * 60 * 60
# --- CONFIG END ---

# --- TELEGRAM KEYBOARD ---
KEYBOARD = {
    "keyboard": [
        [{"text": "✅ Start"}, {"text": "🛑 Stop"}, {"text": "📊 Status"}],
        [{"text": "📉 Force Exit"}, {"text": "💰 Balance"}]
    ],
    "resize_keyboard": True
}

def send_telegram_message(message, use_keyboard=True):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        if use_keyboard: payload["reply_markup"] = json.dumps(KEYBOARD)
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Telegram Error: {e}")

def get_telegram_updates(offset=None):
    if not TELEGRAM_TOKEN: return []
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        params = {"timeout": 5, "offset": offset}
        response = requests.get(url, params=params, timeout=10)
        if response.json().get("ok"): return response.json().get("result", [])
    except: pass
    return []

# --- MATH FUNCTIONS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()

# --- EXCHANGE HELPERS ---
def create_exchange(key, secret):
    return getattr(ccxt, EXCHANGE_ID)({'apiKey': key, 'secret': secret, 'enableRateLimit': True})

def fetch_data(exchange, symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df.astype(float)

def compute_indicators(df):
    df = df.copy()
    df['ema_long'] = ema(df['close'], EMA_LONG_PERIOD)
    return df

def usd_balance_of(exchange):
    if DRY_RUN: return 10000.0
    try: return float(exchange.fetch_balance().get('free', {}).get('USDT', 0.0))
    except: return 0.0

def sol_balance_of(exchange):
    if DRY_RUN: return 0.0
    try:
        currency = SYMBOL.split('/')[0]
        return float(exchange.fetch_balance().get('free', {}).get(currency, 0.0))
    except: return 0.0

def place_order(exchange, side, amount, dry_run):
    if dry_run: return {'price': exchange.fetch_ticker(SYMBOL)['last']}
    try: return exchange.create_order(SYMBOL, 'market', side, amount)
    except Exception as e:
        logger.error(f"Order Fail: {e}")
        return None

# --- POSITION CLASS ---
class Position:
    def __init__(self, filename='bot_state_v5.json'):
        self.filename = filename
        self.open = False
        self.entry_price = 0.0
        self.amount = 0.0
        self.load()

    def save(self):
        with open(self.filename, 'w') as f: 
            json.dump(self.__dict__, f)

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.__dict__.update(data)
            except: pass

    def open_pos(self, price, amount):
        self.open = True
        self.entry_price = float(price)
        self.amount = float(amount)
        self.save()

    def close_pos(self):
        self.open = False
        self.entry_price = 0.0
        self.amount = 0.0
        self.save()

# --- MAIN LOOP ---
def main():
    exchange = create_exchange(API_KEY, API_SECRET)
    pos = Position()
    
    curr_interval = DEFAULT_STATUS_INTERVAL
    bot_running = True
    last_status = time.time()
    last_upd_id = 0
    
    send_telegram_message(f"🤖 <b>Solana Bot v5 (Swing Trend)</b>\nEMA {EMA_LONG_PERIOD} | Waiting for data...")

    # Init telegram offset
    updates = get_telegram_updates()
    if updates: last_upd_id = updates[-1]['update_id']

    while True:
        # 1. Telegram Commands
        try:
            for upd in get_telegram_updates(last_upd_id + 1):
                last_upd_id = upd['update_id']
                text = upd.get('message', {}).get('text', '').lower()
                
                if 'stop' in text: 
                    bot_running = False
                    send_telegram_message("🛑 Bot PAUSED")
                elif 'start' in text:
                    bot_running = True
                    send_telegram_message("✅ Bot RESUMED")
                elif 'status' in text: last_status = 0
                elif 'force exit' in text:
                    if pos.open:
                        send_telegram_message("📉 <b>FORCE EXIT TRIGGERED</b>")
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                    else:
                        send_telegram_message("ℹ️ No position to exit.")
                elif 'balance' in text:
                    bal = usd_balance_of(exchange)
                    send_telegram_message(f"💰 Balance: ${bal:.2f}")
                    
        except: pass

        if bot_running:
            try:
                df = fetch_data(exchange, SYMBOL, TIMEFRAME, FETCH_LIMIT)
                if df.empty or len(df) < EMA_LONG_PERIOD:
                    logger.warning("Not enough data for EMA calculation.")
                    time.sleep(SLEEP_INTERVAL)
                    continue
                
                df = compute_indicators(df)
                row = df.iloc[-1]
                price = row['close']
                ema_val = row['ema_long']

                # --- STRATEGY LOGIC ---
                # Buy if Price > EMA
                # Sell if Price < EMA

                trend = "BULL" if price > ema_val else "BEAR"
                
                if pos.open:
                    # Check Exit (Trend Reversal)
                    if price < ema_val:
                        # SELL SIGNAL
                        pnl_pct = (price - pos.entry_price) / pos.entry_price
                        profit_usd = (price - pos.entry_price) * pos.amount

                        send_telegram_message(
                            f"📉 <b>TREND REVERSAL (SELL)</b>\n"
                            f"Price: ${price:.2f} < EMA: ${ema_val:.2f}\n"
                            f"Profit: {pnl_pct*100:.2f}% (${profit_usd:.2f})"
                        )
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                        
                else:
                    # Check Entry (Trend Start)
                    # We check if we are in Bull Trend.
                    # Ideally we check for crossover (prev candle < EMA), but for robustness (if bot restarted),
                    # we can just buy if Price > EMA and we are not in position.

                    if price > ema_val:
                        # BUY SIGNAL
                        bal = usd_balance_of(exchange)
                        if bal > 10.0: # Min Trade
                            amount = bal / price
                            place_order(exchange, 'buy', amount, DRY_RUN)
                            pos.open_pos(price, amount)

                            send_telegram_message(
                                f"✅ <b>TREND START (BUY)</b>\n"
                                f"Price: ${price:.2f} > EMA: ${ema_val:.2f}\n"
                                f"Size: ${bal:.2f}"
                            )

                # Status Update
                if time.time() - last_status > curr_interval:
                    status = "✅ In Trade" if pos.open else "🔭 Scanning"
                    pnl_str = ""
                    if pos.open:
                        curr_pnl = (price - pos.entry_price) / pos.entry_price * 100
                        pnl_str = f"\nPnL: {curr_pnl:.2f}%"

                    msg = (
                        f"📊 <b>STATUS UPDATE</b>\n"
                        f"Price: ${price:.2f} | EMA: ${ema_val:.2f}\n"
                        f"Trend: {trend}\n"
                        f"State: {status}{pnl_str}"
                    )
                    send_telegram_message(msg)
                    last_status = time.time()
                    
            except Exception as e:
                logger.error(f"Loop Error: {e}")
        
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()
