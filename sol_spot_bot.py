"""
Solana Spot Trading Bot v4 (DCA Strategy)

- Strategy: No Stop Loss, Aggressive DCA.
- Asset: SOL/USDT
- Logic:
    1. Initial Buy: RSI < 40.
    2. DCA: Buy more if price drops 5% below last buy.
    3. Exit: Sell all when price > Avg Entry + 2.5%.
- Risk: Bag Holding (No Stop Loss).
- Sizing: 20% of initial capital per chunk (Max 5 chunks).

Requirements:
- pip install ccxt pandas numpy requests scikit-learn joblib
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
import subprocess

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("solana_bot_v4.log"),
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
BTC_SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
FETCH_LIMIT = 1000

DRY_RUN = False

# --- DCA SETTINGS ---
CHUNK_SIZE_PCT = 0.20 # 20% of Initial Balance
MAX_CHUNKS = 5
DCA_DROP_PCT = 0.05 # 5% drop to buy more
TAKE_PROFIT_PCT = 0.025 # 2.5% profit target
INITIAL_CAPITAL = 300.0 # Used for calculating chunk size if balance fluctuates

# Indicators
RSI_PERIOD = 14
SLEEP_INTERVAL = 15
DEFAULT_STATUS_INTERVAL = 6 * 60 * 60
# --- CONFIG END ---

# --- TELEGRAM KEYBOARD ---
KEYBOARD = {
    "keyboard": [
        [{"text": "✅ Start"}, {"text": "🛑 Stop"}, {"text": "📊 Status"}],
        [{"text": "📉 Force Exit"}, {"text": "💰 Balance"}],
        [{"text": "⏱ 15m"}, {"text": "⏱ 1h"}, {"text": "⏱ 4h"}, {"text": "⏱ 6h"}]
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
def rsi(s, p=14):
    d = s.diff()
    g, l = d.clip(lower=0), -1 * d.clip(upper=0)
    avg_g = g.ewm(span=p, adjust=False).mean()
    avg_l = l.ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

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
    df['rsi'] = rsi(df['close'], RSI_PERIOD)
    return df

def usd_balance_of(exchange):
    if DRY_RUN: return 10000.0
    try: return float(exchange.fetch_balance().get('free', {}).get('USDT', 0.0))
    except: return 0.0

def place_order(exchange, side, amount, dry_run):
    if dry_run: return {'price': exchange.fetch_ticker(SYMBOL)['last']}
    try: return exchange.create_order(SYMBOL, 'market', side, amount)
    except Exception as e:
        logger.error(f"Order Fail: {e}")
        return None

# --- POSITION CLASS (DCA) ---
class DCAPosition:
    def __init__(self, filename='bot_state_v4.json'):
        self.filename = filename
        self.open = False
        self.avg_price = 0.0
        self.total_amount = 0.0
        self.total_cost = 0.0
        self.last_buy_price = 0.0
        self.chunks_count = 0
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

    def add_chunk(self, price, amount, cost):
        self.open = True
        self.total_amount += float(amount)
        self.total_cost += float(cost)
        self.avg_price = self.total_cost / self.total_amount
        self.last_buy_price = float(price)
        self.chunks_count += 1
        self.save()

    def close_pos(self):
        self.open = False
        self.avg_price = 0.0
        self.total_amount = 0.0
        self.total_cost = 0.0
        self.last_buy_price = 0.0
        self.chunks_count = 0
        self.save()

# --- MAIN LOOP ---
def main():
    exchange = create_exchange(API_KEY, API_SECRET)
    pos = DCAPosition()
    
    curr_interval = DEFAULT_STATUS_INTERVAL
    bot_running = True
    last_status = time.time()
    last_upd_id = 0
    
    send_telegram_message(f"🤖 <b>Solana Bot v4 (DCA)</b>\nWaiting for data...")

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
                        place_order(exchange, 'sell', pos.total_amount, DRY_RUN)
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
                if df.empty: 
                    time.sleep(SLEEP_INTERVAL)
                    continue
                
                df = compute_indicators(df)
                row = df.iloc[-1]
                price = row['close']
                rsi_val = row['rsi']

                # --- STRATEGY LOGIC ---
                
                if pos.open:
                    # 1. Check Take Profit
                    target_price = pos.avg_price * (1 + TAKE_PROFIT_PCT)
                    if price > target_price:
                        # SELL
                        pnl_pct = (price - pos.avg_price) / pos.avg_price
                        profit_usd = (price - pos.avg_price) * pos.total_amount

                        send_telegram_message(
                            f"🚀 <b>TAKE PROFIT</b>\n"
                            f"Price: ${price:.2f}\n"
                            f"Avg Entry: ${pos.avg_price:.2f}\n"
                            f"Profit: {pnl_pct*100:.2f}% (${profit_usd:.2f})\n"
                            f"Chunks: {pos.chunks_count}"
                        )
                        place_order(exchange, 'sell', pos.total_amount, DRY_RUN)
                        pos.close_pos()
                        
                    # 2. Check DCA Buy
                    elif pos.chunks_count < MAX_CHUNKS:
                        dca_target = pos.last_buy_price * (1 - DCA_DROP_PCT)
                        if price < dca_target:
                            # BUY MORE
                            usd_to_spend = INITIAL_CAPITAL * CHUNK_SIZE_PCT
                            bal = usd_balance_of(exchange)
                            if bal > usd_to_spend * 0.9:
                                amount = usd_to_spend / price
                                place_order(exchange, 'buy', amount, DRY_RUN)
                                pos.add_chunk(price, amount, usd_to_spend)
                                
                                send_telegram_message(
                                    f"📉 <b>DCA BUY #{pos.chunks_count}</b>\n"
                                    f"Price: ${price:.2f}\n"
                                    f"Drop: {(1-price/pos.last_buy_price)*100:.1f}%\n"
                                    f"New Avg: ${pos.avg_price:.2f}"
                                )
                
                else:
                    # Initial Entry
                    if rsi_val < 40:
                        usd_to_spend = INITIAL_CAPITAL * CHUNK_SIZE_PCT
                        bal = usd_balance_of(exchange)
                        if bal > usd_to_spend * 0.9:
                            amount = usd_to_spend / price
                            place_order(exchange, 'buy', amount, DRY_RUN)
                            pos.add_chunk(price, amount, usd_to_spend)

                            send_telegram_message(
                                f"✅ <b>INITIAL BUY</b>\n"
                                f"Price: ${price:.2f}\n"
                                f"RSI: {rsi_val:.1f}\n"
                                f"Size: ${usd_to_spend:.2f}"
                            )

                # Status Update
                if time.time() - last_status > curr_interval:
                    status = "✅ In Trade" if pos.open else "🔭 Scanning"
                    pnl_str = ""
                    if pos.open:
                        curr_pnl = (price - pos.avg_price) / pos.avg_price * 100
                        pnl_str = f"\nCurrent PnL: {curr_pnl:.2f}% (Chunks: {pos.chunks_count})"

                    msg = (
                        f"📊 <b>STATUS UPDATE</b>\n"
                        f"Price: ${price:.2f} | RSI: {rsi_val:.1f}\n"
                        f"State: {status}{pnl_str}"
                    )
                    send_telegram_message(msg)
                    last_status = time.time()
                    
            except Exception as e:
                logger.error(f"Loop Error: {e}")
        
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()
