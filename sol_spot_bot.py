"""
Solana Spot Trading Bot v3 (Python) + AI + BTC Correlation

- Strategy: 1h Timeframe
- Asset: SOL/USDT (watched), BTC/USDT (for correlation)
- New Features:
    1. BTC Correlation: Checks if Bitcoin is dumping before buying SOL.
    2. Dynamic Sizing: Bets larger size only when AI is highly confident.
    3. Feature Importance: Tells you WHICH indicator triggered the trade.
- ML Engine: Random Forest Classifier
    - Retrains every 6 hours.
    - Persists model to 'ai_model_v3.pkl'.

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
        logging.FileHandler("solana_bot_v3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- ML IMPORTS (With Auto-Install) ---
ML_AVAILABLE = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ ML libraries missing. Auto-installing...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "joblib"])
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        ML_AVAILABLE = True
    except Exception:
        logger.error("❌ Auto-install failed. Please run: pip install scikit-learn joblib")

# --- CONFIG START ---
API_KEY = "API_KEY_HERE" # Paste Binance API Key
API_SECRET = "API_SECRET_HERE" # Paste Binance API Secret

# --- TELEGRAM CONFIG ---
TELEGRAM_TOKEN = "TELEGRAM_TOKEN_HERE"   # Paste Token from @BotFather
TELEGRAM_CHAT_ID = "TELEGRAM_CHAT_ID_HERE" # Paste your Chat ID

EXCHANGE_ID = 'binance'
SYMBOL = 'SOL/USDT'
BTC_SYMBOL = 'BTC/USDT' # New: We watch BTC too
TIMEFRAME = '1h'
FETCH_LIMIT = 1000

DRY_RUN = False
POSITION_MODE = 'single'

# --- DYNAMIC RISK MANAGEMENT ---
BASE_BET_SIZE = 0.40 # Default: Use 40% of balance
MAX_BET_SIZE = 0.80  # Max: Use 80% if AI is super confident
MIN_TRADE_USD = 10.0

# Strategy Settings
USE_TREND_FILTER = True
ADX_THRESHOLD = 25
TAKE_PROFIT_ATR_MULT = 3.0
STOP_LOSS_ATR_MULT = 1.5
USE_TRAILING_STOP = True
TRAILING_STOP_ATR_MULT = 2.5

# ML Settings
ML_CONFIDENCE_THRESHOLD = 0.60
ML_RETRAIN_INTERVAL = 6 * 60 * 60

# Indicators
RSI_PERIOD = 14
RSI_SELL_LEVEL = 75
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_SHORT = 50
EMA_LONG = 200
BOLL_PERIOD = 20
BOLL_STD = 2
ATR_PERIOD = 14
ADX_PERIOD = 14

SLEEP_INTERVAL = 15
DEFAULT_STATUS_INTERVAL = 6 * 60 * 60
# --- CONFIG END ---

# --- TELEGRAM KEYBOARD ---
KEYBOARD = {
    "keyboard": [
        [{"text": "✅ Start"}, {"text": "🛑 Stop"}, {"text": "📊 Status"}],
        [{"text": "🧠 Retrain AI"}, {"text": "📉 Force Exit"}],
        [{"text": "🛡️ Safe Mode"}, {"text": "🚀 Aggressive Mode"}],
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

# --- MACHINE LEARNING MODULE ---
class AITrader:
    def __init__(self):
        self.model = None
        self.last_accuracy = 0.0
        self.is_trained = False
        self.top_features = []
        if ML_AVAILABLE: self.load_model()
            
    def load_model(self):
        if os.path.exists('ai_model_v3.pkl'):
            try:
                data = joblib.load('ai_model_v3.pkl')
                self.model = data['model']
                self.top_features = data.get('top_features', [])
                self.is_trained = True
                logger.info("🧠 AI Model v3 loaded.")
            except: pass

    def prepare_data(self, df):
        data = df.copy()
        
        # Check required columns including BTC
        required = ['rsi', 'adx', 'volume', 'close', 'ema_long', 'btc_close']
        if not all(c in data.columns for c in required):
            return pd.DataFrame() 

        # Clean Data
        data = data.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Feature Engineering
        data['feat_rsi'] = data['rsi'] / 100.0
        data['feat_adx'] = data['adx'] / 100.0
        data['feat_vol_change'] = data['volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        data['feat_close_change'] = data['close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        # BTC Correlation Feature (CRITICAL UPGRADE)
        data['feat_btc_change'] = data['btc_close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        data['feat_rel_strength'] = data['feat_close_change'] - data['feat_btc_change'] # Is SOL beating BTC?

        safe_ema = data['ema_long'].replace(0, 1e-9)
        data['feat_ema_dist'] = (data['close'] - data['ema_long']) / safe_ema
        
        # Target: Price rises > 1.5% in next 4 hours
        future_returns = data['close'].shift(-4)
        data['target'] = (future_returns > data['close'] * 1.015).astype(int)
        
        data.dropna(inplace=True)
        return data

    def train(self, df):
        if not ML_AVAILABLE: return False
        logger.info("🧠 Training AI Model...")
        
        data = self.prepare_data(df)
        if data.empty or len(data) < 50: return False
        
        feature_cols = [c for c in data.columns if c.startswith('feat_')]
        X = data[feature_cols]
        y = data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        predictions = self.model.predict(X_test)
        self.last_accuracy = accuracy_score(y_test, predictions)
        self.is_trained = True
        
        # Extract Top Features
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.top_features = [feature_cols[i] for i in indices[:3]] # Save top 3
        
        try:
            joblib.dump({'model': self.model, 'top_features': self.top_features}, 'ai_model_v3.pkl')
        except: pass
        
        logger.info(f"🧠 Training Complete. Accuracy: {self.last_accuracy*100:.2f}%")
        return True

    def get_confidence(self, df):
        if not self.is_trained or not ML_AVAILABLE or self.model is None: return 0.5
        data = self.prepare_data(df)
        if data.empty: return 0.5
        
        feature_cols = [c for c in data.columns if c.startswith('feat_')]
        latest = data[feature_cols].iloc[-1:] 
        
        try:
            if hasattr(self.model, 'classes_'):
                classes = list(self.model.classes_)
                if 1 in classes:
                    class_idx = classes.index(1)
                    return float(self.model.predict_proba(latest)[0][class_idx])
            return 0.5
        except: return 0.5

# --- FORMATTING ---
def format_dashboard(price, ema, rsi, adx, pos_open, interval, ai_acc, ai_conf, top_feats):
    trend = "🟢 Bull" if price > ema else "🔴 Bear"
    ai_emoji = "🧠" if ai_conf > 0.6 else "🤖"
    
    # Format top features nicely
    feat_str = ", ".join([f.replace('feat_', '') for f in top_feats]) if top_feats else "Gathering Data..."

    msg = (
        f"📊 <b>SOLANA BOT v3 (SMART)</b>\n\n"
        f"<b>Price:</b> ${price:.2f}\n"
        f"<b>Trend:</b> {trend} | <b>RSI:</b> {rsi:.1f}\n"
        f"<b>ADX:</b> {adx:.1f}\n\n"
        f"<b>AI Brain Scan:</b>\n"
        f"• Accuracy: {ai_acc*100:.1f}%\n"
        f"• Confidence: {ai_conf*100:.1f}% {ai_emoji}\n"
        f"• Focus: <i>{feat_str}</i>\n\n"
        f"<b>Status:</b> {'✅ In Trade' if pos_open else '🔭 Scanning'}"
    )
    return msg

# --- MATH FUNCTIONS ---
def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()
def rsi(s, p=14):
    d = s.diff()
    g, l = d.clip(lower=0), -1 * d.clip(upper=0)
    avg_g = g.ewm(span=p, adjust=False).mean()
    avg_l = l.ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

def atr(df, p=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def adx(df, p=14):
    p_dm = df['high'].diff()
    m_dm = df['low'].diff()
    p_dm = p_dm.where((p_dm > m_dm) & (p_dm > 0), 0.0)
    m_dm = m_dm.where((m_dm > p_dm) & (m_dm > 0), 0.0)
    tr = atr(df, p).replace(0, 1e-9)
    p_di = 100 * (p_dm.ewm(alpha=1/p).mean() / tr)
    m_di = 100 * (m_dm.ewm(alpha=1/p).mean() / tr)
    dx = (abs(p_di - m_di) / (p_di + m_di).replace(0, 1e-9)) * 100
    return dx.ewm(alpha=1/p).mean(), p_di, m_di

# --- EXCHANGE HELPERS ---
def create_exchange(key, secret):
    return getattr(ccxt, EXCHANGE_ID)({'apiKey': key, 'secret': secret, 'enableRateLimit': True})

def fetch_data_merged(exchange, symbol, btc_symbol, timeframe, limit):
    # Fetch SOL
    sol_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(sol_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Fetch BTC
    btc_ohlcv = exchange.fetch_ohlcv(btc_symbol, timeframe, limit=limit)
    df_btc = pd.DataFrame(btc_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc.set_index('timestamp', inplace=True)
    
    # Merge BTC close price into SOL dataframe
    df['btc_close'] = df_btc['close']
    return df.astype(float).fillna(method='ffill')

def compute_indicators(df):
    df = df.copy()
    df['ema_long'] = ema(df['close'], EMA_LONG)
    df['rsi'] = rsi(df['close'], RSI_PERIOD)
    
    # MACD
    ef = ema(df['close'], MACD_FAST)
    es = ema(df['close'], MACD_SLOW)
    df['macd_hist'] = (ef - es) - ema(ef - es, MACD_SIGNAL)
    
    # BB
    mid = sma(df['close'], BOLL_PERIOD)
    std = df['close'].rolling(BOLL_PERIOD).std()
    df['bb_lower'] = mid - BOLL_STD * std
    df['bb_upper'] = mid + BOLL_STD * std
    df['bb_mid'] = mid # Needed for feature engineering
    
    df['atr'] = atr(df, ATR_PERIOD)
    df['adx'], df['p_di'], df['m_di'] = adx(df, ADX_PERIOD)
    return df

def generate_signal(df):
    if len(df) < 200: return None
    row, prev = df.iloc[-1], df.iloc[-2]
    
    # 1. Trend Filter
    if USE_TREND_FILTER and row['close'] < row['ema_long']: return 'bearish_skip'
    
    # 2. ADX Filter (Crash Protection)
    if row['adx'] > ADX_THRESHOLD and row['m_di'] > row['p_di']: return 'crash_skip'

    # 3. Triggers
    rsi_buy = prev['rsi'] < 50 and row['rsi'] > prev['rsi'] + 2
    macd_buy = row['macd_hist'] > prev['macd_hist'] and row['macd_hist'] > -0.5
    bb_buy = prev['low'] <= prev['bb_lower'] and row['close'] > row['bb_lower']

    if row['rsi'] > RSI_SELL_LEVEL: return 'sell'
    if rsi_buy and (macd_buy or bb_buy): return 'buy'
    return None

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

# --- POSITION CLASS ---
class Position:
    def __init__(self, filename='bot_state_v3.json'):
        self.filename = filename
        self.open = False
        self.entry_price = 0.0
        self.amount = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.highest_price = 0.0
        self.load()

    def save(self):
        with open(self.filename, 'w') as f: 
            json.dump(self.__dict__, f)

    def load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.__dict__.update(json.load(f))
            except: pass

    def open_pos(self, price, amount, atr_val):
        self.open = True
        self.entry_price = float(price)
        self.amount = float(amount)
        self.highest_price = float(price)
        self.stop_loss = self.entry_price - (atr_val * STOP_LOSS_ATR_MULT)
        self.take_profit = self.entry_price + (atr_val * TAKE_PROFIT_ATR_MULT)
        self.save()

    def update_sl(self, price, atr_val):
        if not self.open: return False
        if price > self.highest_price:
            self.highest_price = price
            if USE_TRAILING_STOP:
                new_sl = self.highest_price - (atr_val * TRAILING_STOP_ATR_MULT)
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    self.save()
                    return True
        return False

    def close_pos(self):
        self.open = False
        self.save()

# --- MAIN LOOP ---
def main():
    exchange = create_exchange(API_KEY, API_SECRET)
    pos = Position()
    ai = AITrader()
    
    curr_interval = DEFAULT_STATUS_INTERVAL
    bot_running = True
    last_train = 0
    last_status = time.time()
    last_upd_id = 0
    
    # Initialize runtime variables with config defaults
    current_bet_size = BASE_BET_SIZE
    
    send_telegram_message(f"🤖 <b>Solana Bot v3 (Correlated)</b>\nWaiting for data...")

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
                elif 'retrain' in text: last_train = 0
                elif 'force exit' in text:
                    if pos.open:
                        send_telegram_message("📉 <b>FORCE EXIT TRIGGERED</b>")
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                    else:
                        send_telegram_message("ℹ️ No position to exit.")
                elif 'safe mode' in text:
                    current_bet_size = 0.25
                    send_telegram_message("🛡️ <b>Safe Mode Activated</b>\nBase bet size: 25%")
                elif 'aggressive mode' in text:
                    current_bet_size = 0.80
                    send_telegram_message("🚀 <b>Aggressive Mode Activated</b>\nBase bet size: 80%")
                elif '15m' in text:
                    curr_interval = 15 * 60
                    send_telegram_message("⏱ Updates: Every 15m")
                elif '1h' in text:
                    curr_interval = 60 * 60
                    send_telegram_message("⏱ Updates: Every 1h")
                elif '4h' in text:
                    curr_interval = 4 * 60 * 60
                    send_telegram_message("⏱ Updates: Every 4h")
                elif '6h' in text:
                    curr_interval = 6 * 60 * 60
                    send_telegram_message("⏱ Updates: Every 6h")
                    
        except: pass

        if bot_running:
            try:
                # 2. Fetch Data (BTC + SOL)
                df = fetch_data_merged(exchange, SYMBOL, BTC_SYMBOL, TIMEFRAME, FETCH_LIMIT)
                if df.empty: 
                    time.sleep(SLEEP_INTERVAL)
                    continue
                
                # 3. Retrain AI
                if time.time() - last_train > ML_RETRAIN_INTERVAL:
                    df_indi = compute_indicators(df)
                    ai.train(df_indi)
                    last_train = time.time()
                    if ai.is_trained:
                        feats = ", ".join([f.replace('feat_', '') for f in ai.top_features])
                        send_telegram_message(f"🧠 <b>AI Retrained</b>\nAccuracy: {ai.last_accuracy*100:.1f}%\nFocusing on: {feats}")

                # 4. Analyze
                df = compute_indicators(df)
                signal = generate_signal(df)
                conf = ai.get_confidence(df)
                
                row = df.iloc[-1]
                
                if pos.open:
                    # Manage Trade
                    pos.update_sl(row['close'], row['atr'])
                    pnl = (row['close'] - pos.entry_price) / pos.entry_price
                    
                    if row['close'] <= pos.stop_loss:
                        send_telegram_message(f"🛑 <b>STOP LOSS</b>\nPnL: {pnl*100:.2f}%")
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                    elif row['close'] >= pos.take_profit:
                        send_telegram_message(f"🚀 <b>TAKE PROFIT</b>\nPnL: {pnl*100:.2f}%")
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                    elif signal == 'sell':
                        send_telegram_message(f"📉 <b>EXIT (Indicator)</b>\nPnL: {pnl*100:.2f}%")
                        place_order(exchange, 'sell', pos.amount, DRY_RUN)
                        pos.close_pos()
                        
                else:
                    # Look for Entry
                    if signal == 'buy':
                        if conf >= ML_CONFIDENCE_THRESHOLD:
                            # DYNAMIC SIZING LOGIC
                            # If Conf < 70%, use current Base Size. If Conf > 80%, Scale up.
                            size_mult = current_bet_size
                            if conf > 0.8: 
                                # Cap scaling at MAX_BET_SIZE
                                size_mult = min(current_bet_size * 1.5, MAX_BET_SIZE)
                            
                            bal = usd_balance_of(exchange)
                            usd_amount = bal * size_mult
                            
                            if usd_amount > MIN_TRADE_USD:
                                amount = usd_amount / row['close']
                                place_order(exchange, 'buy', amount, DRY_RUN)
                                pos.open_pos(row['close'], amount, row['atr'])
                                
                                send_telegram_message(
                                    f"✅ <b>BUY SOL</b>\n"
                                    f"AI Conf: {conf*100:.1f}% (Size: {size_mult*100:.0f}%)\n"
                                    f"Stop: ${pos.stop_loss:.2f}"
                                )
                        else:
                            logger.info(f"Signal BLOCKED by AI (Conf {conf:.2f})")
                
                # 5. Dashboard
                if time.time() - last_status > curr_interval:
                    msg = format_dashboard(row['close'], row['ema_long'], row['rsi'], row['adx'], pos.open, curr_interval, ai.last_accuracy, conf, ai.top_features)
                    send_telegram_message(msg)
                    last_status = time.time()
                    
            except Exception as e:
                logger.error(f"Loop Error: {e}")
        
        time.sleep(SLEEP_INTERVAL)

if __name__ == '__main__':
    main()
