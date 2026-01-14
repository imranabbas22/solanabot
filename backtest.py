
import pandas as pd
import numpy as np
import logging
import argparse
from sol_spot_bot import compute_indicators, generate_signal, AITrader

# Configure logging for backtest
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("backtest")

class Backtester:
    def __init__(self, data_path, btc_data_path, initial_balance=300):
        self.data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        self.btc_data = pd.read_csv(btc_data_path, parse_dates=['timestamp'], index_col='timestamp')

        # Merge BTC data
        self.data = self.data.join(self.btc_data['close'].rename('btc_close'), how='inner')

        self.initial_balance = initial_balance
        self.balance = initial_balance # Cash available
        self.current_position = None # None or object
        self.trades = []

        self.FEE_RATE = 0.001

        # DCA Settings
        self.CHUNK_SIZE_PCT = 0.20 # 20% of Initial Balance (fixed) or Current Balance?
                                   # User said "20% percentage of total wallet".
                                   # If we use Total Equity, it changes. Let's use Fixed 20% of Initial for simplicity, or 20% of Remaining Cash?
                                   # "20% of total wallet" usually means 1/5th of the capital allocated to this bot.
                                   # So if we have $300, each chunk is $60.
        self.MAX_CHUNKS = 5
        self.DCA_DROP_PCT = 0.05 # Buy every 5% drop
        self.TAKE_PROFIT_PCT = 0.025 # Sell at 2.5% profit above AVG entry

    def run(self):
        logger.info(f"Starting backtest with ${self.initial_balance}...")
        self.data = compute_indicators(self.data)
        self.data.dropna(inplace=True)

        total_steps = len(self.data)
        logger.info(f"Total steps: {total_steps}")

        timestamps = self.data.index
        start_index = 200

        for i in range(start_index, total_steps):
            current_time = timestamps[i]
            row = self.data.iloc[i]
            prev = self.data.iloc[i-1]

            self.execute_strategy(row, prev, current_time)

        # End of backtest: Value the bag at current price
        self.close_at_end(self.data.iloc[-1])
        self.report()

    def execute_strategy(self, row, prev, timestamp):
        price = row['close']

        if self.current_position is None:
            # Entry Condition: RSI Dip (< 40) or just Entry?
            # User said "Aggressive DCA... Buy the dip".
            # Let's wait for a dip (RSI < 40) to start the first chunk.
            if row['rsi'] < 40:
                self.buy_chunk(price, timestamp, "Initial Buy (RSI < 40)")
        else:
            pos = self.current_position

            # 1. Check Exit (Profit)
            # Sell if Price > Avg Entry * (1 + TP)
            target_price = pos.avg_price * (1 + self.TAKE_PROFIT_PCT)
            if price > target_price:
                self.sell_all(price, timestamp, "Take Profit")
                return

            # 2. Check DCA Buy (Dip)
            # Buy if Price < Last Buy Price * (1 - DCA_DROP)
            # And we haven't maxed out chunks
            if pos.chunks_count < self.MAX_CHUNKS:
                dca_target = pos.last_buy_price * (1 - self.DCA_DROP_PCT)
                if price < dca_target:
                    self.buy_chunk(price, timestamp, f"DCA Buy #{pos.chunks_count + 1} (-{(1-price/pos.last_buy_price)*100:.1f}%)")

    def buy_chunk(self, price, timestamp, reason):
        # Calculate amount to spend: 20% of Initial Balance ($300 * 0.2 = $60)
        usd_to_spend = self.initial_balance * self.CHUNK_SIZE_PCT

        if self.balance < usd_to_spend * 0.9: # Check if we have enough cash (allow small margin)
             # Cannot buy, maybe fully invested
             return

        # Fee
        fee = usd_to_spend * self.FEE_RATE
        net_usd = usd_to_spend - fee
        amount = net_usd / price

        self.balance -= usd_to_spend

        if self.current_position is None:
            self.current_position = DCAPosition(price, amount, usd_to_spend, timestamp)
        else:
            self.current_position.add_chunk(price, amount, usd_to_spend)

        # logger.info(f"{timestamp} | {reason} | Price: {price:.2f} | Avg: {self.current_position.avg_price:.2f} | Chunks: {self.current_position.chunks_count}")

    def sell_all(self, price, timestamp, reason):
        pos = self.current_position

        gross_usd = pos.total_amount * price
        fee = gross_usd * self.FEE_RATE
        net_usd = gross_usd - fee

        self.balance += net_usd

        net_pnl = net_usd - pos.total_cost
        pnl_pct = (net_pnl / pos.total_cost) * 100

        self.trades.append({
            'entry_time': pos.start_time,
            'exit_time': timestamp,
            'exit_price': price,
            'avg_entry': pos.avg_price,
            'amount': pos.total_amount,
            'chunks': pos.chunks_count,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'balance': self.balance
        })

        # logger.info(f"{timestamp} | {reason} | Sold at {price:.2f} | Profit: {pnl_pct:.2f}% | Bal: {self.balance:.2f}")
        self.current_position = None

    def close_at_end(self, row):
        if self.current_position:
            self.sell_all(row['close'], row.name, "End of Backtest")

    def report(self):
        days = (self.data.index[-1] - self.data.index[0]).days
        years = days / 365.0

        total_pnl = self.balance - self.initial_balance
        pnl_pct = (total_pnl / self.initial_balance) * 100

        initial_price = self.data.iloc[0]['close']
        final_price = self.data.iloc[-1]['close']
        bnh_return = ((final_price - initial_price) / initial_price) * 100

        wins = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        # Calculate Max Drawdown (Approximate based on closed trades balance)
        # Real drawdown should check open equity, but let's stick to balance curve for now or just report final.

        print("\n" + "="*30)
        print(f"BACKTEST RESULTS ({days} days) - DCA STRATEGY")
        print("="*30)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance:   ${self.balance:.2f}")
        print(f"Total Profit:    ${total_pnl:.2f} ({pnl_pct:.2f}%)")
        print(f"Annualized Return: {pnl_pct/years:.2f}%")
        print(f"Buy & Hold Return: {bnh_return:.2f}%")
        print(f"Total Trades (Cycles): {len(self.trades)}")
        print(f"Win Rate:        {win_rate*100:.2f}%")
        print("="*30 + "\n")

class DCAPosition:
    def __init__(self, price, amount, cost, timestamp):
        self.total_amount = amount
        self.total_cost = cost
        self.avg_price = price
        self.last_buy_price = price
        self.chunks_count = 1
        self.start_time = timestamp

    def add_chunk(self, price, amount, cost):
        self.total_amount += amount
        self.total_cost += cost
        self.avg_price = self.total_cost / self.total_amount # Weighted Average
        self.last_buy_price = price
        self.chunks_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('btc_data_path', type=str)
    args = parser.parse_args()

    backtester = Backtester(args.data_path, args.btc_data_path)
    backtester.run()
