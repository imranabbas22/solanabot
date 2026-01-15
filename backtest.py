
import pandas as pd
import numpy as np
import logging
import argparse

# Configure logging for backtest
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("backtest")

class Backtester:
    def __init__(self, data_path, ema_length=800, initial_balance=300):
        self.data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        self.data.sort_index(inplace=True)

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.current_position = None
        self.trades = []

        # Strategy Parameters
        self.BASE_BET_SIZE = 1.0 # 100% Equity Compounding
        self.MIN_TRADE_USD = 10.0
        self.FEE_RATE = 0.001 # 0.1% Taker Fee
        self.EMA_LENGTH = ema_length

    def run(self):
        logger.info(f"Starting backtest with ${self.initial_balance} (EMA {self.EMA_LENGTH})...")

        # Calculate EMA
        self.data['ema_trend'] = self.data['close'].ewm(span=self.EMA_LENGTH, adjust=False).mean()
        self.data.dropna(inplace=True)

        total_steps = len(self.data)
        logger.info(f"Total steps: {total_steps}")

        timestamps = self.data.index
        start_index = 10

        for i in range(start_index, total_steps):
            current_time = timestamps[i]
            row = self.data.iloc[i]
            prev = self.data.iloc[i-1]

            signal = self._optimized_generate_signal(row, prev)
            self.execute_strategy(row, signal, current_time)

        self.close_all_positions(self.data.iloc[-1])
        self.report()

    def _optimized_generate_signal(self, row, prev):
        # STRATEGY: EMA Trend Follower

        # Buy if Close > EMA
        if row['close'] > row['ema_trend'] and prev['close'] <= prev['ema_trend']:
             return 'buy'

        # Sell if Close < EMA
        if row['close'] < row['ema_trend'] and prev['close'] >= prev['ema_trend']:
             return 'sell'

        return None

    def execute_strategy(self, row, signal, timestamp):
        price = row['close']

        if self.current_position:
            pos = self.current_position

            exit_reason = None
            if signal == 'sell':
                exit_reason = "Trend Reversal"

            if exit_reason:
                gross_usd = pos.amount * price
                fee = gross_usd * self.FEE_RATE
                net_usd = gross_usd - fee
                self.balance = net_usd

                net_pnl = net_usd - pos.cost_basis

                self.trades.append({
                    'entry_time': pos.entry_time,
                    'exit_time': timestamp,
                    'entry_price': pos.entry_price,
                    'exit_price': price,
                    'amount': pos.amount,
                    'pnl': net_pnl,
                    'pnl_pct': (net_usd - pos.cost_basis) / pos.cost_basis,
                    'reason': exit_reason,
                    'balance': self.balance
                })
                self.current_position = None

        else:
            if signal == 'buy':
                usd_to_spend = self.balance * self.BASE_BET_SIZE
                if usd_to_spend > self.MIN_TRADE_USD:
                    entry_fee = usd_to_spend * self.FEE_RATE
                    net_investment = usd_to_spend - entry_fee
                    amount = net_investment / price
                    self.current_position = BacktestPosition(price, amount, timestamp, usd_to_spend)

    def close_all_positions(self, row):
        if self.current_position:
            pos = self.current_position
            price = row['close']
            gross_usd = pos.amount * price
            fee = gross_usd * self.FEE_RATE
            net_usd = gross_usd - fee
            self.balance = net_usd
            net_pnl = net_usd - pos.cost_basis
            self.trades.append({
                'entry_time': pos.entry_time,
                'exit_time': row.name,
                'entry_price': pos.entry_price,
                'exit_price': price,
                'amount': pos.amount,
                'pnl': net_pnl,
                'pnl_pct': (net_usd - pos.cost_basis) / pos.cost_basis,
                'reason': "End of Backtest",
                'balance': self.balance
            })
            self.current_position = None

    def report(self):
        days = (self.data.index[-1] - self.data.index[0]).days
        years = days / 365.0 if days > 0 else 1

        total_pnl = self.balance - self.initial_balance
        pnl_pct = (total_pnl / self.initial_balance) * 100

        initial_price = self.data.iloc[0]['close']
        final_price = self.data.iloc[-1]['close']
        bnh_return = ((final_price - initial_price) / initial_price) * 100

        wins = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        print("\n" + "="*30)
        print(f"BACKTEST RESULTS ({days} days) - EMA {self.EMA_LENGTH}")
        print("="*30)
        print(f"Initial Balance: ${self.initial_balance:.2f}")
        print(f"Final Balance:   ${self.balance:.2f}")
        print(f"Total Profit:    ${total_pnl:.2f} ({pnl_pct:.2f}%)")
        print(f"Annualized Return: {pnl_pct/years:.2f}%")
        print(f"Buy & Hold Return: {bnh_return:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        print(f"Win Rate:        {win_rate*100:.2f}%")
        print("="*30 + "\n")

class BacktestPosition:
    def __init__(self, price, amount, entry_time, cost_basis):
        self.entry_price = float(price)
        self.amount = float(amount)
        self.entry_time = entry_time
        self.cost_basis = cost_basis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('--ema', type=int, default=800)
    args = parser.parse_args()

    backtester = Backtester(args.data_path, ema_length=args.ema)
    backtester.run()
