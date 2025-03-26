import backtrader as bt
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model_rf_gold.pkl")

class MLStrategy(bt.Strategy):
    params = (('model', None), ('features', ['SMA_20', 'SMA_50', 'RSI']),)

    def __init__(self):
        # Create indicators to compute current feature values
        self.sma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.sma50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

    def next(self):
        # Build feature array from current indicator values
        current_features = np.array([
            self.sma20[0],
            self.sma50[0],
            self.rsi[0]
        ]).reshape(1, -1)
        
        # Use the ML model to predict signal
        signal = self.p.model.predict(current_features)[0]
        
        # Trading logic: if predicted up (1) and not in position, then buy; if predicted down (0) and in position, then sell.
        if signal == 1 and not self.position:
            self.buy()
        elif signal == 0 and self.position:
            self.close()

if __name__ == "__main__":
    # Create a Cerebro engine
    cerebro = bt.Cerebro()

    # Add the strategy, passing the trained model as a parameter
    cerebro.addstrategy(MLStrategy, model=model)

    # Load processed data for gold from CSV (make sure the index is parsed as dates)
    data = pd.read_csv("data/gold_data_processed.csv", index_col=0, parse_dates=True)
    data_feed = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_feed)

    # Set initial cash and other broker parameters
    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run the backtest
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the results
    cerebro.plot()
