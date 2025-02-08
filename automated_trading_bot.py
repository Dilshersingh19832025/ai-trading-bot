import logging
from ib_insync import *
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to IBKR
def connect_ibkr():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)  # Paper trading port: 7497, Live: 7496
    logging.info("✅ Connected to IBKR!")
    return ib

# Fetch historical data
def get_historical_data(ib, symbol):
    contract = Stock(symbol, 'SMART', 'USD')
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    if not bars:
        logging.error("❌ No data retrieved. Please check the symbol or market conditions.")
        return None
    
    df = util.df(bars)
    logging.info(f"✅ Received {len(df)} data points.")
    return df

# Analyze market and trade
def analyze_and_trade(ib, df, contract):
    if df is None or df.empty:
        logging.warning("⚠️ No valid data to analyze.")
        return
    
    latest_price = df['close'].iloc[-1]
    logging.info(f"📊 Latest closing price: {latest_price}")

    # Example trade logic (buy if price drops below threshold)
    buy_threshold = latest_price * 0.99
    sell_threshold = latest_price * 1.01

    # Buy order
    order = MarketOrder('BUY', 1)
    trade = ib.placeOrder(contract, order)
    logging.info(f"✅ Order placed: {trade}")

    # Monitor and close trade
    time.sleep(5)
    for t in ib.trades():
        if t.orderStatus.status == 'Filled':
            logging.info(f"✅ Trade filled at {t.orderStatus.avgFillPrice}")
            sell_order = MarketOrder('SELL', 1)
            ib.placeOrder(contract, sell_order)
            logging.info("📈 Selling order placed!")
            break

# Main bot function
def run_trading_bot():
    ib = connect_ibkr()
    symbol = input("Enter the stock ticker symbol (e.g., 'TSLA'): ").upper()
    
    logging.info(f"🔍 Checking contract details for {symbol}...")
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    logging.info(f"📊 Requesting historical data for {symbol}...")
    df = get_historical_data(ib, symbol)

    logging.info("🔍 Analyzing market conditions...")
    analyze_and_trade(ib, df, contract)

    ib.disconnect()
    logging.info("✅ Trading bot execution completed.")

if __name__ == "__main__":
    run_trading_bot()







