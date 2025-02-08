from ib_insync import *
import pandas as pd
import talib

# Create IBKR connection
ib = IB()

try:
    print("🔄 Attempting to connect to IBKR...")
    ib.connect('127.0.0.1', 7497, clientId=1)  # Ensure TWS or IB Gateway is running
    print("✅ Connected to IBKR!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

# Get stock symbol from user
symbol = input("Enter the stock ticker symbol (e.g., 'TSLA'): ").upper()

# Define contract
contract = Stock(symbol, 'SMART', 'USD')

# Verify contract exists
print(f"🔍 Checking contract details for {symbol}...")
contract_details = ib.reqContractDetails(contract)

if not contract_details:
    print(f"❌ Invalid contract. {symbol} might not be available.")
    ib.disconnect()
    exit()

print(f"✅ Contract verified: {contract_details[0].contract}")

# Request historical data
print(f"📊 Requesting historical data for {symbol}...")
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='5 mins',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

# Debugging: Log the response from IBKR
if not bars:
    print(f"❌ No data received for {symbol}. Check IBKR API settings.")
    ib.disconnect()
    exit()

print(f"✅ Received {len(bars)} data points.")

# Convert to Pandas DataFrame
df = util.df(bars)

# Print first few rows
print("\n📊 Sample Data Received:")
print(df.head())

# Ensure data columns are float for TA-Lib
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)

# Detect Candlestick Patterns
print("\n🔍 Detecting Candlestick Patterns...")
df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])

# Filter detected patterns
engulfing_signals = df[df['engulfing'] != 0]
hammer_signals = df[df['hammer'] != 0]
doji_signals = df[df['doji'] != 0]

# Print detected patterns
if not engulfing_signals.empty:
    print(f"✅ Engulfing Pattern detected at:\n{engulfing_signals[['date', 'engulfing']]}")

if not hammer_signals.empty:
    print(f"✅ Hammer Pattern detected at:\n{hammer_signals[['date', 'hammer']]}")

if not doji_signals.empty:
    print(f"✅ Doji Pattern detected at:\n{doji_signals[['date', 'doji']]}")

# Trading Logic
print("\n📈 Making Trade Decisions...")
last_close = df['close'].iloc[-1]

if not engulfing_signals.empty:
    print("🚀 Engulfing pattern detected, placing a BUY order...")
    order = MarketOrder("BUY", 1)
    trade = ib.placeOrder(contract, order)
    print(f"✅ Order placed: {trade}")

elif not hammer_signals.empty:
    print("🚀 Hammer pattern detected, placing a BUY order...")
    order = MarketOrder("BUY", 1)
    trade = ib.placeOrder(contract, order)
    print(f"✅ Order placed: {trade}")

elif not doji_signals.empty:
    print("⚠ Doji pattern detected, holding position (uncertain trend).")

else:
    print("❌ No strong pattern detected. No trade executed.")

# Disconnect from IBKR
ib.disconnect()
print("\n✅ Script execution completed.")







