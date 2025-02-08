from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

binance_api_key = os.getenv("BINANCE_API_KEY")
binance_api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(binance_api_key, binance_api_secret)

klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_5MINUTE, limit=5)
print(klines)
