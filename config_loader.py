import os
import yaml
from typing import Dict, Any

def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    # Set defaults with environment variable fallbacks.
    config.setdefault("tws", {})
    config["tws"].setdefault("host", os.getenv("TWS_HOST", "127.0.0.1"))
    config["tws"].setdefault("port", int(os.getenv("TWS_PORT", "7497")))
    config["tws"].setdefault("client_id", int(os.getenv("TWS_CLIENT_ID", "1")))
    
    config.setdefault("risk", {})
    config["risk"].setdefault("account_balance", float(os.getenv("ACCOUNT_BALANCE", "100000")))
    config["risk"].setdefault("risk_per_trade", float(os.getenv("RISK_PER_TRADE", "0.02")))
    
    config.setdefault("polling", {})
    config["polling"].setdefault("interval", float(os.getenv("POLLING_INTERVAL", "0.5")))
    config["polling"].setdefault("fallback_threshold", float(os.getenv("FALLBACK_THRESHOLD", "20.0")))
    config["polling"].setdefault("timeout", float(os.getenv("ORDER_TIMEOUT", "30.0")))
    
    config.setdefault("simulation", True)
    config.setdefault("paper_trading", True)
    
    config.setdefault("alerts", {})
    config["alerts"].setdefault("enabled", os.getenv("ALERTS_ENABLED", "False") == "True")
    config["alerts"].setdefault("telegram_bot_token", os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN"))
    config["alerts"].setdefault("telegram_chat_id", os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID"))
    
    return config

config = load_config()




