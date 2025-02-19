import asyncio
import telegram_send

async def main():
    await telegram_send.send(messages=["Hello! This is a test notification from my trading bot."])

asyncio.run(main())

