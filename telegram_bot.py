"""
Simple Telegram bot that connects directly to your DataDialogue agent.
No OpenClaw needed — this is a direct Telegram → DataDialogue bridge.

Usage:
    python telegram_bot.py
"""

import logging
import os
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DATADIALOGUE_API_URL = "http://localhost:8000/ask"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Hi! I'm your DataDialogue assistant.\n\n"
        "Ask me anything about your company documents, database, or the web!\n\n"
        "Examples:\n"
        "- What day is today?\n"
        "- What is 15% of 85000?\n"
        "- What are Shivam's technical skills?\n"
        "- What's the weather in Bangalore?"
    )


async def query_backend(question: str, session_id: str) -> dict:
    """Send question to DataDialogue API and return the parsed JSON result."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            DATADIALOGUE_API_URL,
            json={"question": question, "session_id": session_id}
        )
        response.raise_for_status()
        return response.json()


def format_reply(result: dict) -> str:
    """Format the raw API result into a clean Telegram Markdown string."""
    reply = result.get("answer", "Sorry, I could not get an answer.")
    
    # 1. Add Citations
    citations = result.get("citations", [])
    if citations:
        sources = [c["source"] for c in citations if c.get("source")]
        if sources:
            unique_sources = list(dict.fromkeys(sources))
            reply += f"\n\n📎 *Sources:* {', '.join(unique_sources)}"

    # 2. Add Tool Badges
    tools_used = result.get("tools_used", [])
    if tools_used:
        tool_icons = {
            "search_company_documents": "📄",
            "search_web": "🌐",
            "calculate": "🧮",
            "generate_chart": "📊",
            "summarise_document": "📋",
            "query_company_database": "🗄️",
        }
        icons = [tool_icons.get(t, "🔧") for t in tools_used[:2]]
        if icons:
            reply += f"\n{''.join(icons)}"
            
    return reply


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forward every Telegram message to DataDialogue and reply with the answer."""
    user = update.effective_user
    question = update.message.text
    session_id = f"telegram_{user.id}"

    logger.info("Message from %s (%s): %s", user.full_name, user.id, question)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        result = await query_backend(question, session_id)
        reply_text = format_reply(result)
        
        await update.message.reply_text(reply_text, parse_mode="Markdown")
        logger.info("Replied to %s: datasource=%s", user.full_name, result.get("datasource"))

    except httpx.ConnectError:
        await update.message.reply_text("⚠️ Cannot reach the DataDialogue backend. Make sure FastAPI is running.")
    except Exception as e:
        logger.exception("Error processing message")
        await update.message.reply_text(f"⚠️ Something went wrong: {str(e)[:200]}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "*DataDialogue Assistant — What I can do:*\n\n"
        "📄 *Documents* — Ask about your uploaded PDFs, Word docs, Excel files\n"
        "🌐 *Web Search* — Live web search for current information\n"
        "🧮 *Calculator* — Safe arithmetic calculations\n"
        "📊 *Charts* — Generate data visualizations\n"
        "🗄️ *Database* — Query your company database\n"
        "💬 *Memory* — I remember your conversation history\n\n"
        "*Example questions:*\n"
        "- What are Shivam's technical skills?\n"
        "- What is 15% of 85000?\n"
        "- What's the current USD to INR rate?\n"
        "- List all tables in the database\n"
        "- What day is today?",
        parse_mode="Markdown"
    )


def main() -> None:
    """Start the bot."""
    print("=" * 50)
    print("DataDialogue Telegram Bot")
    print("=" * 50)
    print(f"Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"DataDialogue API: {DATADIALOGUE_API_URL}")
    print("Starting bot... Press Ctrl+C to stop.")
    print("=" * 50)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
