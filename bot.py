# Прототип чат-бота: поиск по смыслу с помощью ruBERT и интеграция с Telegram

# 🔧 УСТАНОВКА БИБЛИОТЕК (один раз):
# pip install transformers scikit-learn torch pandas python-telegram-bot

# 📄 СТРУКТУРА ПРОЕКТА:
# - bot.py                 ← этот файл (чат-бот)
# - questions.csv          ← база знаний в CSV (два столбца: Вопрос;Ответ)

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# 🔐 УСТАНОВИ СЮДА СВОЙ TELEGRAM ТОКЕН
TELEGRAM_TOKEN = "8156239934:AAFeZX-InTvBNuZHbpO4VpOe3nRhIEJn25A"

# 📘 Загружаем русскоязычную модель ruBERT
MODEL_NAME = "ai-forever/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# 📁 Загрузка базы знаний
csv_path = "questions.csv"  # должен содержать столбцы: Вопрос;Ответ
try:
    df = pd.read_csv(csv_path, sep=';', quotechar='"')
    if not {'Вопрос', 'Ответ'}.issubset(df.columns):
        raise ValueError("Файл должен содержать столбцы 'Вопрос' и 'Ответ'")
    knowledge_base = dict(zip(df["Вопрос"], df["Ответ"]))
except Exception as e:
    print(f"Ошибка при загрузке базы знаний: {e}")
    knowledge_base = {}

# 📌 Функция эмбеддинга текста

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 🧠 Предварительное кодирование базы знаний
kb_questions = list(knowledge_base.keys())
kb_embeddings = [get_embedding(q) for q in kb_questions] if knowledge_base else []

# 🔎 Поиск ближайшего вопроса

def find_best_match(question, threshold=0.6):
    if not kb_embeddings:
        return None
    question_emb = get_embedding(question)
    sims = cosine_similarity([question_emb], kb_embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    if best_score < threshold:
        return None
    return knowledge_base[kb_questions[best_idx]]

# 💬 Обработка входящих сообщений

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    if not knowledge_base:
        await update.message.reply_text("⚠️ База знаний пуста или не загружена. Обратитесь к администратору.")
        return
    answer = find_best_match(user_input)
    if answer:
        await update.message.reply_text(f"🤖 {answer}")
    else:
        await update.message.reply_text("Я пока не знаю ответа на этот вопрос. Попробуй переформулировать его, пожалуйста 🙏")

# 🔹 Команда /start

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я помогу тебе с математикой 😊 Просто задай вопрос, например: 'Что такое деление?' или 'Как сложить числа?'"
    )

# ▶️ Запуск Telegram-бота

if __name__ == "__main__":
    if not TELEGRAM_TOKEN or "вставь" in TELEGRAM_TOKEN:
        print("❌ Укажи корректный токен TELEGRAM_TOKEN.")
    else:
        app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("✅ Бот запущен. Ожидает сообщений в Telegram...")
        app.run_polling()