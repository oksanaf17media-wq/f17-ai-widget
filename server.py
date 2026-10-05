import os
import traceback
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__, template_folder='.')
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
Ты — вежливый и экспертный AI-продавец агентства F17 Media. Твоя задача — общаться с потенциальным клиентом, узнать его нишу и задачи, сориентировать по пакетам и предложить перешагнуть в WhatsApp для бронирования созвона.

Твои продукты и цены:
1. Launch Pack (Только видео): 6 готовых роликов под ключ — £790 разово.
2. Momentum Pack: Видео на постоянную основе (регулярный контент) — рассчитывается индивидуально под объем.
3. Full Funnel (Полная система): Видео + воронка под ключ + запуск рекламы — от £1780 разово + £350–400/мес ведение.

Правила диалога:
- Будь кратким, говори по делу, без «воды» и маркетинговых штампов.
- Задавай по одному вопросу за раз (сначала ниша и цель, потом объем).
- Когда поймешь задачу клиента, назови ориентир по цене и обязательно скажи: «Чтобы зафиксировать условия и обсудить детали, давай перейдем в WhatsApp» (упомяни слово WhatsApp, чтобы в интерфейсе появилась кнопка).
"""

MODEL_NAME = "gemini-3.8-flash"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        history = data.get('history', [])

        contents = []
        for h in history:
            role = "model" if h.get("role") == "assistant" else h.get("role", "user")
            if role not in ("user", "model"):
                role = "user"
            if h.get("parts"):
                p = h["parts"][0]
                text = p.get("text", "") if isinstance(p, dict) else str(p)
            else:
                text = h.get("text", "")
            if not text.strip():
                continue
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

        # история должна начинаться с user и заканчиваться user
        while contents and contents[0].role == "model":
            contents.pop(0)
        while contents and contents[-1].role == "model":
            contents.pop()
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part(text="Привіт")])]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            ),
        )
        return jsonify({"reply": response.text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
