import os
import traceback
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__, template_folder='.')

# Инициализация стабильного SDK
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

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

generation_config = {
    "temperature": 0.7,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=SYSTEM_PROMPT
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        history = data.get('history', [])
        
        gemini_history = []
        for h in history:
            role = h.get("role", "user")
            if role == "assistant":
                role = "model"
            
            text = ""
            if "parts" in h and len(h["parts"]) > 0:
                text = h["parts"][0].get("text", "")
            elif "text" in h:
                text = h["text"]
                
            gemini_history.append({"role": role, "parts": [text]})

        chat_session = model.start_chat(history=gemini_history[:-1] if len(gemini_history) > 0 else [])
        
        last_message = "Привет"
        if len(gemini_history) > 0:
            last_message = gemini_history[-1]["parts"][0]

        response = chat_session.send_message(last_message)
        return jsonify({"reply": response.text})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
