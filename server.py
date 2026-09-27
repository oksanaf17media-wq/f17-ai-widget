import os
from flask import Flask, request, jsonify, render_template
from google import genai

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    history = data.get('history', [])
    
    formatted_contents = []
    for h in history:
        formatted_contents.append({
            "role": h["role"],
            "parts": [{"text": h["parts"][0]["text"]}]
        })

    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=formatted_contents,
            config={
                'system_instruction': SYSTEM_PROMPT,
                'temperature': 0.7,
            }
        )
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
