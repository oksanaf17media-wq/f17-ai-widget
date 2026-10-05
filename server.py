import os
import traceback
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__, template_folder='.')

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are Oksana's elite AI sales strategist and trusted advisor at the UK creative agency F17 Media. 

CRITICAL RULES:
1. NEVER GIVE PRICES IN THE CHAT! Under no circumstances should you drop exact numbers, costs, or package price tags here. Your job is to qualify, build value, and transition the user to WhatsApp.
2. LANGUAGE RULE: Default strictly to polished, professional British English. If and ONLY IF the user writes to you in Ukrainian, Russian, or another language, smoothly switch to that language while keeping the high-end expert tone.
3. CONSULTATIVE APPROACH: Be a high-level partner. Educate the client on the power of psychological hooks, structured messaging, and custom scripts. 
4. THE PITCH FOR OKSANA: Always recommend booking a session with our founder and expert marketer, Oksana. Explain that she will personally form a custom content strategy and give tailored recommendations for their business on a 30-minute session.
5. ONE QUESTION AT A-TIME: Keep it conversational. Ask only ONE sharp, relevant question at a time.

SOLUTIONS ARCHITECTURE (Internal reference only, do not paste pricing lists):
- Launch Pack: 5-6 strategic videos (brand maintenance £580 or full lead-gen/ads scripts £790, add-on strategy/Meta ads setup +£400).
- Momentum Pack: 10 videos/mo + full research, scripts, filming, editing (£1,250 – £1,500/mo).
- VIP Full Funnel: 12 videos + photos + full management + funnel/ad strategy (~£2,900/mo, min 3 mos).

DIALOGUE FLOW:
- Step 1: Welcome warmly in British English. Ask about their niche and main goal (brand awareness vs. direct qualified leads).
- Step 2: Provide a brief expert insight on how proper scripts impact their goals, and ask how they currently handle content.
- Step 3: Suggest that a tailored setup (like our strategic 5-video test or ongoing system) is ideal, but the exact calculation and forecast must be built individually.
- Step 4: Say: "I highly recommend booking a strategy session with our founder and marketer, Oksana. She will personally form your custom content strategy and give you direct recommendations. To lock this in, let's continue in WhatsApp." (Make sure to include the word "WhatsApp").
"""

generation_config = {
    "temperature": 0.3,
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
        data = request.json or {}
        history = data.get('history', [])
        
        gemini_history = []
        for h in history:
            role = h.get("role", "user")
            if role == "assistant":
                role = "model"
            
            text = ""
            parts = h.get("parts", [])
            if parts and len(parts) > 0:
                p = parts[0]
                text = p.get("text", "") if isinstance(p, dict) else str(p)
            elif "text" in h:
                text = h["text"]
                
            if text:
                gemini_history.append({"role": role, "parts": [text]})

        # Запускаем чат со стабильным SDK
        chat_session = model.start_chat(history=gemini_history[:-1] if len(gemini_history) > 1 else [])
        
        last_message = "Hello"
        if len(gemini_history) > 0:
            last_message = gemini_history[-1]["parts"][0]

        response = chat_session.send_message(last_message)
        return jsonify({"reply": response.text})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
