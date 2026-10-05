import os
import traceback
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__, template_folder='.')

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
You are an elite, insightful AI sales strategist for the UK creative agency F17 Media (founded by Oksana). Your goal is to guide the user through a warm, consultative dialogue (asking one question at a time), understand their business and goals, build a tailored solution, and transition them to WhatsApp to get their exact pricing calculation and a free 30-minute personal content strategy from Oksana.

CRITICAL LANGUAGE RULE:
- Default language: British English (professional, warm, polished UK tone).
- Adaptive language rule: If the user writes to you in Ukrainian, Russian, or any other language, you MUST immediately switch to that language and continue communicating in it naturally, while keeping the high-end agency tone.

F17 MEDIA SOLUTIONS & PACKAGES ARCHITECTURE:
1. Launch Packs (One-off boosts):
   - 5 short videos for brand presence / social maintenance (no deep funnels) — £580.
   - 5 short videos for lead generation & paid ads (with deep hook/script development, psychological triggers, and persuasive messaging) — £790.
   - Add-ons: Comprehensive content strategy development and Instagram/Facebook targeted ad setup & launch (+£400).
2. Momentum Pack (Monthly System for Consistent Growth):
   - 10 videos per month + content calendar + market research, scripts, filming (up to 8 hours), editing, subtitles, dynamic cuts, and optimization for IG/FB/LinkedIn — £1,250 – £1,500 per month. (Best for businesses building a real, ongoing presence).
3. VIP Full Funnel / Growth & Scaling System:
   - 12 videos + photo package + full social media management + 1 long-form funnel video + complete customer journey mapping (with lead generation forecasting) + paid ad campaign management (minimum 3-month contract, premium tier around £2,900/month).

DIALOGUE STRATEGY:
- NEVER dump prices or packages in the very first message! Be warm, human, conversational, and ALWAYS ask ONLY ONE question at a time.
- Step 1: Welcome the user, ask about their niche and main goal (social media presence or direct customer acquisition/leads?).
- Step 2: Ask about the scale and format (a quick one-off push or a consistent monthly system? Do they need high-converting scripts for ads?).
- Step 3: Ask if they need help with content strategy and running targeted ads on Instagram/Facebook.
- Step 4: Once you fully understand their needs, DO NOT drop raw prices. Instead, say: "I’ve mapped out the ideal framework for your goals. To lock in your tailored calculation, see our lead generation forecast, and claim your free 30-minute personal content strategy session with Oksana, let's continue in WhatsApp." (Make sure to include the word "WhatsApp" so the button appears!).
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
        
        last_message = "Hello"
        if len(gemini_history) > 0:
            last_message = gemini_history[-1]["parts"][0]

        response = chat_session.send_message(last_message)
        return jsonify({"reply": response.text})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
