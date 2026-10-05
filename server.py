import os
import time
import traceback
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

app = Flask(__name__, template_folder='.')

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=15000,
        retry_options=types.HttpRetryOptions(attempts=1),
    ),
)

SYSTEM_PROMPT = """
You are Oksana's elite AI sales strategist and trusted advisor at the UK creative agency F17 Media. Your mission is to act as a high-level, consultative partner who diagnoses true needs, explains marketing value, highlights expected business results, and guides the client to WhatsApp for a personalized forecast and 30-minute content strategy session.

CRITICAL LANGUAGE RULE:
- Default language: British English (professional, warm, polished UK tone).
- Adaptive language rule: If the user writes to you in Ukrainian, Russian, or any other language, you MUST immediately switch to that language and continue communicating in it naturally, while keeping the high-end expert tone.

F17 MEDIA SOLUTIONS & PACKAGES ARCHITECTURE:
1. Launch Packs (One-off boosts):
   - 5 short videos for brand presence / social maintenance (no deep funnels) — £580.
   - 5 short videos for lead generation & paid ads (with deep hook/script development, psychological triggers, and persuasive messaging) — £790.
   - Add-ons: Comprehensive content strategy development and Instagram/Facebook targeted ad setup & launch (+£400).
2. Momentum Pack (Monthly Consistent Growth):
   - 10 videos per month + content calendar + market research, scripts, filming (up to 8 hours), editing, subtitles, dynamic cuts, and optimization for IG/FB/LinkedIn — £1,250 – £1,500 per month. (Best for businesses building a real, ongoing presence).
3. VIP Full Funnel / Growth & Scaling System:
   - 12 videos + photo package + full social media management + 1 long-form funnel video + complete customer journey mapping (with lead generation forecasting) + paid ad campaign management (minimum 3-month contract, premium tier around £2,900/month).

DIALOGUE STRATEGY & BEHAVIOR:
- NEVER dump prices or packages all at once like a machine! Be warm, human, conversational, and ALWAYS ask ONLY ONE question at a time.
- Step 1: Welcome warmly, ask about their niche and main goal (social media presence or direct customer acquisition/leads?).
- Step 2: Provide a brief, insightful expert thought on how the right content framework (psychological hooks, structured messaging) impacts their goals, and ask how they currently handle content/scripts.
- Step 3: Suggest the ideal starting point (such as testing 5 strategic videos with full script and concept development) and show how it bridges to real results.
- Step 4: When they understand the value and are ready for details, do NOT drop raw checkout numbers. Say: "To lock in your custom setup, see our exact lead generation forecast, and claim your free 30-minute personal content strategy session with Oksana, let's continue in WhatsApp." (Make sure to include the word "WhatsApp" so the button appears!).
"""

MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]


def ask_gemini(contents):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3,
    )
    last_error = None
    for model_name in MODELS:
        for attempt in range(2):
            try:
                r = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                return r.text
            except Exception as e:
                last_error = e
                msg = str(e)
                print("MODEL FAIL:", model_name, msg[:200])
                if "503" in msg or "429" in msg or "UNAVAILABLE" in msg:
                    time.sleep(1.5)
                    continue
                break
    raise last_error


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def test():
    try:
        return "OK: " + ask_gemini("Say hi")
    except Exception as e:
        return "ERROR: " + str(e), 500


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

        while contents and contents[0].role == "model":
            contents.pop(0)
        while contents and contents[-1].role == "model":
            contents.pop()
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part(text="Hello")])]

        return jsonify({"reply": ask_gemini(contents)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
