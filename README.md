# 🔮 Prompt Generator (Private MVP)

**Prompt Gen** is a simple yet powerful AI-powered prompt generator for tools like **Gamma** and **Canva**. This MVP version lets users choose their content type, style, topic, and more, and then generates a high-quality prompt either offline or using Google Gemini's API.

## 🚀 Features
- Generate prompt templates instantly
- Optional integration with **Google Gemini (AI Studio)**
- Tool-specific customization (Gamma / Canva)
- Offline fallback prompt logic
- Built with **Streamlit**

## 🧪 Technologies Used
- 🐍 Python
- 📺 Streamlit
- 🤖 Google Gemini API (via REST)
- 💡 Simple offline logic fallback

## 💻 How to Run This Locally

1. **Clone the Repository**:
git clone https://github.com/YOUR_USERNAME/prompt-gen.git
cd prompt-gen

2. **(Optional) Create Virtual Environment** :
python -m venv venv
venv\Scripts\activate  # for Windows

3. **Install Dependencies**:
pip install streamlit requests

4. **Add your API Key
Edit the line in prompt_generator.py:**:
GEMINI_API_KEY = "your-api-key-here"

5. **Run the App**:
streamlit run prompt_generator.py

🧠 Future Plans
1. Add support for more tools (Beautify, Presentations, etc.)
2. Integrate offline AI model (custom LLM)
3. Support image-based prompt previews

⚠️ This repository is private and meant for MVP testing only. Do not share or expose your API key.

Made with ❤️ by Parth Shinge
