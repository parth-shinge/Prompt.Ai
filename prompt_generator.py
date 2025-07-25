import streamlit as st
import requests

# ==== API KEY ====
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # your AI Studio key

# ==== OFFLINE TEMPLATE GENERATOR ====
def generate_template_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    if tool.lower() == "gamma":
        return (
            f"Create a {style} {content_type} about {topic}. "
            "Include engaging visuals, clear text, and a professional layout."
        )
    elif tool.lower() == "canva":
        return (
            f"Design a {style} {content_type} for {platform} about {topic}. "
            f"Use {color_palette} colors and a {mood} mood."
        )
    return f"Create a {style} {content_type} about {topic}."

# ==== GEMINI (AI STUDIO) ====
def generate_gemini_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    # Build the user message
    user_msg = f"I want to create a {style} {content_type} "
    if platform:
        user_msg += f"for {platform} "
    user_msg += f"about {topic}."
    if color_palette or mood:
        user_msg += " I want it to include"
        if color_palette:
            user_msg += f" {color_palette} colors"
        if color_palette and mood:
            user_msg += " and"
        if mood:
            user_msg += f" a {mood} mood"
        user_msg += "."
    user_msg += f" Write the prompt as if the user will paste it into {tool}."

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
    )
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {"contents": [{"parts": [{"text": user_msg}]}]}

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    # Handle quota exceeded (429)
    if response.status_code == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") == 429):
        fallback = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
        return (
            "⚠️ Gemini free-tier quota exceeded; showing offline template instead:\n\n"
            + fallback
        )

    # Any other non-200
    if not response.ok:
        return f"Gemini API error: {data}"

    # Success – extract the generated text
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"Unexpected response format: {data}"

# ==== STREAMLIT UI ====
def main():
    st.set_page_config(page_title="Prompt Gen", page_icon="✨")
    st.title("✨ Prompt Generator for AI Tools")
    st.markdown("Generate perfect prompts for **Gamma** and **Canva** with optional Gemini AI support!")

    # Sidebar examples and footer
    st.sidebar.header("🔍 Examples")
    st.sidebar.markdown("- Design a marketing poster")
    st.sidebar.markdown("- Generate presentation content")
    st.sidebar.markdown("- Write a blog header image prompt")
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Made with ❤️ by [Parth Shinge](https://github.com/parth-shinge)")

    # Main inputs
    tool = st.selectbox("Choose a tool:", ["Gamma", "Canva"])
    content_type = st.text_input("Content type (e.g. presentation, infographic, poster):")
    topic = st.text_input("Topic:")
    style = st.text_input("Style (e.g. modern, playful, minimalist):")

    platform = color_palette = mood = None
    if tool == "Canva":
        platform = st.text_input("Platform (e.g. Instagram, Facebook):")
        color_palette = st.text_input("Color palette (e.g. bright, pastel, dark):")
        mood = st.text_input("Mood (e.g. energetic, calm, elegant):")

    use_ai = st.checkbox("Use Gemini AI for richer prompts?")

    if st.button("Generate Prompt"):
        with st.spinner("Generating..."):
            if use_ai:
                prompt = generate_gemini_prompt(
                    tool, content_type, topic, style,
                    platform, color_palette, mood
                )
            else:
                prompt = generate_template_prompt(
                    tool, content_type, topic, style,
                    platform, color_palette, mood
                )

        st.success("✅ Prompt Generated:")
        st.code(prompt, language="markdown")

if __name__ == "__main__":
    main()
