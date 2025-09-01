import streamlit as st
import requests
import hashlib
import hmac
from database import Prompt, User, Feedback, SessionLocal
from sqlalchemy import select
import pandas as pd

# ==== API KEY ====
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ==== ADMIN CREDENTIALS from Streamlit secrets ====
# Expect these keys in .streamlit/secrets.toml or Streamlit Cloud secrets:
# ADMIN_USERNAME, ADMIN_PW_SALT, ADMIN_PW_HASH
ADMIN_USERNAME = st.secrets.get("ADMIN_USERNAME", None)
ADMIN_PW_SALT = st.secrets.get("ADMIN_PW_SALT", None)
ADMIN_PW_HASH = st.secrets.get("ADMIN_PW_HASH", None)

# ==== HELPERS: password verification ====
def verify_password(plain_password: str) -> bool:
    """Hash the provided plain_password using stored salt and compare to stored hash."""
    if not (ADMIN_PW_SALT and ADMIN_PW_HASH):
        return False
    # pbkdf2_hmac using same params as the generator (sha256, 200k iter)
    computed = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode(),
        ADMIN_PW_SALT.encode(),
        200_000
    ).hex()
    # use constant-time compare
    return hmac.compare_digest(computed, ADMIN_PW_HASH)

# ==== OFFLINE TEMPLATE GENERATOR ====
def generate_template_prompt(tool, content_type, topic, style):
    if tool.lower() == "gamma":
        return (
            f"Create a {style} {content_type} about {topic}. "
            "Include engaging visuals, clear text, and a professional layout."
        )
    elif tool.lower() == "canva":
        return (
            f"Design a {style} {content_type} for social media about {topic}. "
            "Use creative colors and engaging tone."
        )
    return f"Create a {style} {content_type} about {topic}."

# ==== GEMINI (AI STUDIO) ====
def generate_gemini_prompt(tool, content_type, topic, style):
    user_msg = f"I want to create a {style} {content_type} about {topic}. "
    user_msg += f"Write the prompt as if the user will paste it into {tool}."

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
    try:
        data = response.json()
    except Exception:
        return "Gemini API error: invalid JSON response."

    # quota fallback
    if response.status_code == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") == 429):
        fallback = generate_template_prompt(tool, content_type, topic, style)
        return (
            "⚠️ Gemini free-tier quota exceeded; showing offline template instead:\n\n"
            + fallback
        )

    if not response.ok:
        return f"Gemini API error: {data}"

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"Unexpected response format: {data}"

# ==== SAVE / DELETE PROMPT ====
def save_prompt(tool, content_type, topic, style, generated_prompt, model_used="offline"):
    with SessionLocal() as session:
        new_prompt = Prompt(
            tool=tool,
            content_type=content_type,
            topic=topic,
            style=style,
            generated_text=generated_prompt,
            model_used=model_used,
        )
        session.add(new_prompt)
        session.commit()

def delete_prompt(prompt_id):
    with SessionLocal() as session:
        prompt = session.get(Prompt, prompt_id)
        if prompt:
            session.delete(prompt)
            session.commit()
            return True
        return False

def handle_delete(prompt_id):
    deleted = delete_prompt(prompt_id)
    if deleted:
        st.success("Prompt deleted.")
    else:
        st.error("Prompt not found.")
    # Streamlit auto re-runs after widget callbacks

# ==== SESSION STATE defaults used for search/history ====
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None
if "last_search_tool" not in st.session_state:
    st.session_state["last_search_tool"] = "All"
if "last_search_topic" not in st.session_state:
    st.session_state["last_search_topic"] = ""
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

# ==== ADMIN LOGIN UI ====
def admin_login_ui():
    """Return True if login successful, False otherwise."""
    st.subheader("🔐 Admin Login")
    with st.form("admin_login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if ADMIN_USERNAME is None or ADMIN_PW_SALT is None or ADMIN_PW_HASH is None:
            st.error("Admin credentials are not configured. Add them to Streamlit secrets.")
            return False
        if username != ADMIN_USERNAME:
            st.error("Invalid username.")
            return False
        if verify_password(password):
            st.session_state["admin_logged_in"] = True
            st.success("✅ Admin login successful.")
            return True
        else:
            st.error("Invalid password.")
            return False
    return False

# ==== ADMIN PANEL UI ====
def admin_panel():
    st.title("🛠️ Admin Panel")
    menu = ["View Users", "View Prompts", "View Feedback"]
    choice = st.selectbox("Choose view", menu, key="admin_menu_select")

    with SessionLocal() as session:
        if choice == "View Users":
            st.subheader("👤 Users")
            users = session.execute(select(User)).scalars().all()
            if users:
                df = pd.DataFrame([{
                    "id": u.id,
                    "username": u.username,
                    "email": u.email
                } for u in users])
                st.dataframe(df)
            else:
                st.info("No users found.")

        elif choice == "View Prompts":
            st.subheader("📝 Prompts")
            prompts = session.execute(select(Prompt)).scalars().all()
            if prompts:
                for p in prompts:
                    st.markdown(f"**ID:** {p.id} | **Tool:** {p.tool} | **Topic:** {p.topic} | **Model:** {p.model_used}")
                    st.code(p.generated_text, language="markdown")
                    col_d, col_sp = st.columns([0.6, 8])
                    with col_d:
                        st.button("🗑️ Delete", key=f"admin_delete_{p.id}", on_click=handle_delete, args=(p.id,))
                    st.markdown("---")
            else:
                st.info("No prompts found.")

        elif choice == "View Feedback":
            st.subheader("💬 Feedback")
            feedbacks = session.execute(select(Feedback)).scalars().all()
            if feedbacks:
                df = pd.DataFrame([{
                    "id": f.id,
                    "rating": f.rating,
                    "comments": f.comments,
                    "timestamp": f.timestamp,
                    "user_id": f.user_id,
                    "prompt_id": f.prompt_id
                } for f in feedbacks])
                st.dataframe(df)
            else:
                st.info("No feedback found.")

# ==== MAIN: generator + history UI (unchanged but wrapped) ====
def run_prompt_generator():
    st.title("✨ Prompt Generator for AI Tools")
    st.markdown("Generate perfect prompts for **Gamma** and **Canva** with optional Gemini AI support!")

    # Sidebar examples (keep)
    st.sidebar.header("🔍 Examples")
    st.sidebar.markdown("- Gamma: Modern sales deck with key metrics and customer testimonials")
    st.sidebar.markdown("- Gamma: Educational presentation on renewable energy trends")
    st.sidebar.markdown("- Canva: Instagram story announcing a product launch")
    st.sidebar.markdown("- Canva: Facebook post graphic for a holiday sale")
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍💻 Made with ❤️ by [Parth Shinge](https://github.com/parth-shinge)")

    # Tabs layout
    tab1, tab2 = st.tabs(["✨ Generate Prompt", "📂 Search History"])

    # TAB 1
    with tab1:
        st.subheader("✨ Generate a New Prompt")
        with st.form("generate_form", clear_on_submit=True):
            tool = st.selectbox("Choose a tool:", ["Gamma", "Canva"])
            content_type = st.text_input("Content type (e.g. presentation, infographic, poster):")
            topic = st.text_input("Topic:")
            style = st.text_input("Style (e.g. modern, playful, minimalist):")
            use_ai = st.checkbox("Use Gemini AI for richer prompts?")
            submitted = st.form_submit_button("Generate Prompt")

        if submitted:
            with st.spinner("Generating..."):
                if use_ai:
                    prompt = generate_gemini_prompt(tool, content_type, topic, style)
                    model_used = "gemini"
                else:
                    prompt = generate_template_prompt(tool, content_type, topic, style)
                    model_used = "offline"

            save_prompt(tool, content_type, topic, style, prompt, model_used)
            st.success("✅ Prompt Generated & Saved:")
            st.code(prompt, language="markdown")

    # TAB 2
    with tab2:
        st.subheader("📂 Search Your Prompt History")
        search_tool = st.selectbox("Filter by tool:", ["All", "Gamma", "Canva"], key="search_tool_select")
        search_topic = st.text_input("Search by topic:", key="search_topic_input")
        col_btn1, col_btn2, _ = st.columns([0.8, 0.8, 8])
        search_clicked = col_btn1.button("🔎 Search", key="btn_search")
        view_all_clicked = col_btn2.button("📂 View All", key="btn_view_all")

        if search_clicked:
            st.session_state["last_action"] = "search"
            st.session_state["last_search_tool"] = search_tool
            st.session_state["last_search_topic"] = search_topic
        elif view_all_clicked:
            st.session_state["last_action"] = "view_all"

        results = []
        with SessionLocal() as session:
            if st.session_state["last_action"] == "search":
                stmt = select(Prompt)
                if st.session_state["last_search_tool"] != "All":
                    stmt = stmt.where(Prompt.tool == st.session_state["last_search_tool"])
                if st.session_state["last_search_topic"]:
                    stmt = stmt.where(Prompt.topic.contains(st.session_state["last_search_topic"]))
                results = session.execute(stmt).scalars().all()
            elif st.session_state["last_action"] == "view_all":
                results = session.execute(select(Prompt)).scalars().all()

        if results:
            with st.expander("🔎 Show Results", expanded=True):
                for r in results:
                    st.markdown(
                        f"**Tool:** {r.tool} | **Topic:** {r.topic} | **Style:** {r.style} | **Model:** {r.model_used}"
                    )
                    st.code(r.generated_text, language="markdown")
                    colA, colB, colC = st.columns([1.1, 1.8, 8])
                    with colA:
                        st.button(
                            label="🗑️ Delete",
                            key=f"delete_btn_{r.id}",
                            on_click=handle_delete,
                            args=(r.id,)
                        )
                    with colB:
                        st.download_button(
                            label="📥 Download as .txt",
                            data=r.generated_text,
                            file_name=f"prompt_{r.id}.txt",
                            mime="text/plain",
                            key=f"download_{r.id}"
                        )
                    st.markdown("---")
        else:
            if st.session_state["last_action"] in ("search", "view_all"):
                st.info("Nothing here yet — adjust your search or generate a new prompt.")

# ==== App entry: navigation between generator and admin panel ====
def main():
    st.set_page_config(page_title="Prompt Gen", page_icon="✨", layout="wide")
    # top-level sidebar navigation
    app_mode = st.sidebar.radio("Navigation", ["Prompt Generator", "Admin Panel"], index=0)

    if app_mode == "Prompt Generator":
        run_prompt_generator()
    else:  # Admin Panel
        # require login
        if st.session_state.get("admin_logged_in", False):
            # show a logout button
            if st.sidebar.button("Logout Admin"):
                st.session_state["admin_logged_in"] = False
                st.experimental_rerun()
            admin_panel()
        else:
            # show login form; if successful, admin_login_ui() sets admin_logged_in
            logged_in_now = admin_login_ui()
            if logged_in_now:
                admin_panel()

if __name__ == "__main__":
    main()
