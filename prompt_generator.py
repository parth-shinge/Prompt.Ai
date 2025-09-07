# (This is the full updated prompt_generator.py)
import streamlit as st
import requests
import hashlib
import hmac
import random
import matplotlib.pyplot as plt
from sqlalchemy import select

import pandas as pd

# DB helpers
from database import (
    Prompt,
    User,
    Feedback,
    SessionLocal,
    register_user,
    authenticate_user,
    change_password,
    add_feedback,
    get_top_topics,
    get_top_styles,
    get_feedback_counts_by_model,
    get_active_users,
    record_choice,
    get_choice_dataset
)

# ranker utilities
from ranker import train_basic, train_with_embeddings, compare_models, load_ranker, predict_with_ranker

# ==== CONFIG/SECRETS ====
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
ADMIN_USERNAME = st.secrets.get("ADMIN_USERNAME", None)
ADMIN_PW_SALT = st.secrets.get("ADMIN_PW_SALT", None)
ADMIN_PW_HASH = st.secrets.get("ADMIN_PW_HASH", None)

RANKER_PATH = "ranker.pkl"

# helpers
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass


def _clear_generation_state():
    """Clear any previously displayed outputs or hybrid state when inputs change."""
    st.session_state["show_output"] = False
    st.session_state["generated_text"] = None
    st.session_state["generated_prompt_id"] = None
    st.session_state["awaiting_hybrid_choice"] = False
    st.session_state["hybrid_offline_text"] = None
    st.session_state["hybrid_gemini_text"] = None
    st.session_state["hybrid_offline_id"] = None
    st.session_state["hybrid_gemini_id"] = None
    st.session_state["hybrid_choice_model"] = None

def verify_admin_password(plain_password: str) -> bool:
    if not (ADMIN_PW_SALT and ADMIN_PW_HASH):
        return False
    computed = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode(),
        ADMIN_PW_SALT.encode(),
        200_000
    ).hex()
    return hmac.compare_digest(computed, ADMIN_PW_HASH)


# prompt generators
def generate_template_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    if tool.lower() == "gamma":
        return (
            f"Create a {style} {content_type} about {topic}. "
            "Include engaging visuals, clear text, and a professional layout."
        )

    elif tool.lower() == "canva":
        target = platform if platform else "social media"
        sentence = f"Design a {style} {content_type} about {topic}, tailored for {target}. "

        if color_palette:
            sentence += f"Use a {color_palette} color scheme"
            if mood:
                sentence += f" to create a {mood} atmosphere. "
            else:
                sentence += ". "
        elif mood:
            sentence += f"Aim for a {mood} feel in the overall design. "

        sentence += "Make it engaging and visually appealing."
        return sentence

    return f"Create a {style} {content_type} about {topic}."


def generate_gemini_prompt(tool, content_type, topic, style, platform=None, color_palette=None, mood=None):
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    user_parts = [f"I want to create a {style} {content_type} about {topic}."]
    if platform:
        user_parts.append(f"Tailored for {platform}.")
    if color_palette and mood:
        user_parts.append(f"Use a {color_palette} color scheme to create a {mood} atmosphere.")
    elif color_palette:
        user_parts.append(f"Use a {color_palette} color scheme.")
    elif mood:
        user_parts.append(f"Create a {mood} atmosphere.")

    user_parts.append(f"Write the prompt as if the user will paste it into {tool}.")
    user_msg = " ".join([p for p in user_parts if p])

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.0-flash:generateContent"
    )
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": user_msg}]}]}

    response = requests.post(url, headers=headers, json=payload)
    try:
        data = response.json()
    except Exception:
        return "Gemini API error: invalid JSON response."

    if response.status_code == 429 or (isinstance(data, dict) and data.get("error", {}).get("code") == 429):
        fallback = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
        return "⚠️ Gemini free-tier quota exceeded; showing offline template instead:\n\n" + fallback

    if not response.ok:
        return f"Gemini API error: {data}"

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return f"Unexpected response format: {data}"


# DB operations for prompts (save/delete)
def save_prompt(tool, content_type, topic, style, generated_prompt, model_used="offline", user_id=None,
                platform_name=None, color_palette=None, mood=None, used_hybrid=False):
    with SessionLocal() as session:
        new_prompt = Prompt(
            tool=tool,
            content_type=content_type,
            topic=topic,
            style=style,
            platform_name=platform_name,
            color_palette=color_palette,
            mood=mood,
            generated_text=generated_prompt,
            model_used=model_used,
            used_hybrid=bool(used_hybrid),
            user_id=user_id
        )
        session.add(new_prompt)
        session.commit()
        session.refresh(new_prompt)
        return new_prompt

def delete_prompt(prompt_id):
    with SessionLocal() as session:
        prompt = session.get(Prompt, prompt_id)
        if prompt:
            session.delete(prompt)
            session.commit()
            return True
        return False

def handle_delete(prompt_id):
    if delete_prompt(prompt_id):
        st.success("Prompt deleted.")
    else:
        st.error("Prompt not found.")


# session_state defaults
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None
if "last_search_tool" not in st.session_state:
    st.session_state["last_search_tool"] = "All"
if "last_search_topic" not in st.session_state:
    st.session_state["last_search_topic"] = ""
if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# generation/selection state
for key, default in [
    ("show_output", False),
    ("generated_text", None),
    ("generated_prompt_id", None),
    ("awaiting_hybrid_choice", False),
    ("hybrid_offline_text", None),
    ("hybrid_gemini_text", None),
    ("hybrid_offline_id", None),
    ("hybrid_gemini_id", None),
    ("hybrid_choice_model", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# admin UI
def admin_login_ui():
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
        if verify_admin_password(password):
            st.session_state["admin_logged_in"] = True
            st.success("✅ Admin login successful.")
            return True
        else:
            st.error("Invalid password.")
            return False
    return False


# Admin: Dashboard & Ranker controls
def show_dashboard():
    st.header("📊 Dashboard")
    st.markdown("Basic analytics: top topics/styles, feedback per model, active users.")

    top_topics = get_top_topics(10)
    if top_topics:
        topics = [t for t, c in top_topics]
        counts = [c for t, c in top_topics]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(topics, counts)
        ax.set_title("Top Topics")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    else:
        st.info("No prompt data for topics yet.")

    top_styles = get_top_styles(10)
    if top_styles:
        styles = [s or "(empty)" for s, c in top_styles]
        counts = [c for s, c in top_styles]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(styles, counts)
        ax.set_title("Top Styles")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
    else:
        st.info("No prompt data for styles yet.")

    # feedback_by_model = get_feedback_counts_by_model()
    # if feedback_by_model:
    #     models = []
    #     pos = []
    #     neg = []
    #     for model, stats in feedback_by_model.items():
    #         models.append(model)
    #         pos.append(stats.get("positive", 0))
    #         neg.append(stats.get("negative", 0))
    #     x = range(len(models))
    #     fig, ax = plt.subplots(figsize=(6, 3))
    #     width = 0.35
    #     ax.bar([i - width/2 for i in x], pos, width=width)
    #     ax.bar([i + width/2 for i in x], neg, width=width)
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(models, rotation=45)
    #     ax.legend(["Positive", "Negative"])
    #     st.pyplot(fig)
    # else:
    #     st.info("No feedback data yet.")

    active = get_active_users(20)
    if active:
        df = pd.DataFrame([{"username": u, "prompt_count": c} for u, c in active])
        st.subheader("Active users (by prompts created)")
        st.dataframe(df)
    else:
        st.info("No user activity yet.")


def admin_panel():
    st.title("🛠️ Admin Panel")
    menu = ["Dashboard", "Train Ranker", "View Users", "View Prompts", "Choices Dataset"]
    choice = st.selectbox("Choose view", menu, key="admin_menu_select")

    if choice == "Dashboard":
        show_dashboard()
        return

    if choice == "Train Ranker":
        st.header("Train / Retrain Ranker")
        st.markdown("Train the selector/ranker using recorded choices (preferred) or positive feedback.")

        emb_models = [
            "all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
            "paraphrase-MiniLM-L12-v2"
        ]
        selected_emb = st.selectbox("Embedding model", emb_models, index=0, key="emb_model_select")

        if st.button("Train embedding-based ranker (if available)"):
            # It will call ranker.train_with_embeddings using dataset from get_choice_dataset
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available. Generate both variants and choose a variant in the app to collect data.")
            else:
                texts, labels = zip(*dataset)
                try:
                    acc, rep = train_with_embeddings(list(texts), list(labels), embed_model_name=selected_emb, save_path=RANKER_PATH)
                    st.success(f"Embedding-based ranker trained (cv acc ≈ {acc:.3f})")
                    st.json(rep)
                except Exception as e:
                    st.error("Embedding train failed: " + str(e))

        if st.button("Train TF-IDF ranker (fallback)"):
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available.")
            else:
                texts, labels = zip(*dataset)
                try:
                    acc, rep = train_basic(list(texts), list(labels), save_path=RANKER_PATH)
                    st.success(f"TF-IDF ranker trained (cv acc ≈ {acc:.3f})")
                    st.json(rep)
                except Exception as e:
                    st.error("TF-IDF train failed: " + str(e))

        st.markdown("---")
        st.subheader("Model comparison (5-fold CV)")
        if st.button("Run quick comparison"):
            dataset = get_choice_dataset()
            if not dataset:
                st.error("No choice dataset available.")
            else:
                texts, labels = zip(*dataset)
                try:
                    results = compare_models(list(texts), list(labels), embed_model_name=selected_emb, cv=5)
                    st.json(results)
                    # Simple bar viz of mean accuracies
                    labels_x = []
                    means = []
                    for k, v in results.items():
                        labels_x.append(k)
                        means.append(float(v.get("mean", 0.0)))
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.bar(labels_x, means)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Mean CV Accuracy")
                    st.pyplot(fig)
                except Exception as e:
                    st.error("Comparison failed: " + str(e))
        return

    with SessionLocal() as session:
        if choice == "View Users":
            st.subheader("👤 Users")
            users = session.execute(select(User)).scalars().all()
            if users:
                df = pd.DataFrame([{
                    "id": u.id,
                    "username": u.username,
                    "email": u.email,
                    "role": u.role,
                    "created_at": u.created_at
                } for u in users])
                st.dataframe(df)
            else:
                st.info("No users found.")

        elif choice == "View Prompts":
            st.subheader("📝 Prompts")
            prompts = session.execute(select(Prompt)).scalars().all()
            if prompts:
                for p in prompts:
                    st.markdown(f"**ID:** {p.id} | **Tool:** {p.tool} | **Topic:** {p.topic} | **Model:** {p.model_used} | **HybridUsed:** {p.used_hybrid} | **User ID:** {p.user_id}")
                    st.code(p.generated_text, language="markdown")
                    col_d, col_sp = st.columns([0.6, 8])
                    with col_d:
                        st.button("🗑️ Delete", key=f"admin_delete_{p.id}", on_click=handle_delete, args=(p.id,))
                    st.markdown("---")
            else:
                st.info("No prompts found.")

        # elif choice == "View Feedback":
        #     st.subheader("💬 Feedback")
        #     feedbacks = session.execute(select(Feedback)).scalars().all()
        #     if feedbacks:
        #         df = pd.DataFrame([{
        #             "id": f.id,
        #             "rating": f.rating,
        #             "comments": f.comments,
        #             "timestamp": f.timestamp,
        #             "user_id": f.user_id,
        #             "prompt_id": f.prompt_id
        #         } for f in feedbacks])
        #         st.dataframe(df)
        #     else:
        #         st.info("No feedback found.")

        elif choice == "Choices Dataset":
            st.subheader("🎯 Choices dataset (for ranker)")
            rows = get_choice_dataset()
            if rows:
                st.info(f"{len(rows)} choice examples available.")
                df = pd.DataFrame([{"text": t, "label": l} for t, l in rows])
                st.dataframe(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="choices_dataset.csv", mime="text/csv")
            else:
                st.info("No choices recorded yet. Use Generate → Hybrid → Generate both and choose to record choices.")


# User auth UI
def user_auth_ui():
    if st.session_state.get("user"):
        return True

    st.subheader("🔑 User Login / Signup")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Login**")
        login_username = st.text_input("Username (login)", key="login_username")
        login_password = st.text_input("Password (login)", type="password", key="login_password")
        if st.button("Login as user", key="login_btn"):
            user = authenticate_user(login_username, login_password)
            if user:
                if not user.email:
                    st.error("Email required for login; user record missing email.")
                else:
                    st.session_state["user"] = {"id": user.id, "username": user.username, "role": getattr(user, "role", None)}
                    st.success(f"Welcome back, {user.username}!")
                    safe_rerun()
            else:
                st.error("Invalid username or password.")

    with col2:
        st.markdown("**Signup**")
        signup_username = st.text_input("Choose username", key="signup_username")
        signup_password = st.text_input("Choose password", type="password", key="signup_password")
        signup_email = st.text_input("Email (required)", key="signup_email")
        if st.button("Create account", key="signup_btn"):
            if not signup_username or not signup_password or not signup_email:
                st.error("Enter username, password, and email (email is mandatory).")
            else:
                success, payload = register_user(signup_username, signup_password, signup_email)
                if not success:
                    if payload == "username_exists":
                        st.error("Username already exists — choose another.")
                    elif payload == "email_exists":
                        st.error("Email already in use.")
                    elif payload == "email_required":
                        st.error("Email required.")
                    else:
                        st.error("Could not create account (db error).")
                else:
                    st.success("Account created. You can now log in.")
    return False

def account_sidebar():
    if not st.session_state.get("user"):
        return

    st.sidebar.markdown(f"👋 Logged in as **{st.session_state['user']['username']}**")
    st.sidebar.markdown("### Account")
    with st.sidebar.expander("🔒 Change password", expanded=False):
        old_pw = st.text_input("Current password", type="password", key="cp_old")
        new_pw = st.text_input("New password", type="password", key="cp_new")
        confirm_pw = st.text_input("Confirm new password", type="password", key="cp_confirm")
        if st.button("Change password", key="cp_btn"):
            if not old_pw or not new_pw:
                st.error("Enter both current and new password.")
            elif new_pw != confirm_pw:
                st.error("New password and confirm do not match.")
            else:
                username = st.session_state["user"]["username"]
                ok, msg = change_password(username, old_pw, new_pw)
                if ok:
                    st.success("Password changed successfully.")
                else:
                    if msg == "incorrect_old_password":
                        st.error("Current password is incorrect.")
                    else:
                        st.error("Could not change password. Try again.")

    if st.sidebar.button("🚪 Logout"):
        st.session_state["user"] = None
        st.success("Logged out successfully.")
        safe_rerun()


# Main app: prompt generation + history with choice recording
def run_prompt_generator():
    st.title("✨ Prompt Generator for AI Tools")
    st.markdown("Generate perfect prompts for **Gamma** and **Canva** with optional Gemini AI support!")

    st.sidebar.header("🔍 Examples")
    st.sidebar.markdown("- Gamma: Modern sales deck with key metrics and customer testimonials")
    st.sidebar.markdown("- Gamma: Educational presentation on renewable energy trends")
    st.sidebar.markdown("- Canva: Instagram story announcing a product launch")
    st.sidebar.markdown("- Canva: Facebook post graphic for a holiday sale")
    st.sidebar.markdown("---")
    account_sidebar()
    st.sidebar.markdown("👨‍💻 Made with ❤️ by [Parth Shinge](https://github.com/parth-shinge)")

    def _on_global_change():
        _clear_generation_state()
    model_choice = st.selectbox("Model choice:", ["Offline", "Gemini", "Hybrid"], key="model_choice_select", on_change=_on_global_change)
    tool = st.selectbox("Choose a tool:", ["Gamma", "Canva"], key="tool_select", on_change=_on_global_change)

    tab1, tab2 = st.tabs(["✨ Generate Prompt", "📂 Search History"])

    # Generate tab
    with tab1:
        if not user_auth_ui():
            st.info("Please login or sign up to generate and save prompts.")
            return

        st.subheader("✨ Generate a New Prompt")
        with st.form("generate_form", clear_on_submit=True):
            content_type = st.text_input("Content type (e.g. presentation, infographic, poster):", key="content_type_input")
            topic = st.text_input("Topic:", key="topic_input")
            style = st.text_input("Style (e.g. modern, playful, minimalist):", key="style_input")

            # canva extras
            platform = color_palette = mood = None
            if st.session_state.get("tool_select") == "Canva":
                platform = st.text_input("Platform (e.g. Instagram, Facebook):", key="canva_platform")
                color_palette = st.text_input("Color palette (e.g. bright, pastel, dark):", key="canva_colors")
                mood = st.text_input("Mood (e.g. energetic, calm, elegant):", key="canva_mood")

            if model_choice in ["Offline", "Gemini"]:
                gen_both_and_choose = st.checkbox("Generate both variants and choose (Hybrid mode)", value=False, key="gen_both_checkbox")
            else: 
                gen_both_and_choose = False

            submitted = st.form_submit_button("Generate Prompt")

        if submitted:
            # Clear outputs when inputs changed compared to last submission
            signature_parts = [
                st.session_state.get("content_type_input") or "",
                st.session_state.get("topic_input") or "",
                st.session_state.get("style_input") or "",
                st.session_state.get("canva_platform") or "",
                st.session_state.get("canva_colors") or "",
                st.session_state.get("canva_mood") or "",
                st.session_state.get("model_choice_select") or "",
                st.session_state.get("tool_select") or "",
            ]
            new_sig = "||".join(signature_parts)
            if st.session_state.get("last_input_signature") != new_sig:
                _clear_generation_state()
                st.session_state["last_input_signature"] = new_sig
            with st.spinner("Generating..."):
                used_hybrid_flag = False
                actual = None

                # When the checkbox is enabled in any mode, force hybrid-style generation (generate both)
                if gen_both_and_choose and model_choice in ("Offline", "Gemini"):
                    offline_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                    gemini_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    offline_obj = save_prompt(tool, content_type, topic, style, offline_text, model_used="offline", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)
                    gemini_obj = save_prompt(tool, content_type, topic, style, gemini_text, model_used="gemini", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)

                    used_hybrid_flag = True
                    st.session_state["awaiting_hybrid_choice"] = True
                    st.session_state["hybrid_offline_text"] = offline_text
                    st.session_state["hybrid_gemini_text"] = gemini_text
                    st.session_state["hybrid_offline_id"] = offline_obj.id
                    st.session_state["hybrid_gemini_id"] = gemini_obj.id
                    st.session_state["show_output"] = False

                elif model_choice == "Hybrid":
                    # Always generate both variants for Hybrid
                    offline_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                    gemini_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    offline_obj = save_prompt(tool, content_type, topic, style, offline_text, model_used="offline", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)
                    gemini_obj = save_prompt(tool, content_type, topic, style, gemini_text, model_used="gemini", user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=True)

                    used_hybrid_flag = True

                    if gen_both_and_choose:
                        # Ask the user to choose
                        st.session_state["awaiting_hybrid_choice"] = True
                        st.session_state["hybrid_offline_text"] = offline_text
                        st.session_state["hybrid_gemini_text"] = gemini_text
                        st.session_state["hybrid_offline_id"] = offline_obj.id
                        st.session_state["hybrid_gemini_id"] = gemini_obj.id
                        st.session_state["show_output"] = False
                    else:
                        # Auto-pick using ranker if available; else ask the user
                        try:
                            from ranker import load_ranker, predict_with_ranker
                            have_ranker = load_ranker(RANKER_PATH) is not None
                        except Exception:
                            have_ranker = False

                        if have_ranker:
                            topic_val = topic or ""
                            style_val = style or ""
                            txt = f"{topic_val} | {style_val} | OFFLINE: {offline_text} || GEMINI: {gemini_text}"
                            pred, _ = predict_with_ranker(txt, RANKER_PATH)
                            ai_choice = pred if pred in ("offline", "gemini") else random.choice(["offline", "gemini"])
                            # Record and show
                            chosen_id = offline_obj.id if ai_choice == "offline" else gemini_obj.id
                            uid = st.session_state["user"]["id"] if st.session_state.get("user") else None
                            record_choice(uid, offline_obj.id, gemini_obj.id, chosen_id, ai_choice)
                            with SessionLocal() as session:
                                saved = session.get(Prompt, chosen_id)
                            st.session_state["show_output"] = True
                            st.session_state["generated_text"] = saved.generated_text
                            st.session_state["generated_prompt_id"] = saved.id
                        else:
                            # No ranker: ask the user (do not randomize)
                            st.warning("No trained ranker found. Please choose a variant or train the ranker in Admin → Train Ranker.")
                            st.session_state["awaiting_hybrid_choice"] = True
                            st.session_state["hybrid_offline_text"] = offline_text
                            st.session_state["hybrid_gemini_text"] = gemini_text
                            st.session_state["hybrid_offline_id"] = offline_obj.id
                            st.session_state["hybrid_gemini_id"] = gemini_obj.id
                            st.session_state["show_output"] = False
                else:
                    actual = model_choice.lower()

                if model_choice != "Hybrid":
                    if actual == "gemini":
                        prompt_text = generate_gemini_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                        model_used = "gemini"
                    else:
                        prompt_text = generate_template_prompt(tool, content_type, topic, style, platform, color_palette, mood)
                        model_used = "offline"

                    user_id = st.session_state["user"]["id"] if st.session_state.get("user") else None
                    saved = save_prompt(tool, content_type, topic, style, prompt_text, model_used=model_used, user_id=user_id, platform_name=platform, color_palette=color_palette, mood=mood, used_hybrid=used_hybrid_flag)
                    st.session_state["show_output"] = True
                    st.session_state["generated_text"] = saved.generated_text
                    st.session_state["generated_prompt_id"] = saved.id

            if st.session_state.get("awaiting_hybrid_choice"):
                st.info("Two variants generated. Please choose one or let AI choose.")

        # When awaiting choice, render both prompts and provide actions
        if st.session_state.get("awaiting_hybrid_choice"):
            st.subheader("Compare Variants")
            st.markdown("---")
            st.markdown("**Offline variant:**")
            st.code(st.session_state.get("hybrid_offline_text") or "", language="markdown")
            st.markdown("**Gemini variant:**")
            st.code(st.session_state.get("hybrid_gemini_text") or "", language="markdown")

            col_left, col_mid, col_ai = st.columns([1, 1, 1])

            def _choose(model: str):
                uid = st.session_state["user"]["id"] if st.session_state.get("user") else None
                offline_id = st.session_state.get("hybrid_offline_id")
                gemini_id = st.session_state.get("hybrid_gemini_id")
                chosen_id = offline_id if model == "offline" else gemini_id
                ok, res = record_choice(uid, offline_id, gemini_id, chosen_id, model)
                if not ok:
                    st.error("Could not record choice: " + (res or "unknown reason"))
                    return
                # Show chosen result
                with SessionLocal() as session:
                    saved = session.get(Prompt, chosen_id)
                _clear_generation_state()
                st.session_state["show_output"] = True
                st.session_state["generated_text"] = saved.generated_text
                st.session_state["generated_prompt_id"] = saved.id
                st.success("Choice recorded — thanks!")

            with col_left:
                if st.button("✅ Keep Offline", key="keep_offline_btn"):
                    _choose("offline")
                    safe_rerun()
            with col_mid:
                if st.button("✅ Keep Gemini", key="keep_gemini_btn"):
                    _choose("gemini")
                    safe_rerun()
            with col_ai:
                if st.button("🤖 Let AI Choose", key="ai_choose_btn"):
                    with st.spinner("Letting ranker decide..."):
                        # Try ranker, fallback to random
                        try:
                            from ranker import load_ranker, predict_with_ranker
                            have_ranker = load_ranker(RANKER_PATH) is not None
                        except Exception:
                            have_ranker = False

                        if have_ranker:
                            # build text like dataset format for prediction
                            topic_val = st.session_state.get("topic_input") or ""
                            style_val = st.session_state.get("style_input") or ""
                            txt = f"{topic_val} | {style_val} | OFFLINE: {st.session_state.get('hybrid_offline_text') or ''} || GEMINI: {st.session_state.get('hybrid_gemini_text') or ''}"
                            try:
                                pred, _ = predict_with_ranker(txt, RANKER_PATH)
                                ai_choice = str(pred) if pred in ("offline", "gemini") else None
                            except Exception:
                                ai_choice = None
                        else:
                            ai_choice = None

                    if ai_choice is None:
                        st.warning("No trained ranker available or prediction failed. Please choose manually or train the ranker in Admin → Train Ranker.")
                    else:
                        _choose(ai_choice)
                        safe_rerun()

        if st.session_state.get("show_output") and st.session_state.get("generated_text"):
            st.markdown("---")
            st.subheader("Generated Prompt")
            st.success("✅ Prompt Generated & Saved:")
            st.code(st.session_state["generated_text"], language="markdown")

    # History tab
    with tab2:
        if not st.session_state.get("user"):
            st.info("Please login to view your prompt history.")
            return

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
            current_user_id = st.session_state["user"]["id"]
            if st.session_state["last_action"] == "search":
                stmt = select(Prompt).where(Prompt.user_id == current_user_id)
                if st.session_state["last_search_tool"] != "All":
                    stmt = stmt.where(Prompt.tool == st.session_state["last_search_tool"])
                if st.session_state["last_search_topic"]:
                    stmt = stmt.where(Prompt.topic.contains(st.session_state["last_search_topic"]))
                results = session.execute(stmt).scalars().all()
            elif st.session_state["last_action"] == "view_all":
                stmt = select(Prompt).where(Prompt.user_id == current_user_id)
                results = session.execute(stmt).scalars().all()

        if results:
            with st.expander("🔎 Show Results", expanded=True):
                for r in results:
                    st.markdown(
                        f"**Tool:** {r.tool} | **Topic:** {r.topic} | **Style:** {r.style} | **Model:** {r.model_used}"
                    )
                    if r.platform_name or r.color_palette or r.mood:
                        meta = []
                        if r.platform_name:
                            meta.append(f"Platform: {r.platform_name}")
                        if r.color_palette:
                            meta.append(f"Colors: {r.color_palette}")
                        if r.mood:
                            meta.append(f"Mood: {r.mood}")
                        st.caption(" • ".join(meta))

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
                st.info("No prompts found matching your filters or there were no prompts generated.")


# App entry
def main():
    st.set_page_config(page_title="Prompt Gen", page_icon="✨", layout="wide")
    app_mode = st.sidebar.radio("Navigation", ["Prompt Generator", "Admin Panel"], index=0)

    if app_mode == "Prompt Generator":
        run_prompt_generator()
    else:
        if st.session_state.get("admin_logged_in", False):
            if st.sidebar.button("Logout Admin"):
                st.session_state["admin_logged_in"] = False
                safe_rerun()
            admin_panel()
        else:
            logged_in_now = admin_login_ui()
            if logged_in_now:
                admin_panel()

if __name__ == "__main__":
    main()
