
"""Streamlit UI for the AI lawyer assistant."""

from __future__ import annotations

import sys
from pathlib import Path

from openai import chat
import requests
import streamlit as st
import time
import re


from streamlit_mic_recorder import speech_to_text
import speech_recognition as sr 
import io 
import wave


from streamlit_agraph import (
    agraph,
    Node,
    Edge,
    Config
)




SUGGESTED_QUESTIONS = {
    "أجر": [
        "ما الفرق بين الأجر الأساسي والمتغير؟",
        "متى يستحق العامل العلاوة السنوية؟",
        "هل يجوز تخفيض أجر العامل؟"
    ],

    "إجازة": [
        "ما مدة الإجازة السنوية للعامل؟",
        "هل يحق للعامل إجازة مرضية؟",
        "متى تسقط الإجازات السنوية؟"
    ],

    "فصل": [
        "متى يجوز فصل العامل؟",
        "ما عقوبة الفصل التعسفي؟",
        "هل يحق للعامل التعويض؟"
    ],

    "عقد": [
        "ما الفرق بين العقد المحدد وغير المحدد؟",
        "هل يجوز إنهاء العقد المؤقت؟",
        "ما شروط عقد العمل؟"
    ]
}




_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

PRIMARY_K = 5

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="مساعد قانون العمل المصري",
    page_icon="⚖️",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>

html, body, [class*="st-"] {
    direction: rtl;
    text-align: right;
    background: #050816;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar Fix */
section[data-testid="stSidebar"] {
    min-width: 320px;
    background: rgba(17,24,39,0.72);
    backdrop-filter: blur(18px);

}

section[data-testid="stSidebar"] * {
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
}

/* Sidebar collapse fix */

button[kind="header"] {

    direction: ltr !important;

}

[data-testid="collapsedControl"] {

    direction: ltr !important;

    transform: rotate(0deg);

}

/* يمنع النص العمودي الغريب */

section[data-testid="stSidebar"] {

    overflow-x: hidden !important;
}



/* TextArea */
.stTextArea textarea {
    direction: rtl;
    text-align: right;
    font-size: 1.1rem !important;
    border-radius: 18px !important;
    background: #1c1f2e !important;
    color: white !important;
border: 1px solid rgba(255,255,255,0.08) !important;
transition: all 0.25s ease;

}


.stTextArea textarea:focus {

    border: 1px solid rgba(255,255,255,0.18) !important;

    box-shadow: 0 0 20px rgba(255,255,255,0.04);

}


.answer-box,
.custom-card,
.source-card {

    animation: fadeUp 0.35s ease;

}

@keyframes fadeUp {

    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0px);
    }

}






/* Buttons */
.stButton button {
    border-radius: 12px !important;

    background: rgba(255,255,255,0.03) !important;

    color: white !important;

    border: 1px solid rgba(255,255,255,0.12) !important;

    transition: all 0.25s ease;

    font-weight: 600;

    backdrop-filter: blur(10px);
}

.stButton button:hover {

    border: 1px solid rgba(255,255,255,0.25) !important;

    background: rgba(255,255,255,0.06) !important;

    transform: translateY(-1px);

    box-shadow: 0 0 12px rgba(255,255,255,0.08);
}



/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
}

/* Cards */
.custom-card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 20px;
}

/* Answer */
.answer-box {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    line-height: 2;
    font-size: 1.05rem;
}

/* Source Card */
.source-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* ========================= */
/* Chat Bubbles */
/* ========================= */

.user-bubble {

    background: rgba(255,255,255,0.04);

    border: 1px solid rgba(255,255,255,0.06);

    padding: 18px;

    border-radius: 18px;

    margin-bottom: 14px;

    line-height: 1.9;

    animation: fadeUp 0.4s ease;

}

.assistant-bubble {

    background: rgba(255,75,75,0.06);

    border: 1px solid rgba(255,75,75,0.14);

    padding: 22px;

    border-radius: 18px;

    margin-bottom: 22px;

    line-height: 2;

    animation: fadeUp 0.5s ease;
}


/* Hide Streamlit Footer */
footer {
    visibility: hidden;
}


/* ───────────────────────── */
/* Sidebar Toggle Fix */
/* ───────────────────────── */

button[kind="header"] {

    position: relative !important;

    color: transparent !important;

}

/* اخفاء النص القديم */

button[kind="header"] svg {

    display: none !important;

}

/* اخفاء أي text غريب */

button[kind="header"] * {

    color: transparent !important;

    font-size: 0px !important;

}

/* الأيقونة الجديدة */

button[kind="header"]::after {

    content: "⚙️";

    position: absolute;

    left: 12px;

    top: 8px;

    font-size: 20px;

    color: white !important;

}

/* hover */

button[kind="header"]:hover {

    background: rgba(255,255,255,0.05) !important;

    border-radius: 12px;

}



/* ───────────────────────────── */
/* FIX STREAMLIT MATERIAL ICON TEXT */
/* ───────────────────────────── */

/* اخفاء أي text غريب للأيقونات */

span.material-symbols-rounded {
    
    font-size: 0px !important;
    
    visibility: hidden !important;
    
    position: relative;
}

/* Sidebar toggle icon */

span.material-symbols-rounded::after {

    content: "☰";
    
    visibility: visible !important;

    font-size: 20px !important;

    color: white !important;

    position: absolute;

    inset: 0;
}

/* Menu icon (الثلاث نقاط) */

button[title="Main menu"] span.material-symbols-rounded::after {

    content: "⋮";

    font-size: 22px !important;
}

/* تحسين شكل الأزرار العلوية */

button[kind="header"],
button[title="Main menu"] {

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;
}

/* Hover محترم */

button[kind="header"]:hover,
button[title="Main menu"]:hover {

    background: rgba(255,255,255,0.06) !important;

    border-radius: 10px;
}





</style>
""",
    unsafe_allow_html=True,
)
  

# ─────────────────────────────────────────────────────────────
# HIGHLIGHT KEYWORDS
# ─────────────────────────────────────────────────────────────

def highlight_keywords(text: str, question: str) -> str:

    # تنظيف السؤال من الرموز
    cleaned_question = re.sub(r"[^\w\s]", "", question)

    words = [
        w.strip()
        for w in cleaned_question.split()
        if len(w.strip()) > 2
    ]

    highlighted = text

    for word in words:

        pattern = rf"({re.escape(word)})"

        highlighted = re.sub(
            pattern,
            r'''
<span style="
background: rgba(255,75,75,0.22);
padding:2px 6px;
border-radius:6px;
font-weight:700;
color:#fff;
">\1</span>
''',
            highlighted,
            flags=re.IGNORECASE,
        )

    return highlighted



## ─────────────────────────────────────────────────────────────
## SUGGESTED QUESTIONS
## ─────────────────────────────────────────────────────────────    
def get_suggested_questions(question: str):

    for keyword, suggestions in SUGGESTED_QUESTIONS.items():

        if keyword in question:

            return suggestions

    return [
        "ما هي حقوق العامل؟",
        "ما واجبات صاحب العمل؟",
        "ما عقوبة مخالفة قانون العمل؟"
    ]





# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "question" not in st.session_state:
    st.session_state["question"] = ""

if "auto_ask" not in st.session_state:
    st.session_state["auto_ask"] = False

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## ⚙️ الإعدادات")

    primary_k = st.slider(
        "عدد النتائج الأساسية (Top-K)",
        min_value=1,
        max_value=15,
        value=PRIMARY_K,
        step=1,
    )

    st.caption(
        "التوصية: 5 نتائج أساسية ثم توسيع عبر الرسم البياني."
    )

    st.divider()

    st.markdown("## 🤖 النماذج")

    st.markdown("""
- **التضمين:** Azure OpenAI text-embedding-3-large  
- **التوليد:** Azure OpenAI gpt-4.1  
- **التخزين:** Neo4j (GraphRAG)
""")

    st.divider()

    st.markdown("## 💡 أمثلة على الأسئلة")

    examples = [
        "ما هو تعريف الأجر؟",
        "ما هي مدة الإجازة السنوية للعامل؟",
        "ما هي عقوبة مخالفة المادة 70؟",
        "ما هي حقوق العامل أثناء إجازة الحج؟",
        "متى يجوز فصل العامل؟",
        "ما الفرق بين العمل المؤقت والعمل العرضى؟",
    ]

    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question"] = ex
            st.session_state["auto_ask"] = True
            st.rerun()

# ─────────────────────────────────────────────────────────────
# CLEAR CHAT
# ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([8, 2])

with col2:
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# ─────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<h1 style="
    text-align:center;
    font-size:3.2rem;
    background: linear-gradient(90deg,#ffffff,#ff4b4b);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    font-weight:800;
">
⚖️ مساعد قانون العمل المصري
</h1>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<p style='text-align:center;color:#999'>
قانون رقم 14 لسنة 2025 — للاستخدام التجريبي فقط وليس بديلاً عن استشارة محامٍ مختص.
</p>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────


for chat_idx, chat in enumerate(st.session_state["chat_history"]):



    # ─────────────────────────────────────
    # TIMING METRICS
    # ─────────────────────────────────────

    elapsed = chat.get("elapsed_ms", {})

    if elapsed:

        total_time = sum(elapsed.values())

        st.caption(
            f"⏱️ التوجيه: {elapsed.get('route',0)} ms · "
            f"التضمين: {elapsed.get('embed',0)} ms · "
            f"الاسترجاع: {elapsed.get('retrieve',0)} ms · "
            f"التوليد: {elapsed.get('generate',0)} ms · "
            f"الإجمالي: {total_time} ms"
        )


    # ─────────────────────────────────────
    # EXPLICIT REFERENCES
    # ─────────────────────────────────────

    detected_refs = chat.get("detected_refs", [])

    if detected_refs:

        refs_str = "، ".join(
            f"المادة {n}"
            for n in detected_refs
        )

        st.info(
            f"📌 تم رصد إشارة صريحة إلى: {refs_str}"
        )
        
        
    # USER MESSAGE
    st.markdown(
        f"""
<div class="user-bubble">

👤 <strong>أنت</strong>

<br><br>

{chat["question"]}

</div>
""",
        unsafe_allow_html=True,
    )

    # ASSISTANT MESSAGE
    st.markdown(
        f"""
<div class="assistant-bubble">

⚖️ <strong>المساعد القانوني</strong>

<br><br>

{chat["answer"]}

</div>
""",
        unsafe_allow_html=True,
    )


    # ─────────────────────────────────────
    # SOURCES
    # ─────────────────────────────────────

    with st.expander(
        f"📚 المصادر المرجعية ({len(chat['sources'])})",
        expanded=False
    ):

        for source in chat["sources"]:

            source_type = source.get("source", "primary")
            
            node_color = {
            "primary": "#3b82f6",
            "cross_ref": "#22c55e",
            "definition": "#a855f7",
            "explicit": "#ff4b4b",
        }.get(source_type, "#3b82f6")

            source_label = {
                "explicit": "📌 مادة مطلوبة صراحة",
                "primary": "🎯 نتيجة أساسية",
                "definition": "📖 تعريف موسع",
                "cross_ref": "🔗 مادة مُحال إليها",
            }.get(source_type, "📄 مصدر قانوني")

            st.markdown(
                f"""
<div class="source-card">

### {source_label} — المادة {source['article']}

**درجة التشابه:** {source['score']:.3f}

<small style="color:#999">
كلما زادت الدرجة زاد ارتباط المادة بالسؤال.
</small>

<br><br>

{highlight_keywords(source['excerpt'], chat['question'])}

</div>
""",
                unsafe_allow_html=True,
            )
   

    # ─────────────────────────────────────
    # Suggested Questions
    # ─────────────────────────────────────

    st.markdown("### 💡 قد يهمك أيضًا")

    suggestions = get_suggested_questions(chat["question"])

    cols = st.columns(len(suggestions))

    for idx, suggestion in enumerate(suggestions):

        with cols[idx]:

            if st.button(
                suggestion,
                
            key=f"suggestion_{chat_idx}_{idx}_{chat['question']}"


            ):

                st.session_state["question"] = suggestion

                st.session_state["auto_ask"] = True

                st.rerun()

   
    
    # ─────────────────────────────────────
    # GRAPH VISUALIZATION
    # ─────────────────────────────────────

    st.markdown("### 🕸️ الرسم البياني للعلاقات")

    nodes = []

    edges = []

    added_articles = set()

    # السؤال الرئيسي
    nodes.append(
        Node(
            id="question",
            label="سؤال المستخدم",
            size=30,
            color="#ff4b4b"
        )
    )

    for source in chat["sources"]:

        article_id = f"article_{source['article']}"

        if article_id not in added_articles:

            nodes.append(
                Node(
                    id=article_id,
                    label=f"مادة {source['article']}",
                    size=24,
                    color=node_color
                )
            )

            edges.append(
                Edge(
                    source="question",
                    target=article_id
                )
            )

            added_articles.add(article_id)

    config = Config(
        width="100%",
        height=400,
        directed=True,
        physics=True,
        hierarchical=False,
    )

    agraph(
        nodes=nodes,
        edges=edges,
        config=config
    )









# ─────────────────────────────────────────────────────────────
# VOICE INPUT
# ─────────────────────────────────────────────────────────────

if "last_voice_text" not in st.session_state:
    st.session_state["last_voice_text"] = ""

voice_text = speech_to_text(
    language="ar",
    start_prompt="🎤 اضغط للتحدث",
    stop_prompt="⏹️ جاري التحويل...",
    use_container_width=True,
    just_once=False,
    key="voice_recorder",
)

if (
    voice_text
    and voice_text != st.session_state["last_voice_text"]
):

    st.session_state["last_voice_text"] = voice_text

    st.session_state["question"] = voice_text

    st.session_state["auto_ask"] = True

    st.rerun()



# ─────────────────────────────────────────────────────────────
# QUESTION INPUT
# ─────────────────────────────────────────────────────────────

question = st.text_area(
    "اكتب سؤالك القانوني بالعربية:",
    value=st.session_state.get("question", ""),
    height=120,
    placeholder="مثال: ما هي مدة الإجازة السنوية للعامل؟"
)




submit = st.button(
    "⚖️ اسأل الآن",
    use_container_width=True
)


auto_ask = st.session_state.get("auto_ask", False)

# ─────────────────────────────────────────────────────────────
# ASK
# ─────────────────────────────────────────────────────────────
if (submit or auto_ask) and question.strip():

    thinking_placeholder = st.empty()

    thinking_placeholder.markdown(
        """
<div class="custom-card">
<h3>⚖️ جارِ تحليل مواد القانون...</h3>
</div>
""",
        unsafe_allow_html=True,
    )

    try:

        api_response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={
                "question": question.strip()
            },
            timeout=60
            
        )

        response = api_response.json()
        
        if api_response.status_code != 200:

            st.error("حدث خطأ أثناء الاتصال بالـ API")

            st.stop() 
        
        if not response.get("sources"):
            thinking_placeholder.empty()

            st.warning(
                "لم نجد مواد قانونية مرتبطة بالسؤال. حاول إعادة صياغته."
            )

            st.stop()

        thinking_placeholder.empty()

        

        full_answer = response["answer"]

        streamed_text = ""

        assistant_placeholder = st.empty()

        for word in full_answer.split():

            streamed_text += word + " "

            assistant_placeholder.markdown(
                f"""
<div class="assistant-bubble">

⚖️ <strong>المساعد القانوني</strong>

<br><br>

{streamed_text}

</div>
""",
                unsafe_allow_html=True,
            )

            time.sleep(0.03)

        assistant_placeholder.empty()

        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": full_answer,
                "sources": response["sources"],
                "elapsed_ms": response.get("elapsed_ms", {}),
               "detected_refs": response.get("detected_refs" , [])
            }
        )


        st.session_state["auto_ask"] = False

        st.rerun()

    except Exception as e:

        thinking_placeholder.empty()

        err_text = str(e)

        if "Neo.ClientError" in err_text:
            st.error("تعذر الاتصال بـ Neo4j")

        elif "AuthenticationError" in err_text:
            st.error("مشكلة في Azure OpenAI Key")

        else:
            st.error(f"حدث خطأ: {err_text}")

# ─────────────────────────────────────────────────────────────
# EMPTY QUESTION
# ─────────────────────────────────────────────────────────────
elif submit:
    st.warning("الرجاء كتابة سؤال أولاً.")

