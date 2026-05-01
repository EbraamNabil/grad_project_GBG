"""Streamlit UI for the AI lawyer assistant.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.rag import answer_question, PRIMARY_K  # noqa: E402

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="مساعد قانون العمل المصري",
    page_icon="⚖️",
    layout="wide",
)

# RTL styling for Arabic content.
st.markdown(
    """
    <style>
    html, body, [class*="st-"] { direction: rtl; text-align: right; }
    .stTextArea textarea { direction: rtl; text-align: right; font-size: 1.05rem; }
    .stMarkdown { direction: rtl; text-align: right; }
    code { direction: ltr; display: inline-block; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### الإعدادات")
    primary_k = st.slider("عدد النتائج الأساسية (Top-K)", min_value=1, max_value=15, value=PRIMARY_K, step=1)
    st.caption(
        "التوصية: 5 نتائج أساسية ثم توسيع عبر الرسم البياني للتعريفات والإحالات. "
        "زيادة العدد فوق 10 قد تُدخل ضوضاء، وإنقاصه تحت 3 قد يُفقد إجابة."
    )
    st.divider()
    st.markdown("### نماذج")
    st.markdown(
        "- **التضمين:** Azure OpenAI text-embedding-3-large\n"
        "- **التوليد:** Azure OpenAI gpt-4.1\n"
        "- **التخزين:** Neo4j (GraphRAG)"
    )
    st.divider()
    st.markdown("### أمثلة على الأسئلة")
    examples = [
        "ما هو تعريف الأجر؟",
        "ما هي مدة الإجازة السنوية للعامل؟",
        "ما هي عقوبة مخالفة المادة 70؟",
        "ما هي حقوق العامل أثناء إجازة الحج؟",
        "متى يجوز فصل العامل؟",
        "ما الفرق بين العمل المؤقت والعمل العرضى؟",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["question"] = ex


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("⚖️ مساعد قانون العمل المصري")
st.caption("قانون رقم 14 لسنة 2025 — للاستخدام التجريبي فقط، وليس بديلاً عن استشارة محامٍ مختص.")

if "question" not in st.session_state:
    st.session_state["question"] = ""

question = st.text_area(
    "اكتب سؤالك القانوني بالعربية:",
    value=st.session_state.get("question", ""),
    height=110,
    key="question_input",
    placeholder="مثال: ما هي مدة الإجازة السنوية للعامل؟",
)

submit = st.button("اسأل", type="primary", use_container_width=False)

if submit and question.strip():
    with st.spinner("جارِ البحث في القانون وصياغة الإجابة..."):
        try:
            response = answer_question(question.strip(), primary_k=primary_k)
        except Exception as e:
            err_text = str(e)
            if "Neo.ClientError" in err_text or "ServiceUnavailable" in err_text or "ConnectionRefused" in err_text:
                st.error(
                    "تعذر الاتصال بقاعدة البيانات Neo4j. "
                    "تحقق من تشغيل Neo4j محلياً وصحة بيانات `.env`."
                )
            elif "DeploymentNotFound" in err_text or "Resource not found" in err_text:
                st.error(
                    "لم يتم العثور على نموذج Azure OpenAI. "
                    "تحقق من اسم النشر `AZURE_OPENAI_CHAT_DEPLOYMENT` و `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` في `.env`."
                )
            elif "AuthenticationError" in err_text or "401" in err_text:
                st.error("مفتاح Azure OpenAI غير صحيح أو منتهي الصلاحية. حدّث `AZURE_OPENAI_KEY` في `.env`.")
            else:
                st.error(f"حدث خطأ: {err_text}")
            st.stop()

    if not response.chunks:
        st.warning("لم نجد نصوصاً مطابقة في القانون. حاول إعادة صياغة السؤال.")
    else:
        st.markdown("### الإجابة")
        st.markdown(response.answer)

        st.caption(
            f"⏱️ زمن الاستجابة: تضمين {response.elapsed_ms.get('embed', 0)} ms · "
            f"استرجاع {response.elapsed_ms.get('retrieve', 0)} ms · "
            f"توليد {response.elapsed_ms.get('generate', 0)} ms · "
            f"الإجمالي {sum(response.elapsed_ms.values())} ms"
        )

        with st.expander(f"📚 المصادر المرجعية ({len(response.chunks)} قطعة)", expanded=False):
            for i, c in enumerate(response.chunks, start=1):
                source_label = {
                    "primary": "🎯 نتيجة أساسية",
                    "definition": "📖 تعريف موسّع",
                    "cross_ref": "🔗 مادة مُحال إليها",
                }.get(c.source, c.source)

                score_text = f"تشابه: {c.score:.3f}" if c.score > 0 else "توسعة من الرسم البياني"
                st.markdown(
                    f"**{i}. {source_label}** — `{c.node_id}` · {score_text}"
                )
                if c.breadcrumb:
                    st.caption(c.breadcrumb)
                st.text(c.text[:600] + ("..." if len(c.text) > 600 else ""))
                st.divider()
elif submit:
    st.warning("الرجاء كتابة سؤال أولاً.")
