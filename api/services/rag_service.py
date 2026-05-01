from app.rag import answer_question


def ask_rag(question: str):

    result = answer_question(question)

    return result