from scripts.chatbot_rag import ask_rag


def test_ask_rag_returns_string():
    question = "Quels événements musicaux sont prévus à Lille ?"
    answer = ask_rag(question)

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


def test_ask_rag_general_question():
    question = "concert Lille"
    answer = ask_rag(question)

    assert isinstance(answer, str)
    assert len(answer) > 0


def test_ask_rag_no_crash():
    question = "asdlkjasdlkj"  # question un peu random
    answer = ask_rag(question)

    assert isinstance(answer, str)