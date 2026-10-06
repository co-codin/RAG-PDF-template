from langchain_core.messages import HumanMessage, SystemMessage

from rag_app.infrastructure.llm import FakeChatModel
from rag_app.rag.prompts import ANSWER_HUMAN, ANSWER_SYSTEM


def test_fake_answers_from_context_and_streams():
    prompt = ANSWER_HUMAN.format(context="[1] (source: a.md)\nThe battery lasts 11 hours on a full charge.",
                                 question="How long does the battery last?")
    msgs = [SystemMessage(ANSWER_SYSTEM), HumanMessage(prompt)]
    llm = FakeChatModel()
    answer = llm.invoke(msgs).content
    assert "11 hours" in answer and answer.endswith("[1]")
    assert "".join(c.content for c in llm.stream(msgs)) == answer


def test_fake_says_unknown_without_context():
    prompt = ANSWER_HUMAN.format(context="(no context)", question="Who?")
    assert "don't know" in FakeChatModel().invoke([HumanMessage(prompt)]).content
