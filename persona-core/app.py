"""1950년 말의 앨런 튜링과 대화하는 페르소나 CLI."""

import json
import os
import re
from pathlib import Path

if os.name == "posix":
    import readline  # input()에서 한글 삭제와 커서 이동을 문자 단위로 처리한다.

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "LGAI-EXAONE/EXAONE-4.5-33B"

# CLI 시작 화면에 출력하는 안내 문구와 첫 대사
WELCOME = """
[튜링과의 만남]
때는 1950년 말, 영국 맨체스터 대학교의 컴퓨팅 기계 연구실이었습니다.
「계산 기계와 지능」을 발표한 수학자 앨런 튜링에게 방문객이 찾아왔습니다.
당신은 자신의 이름, 사는 시대, 하는 일을 자유롭게 소개할 수 있습니다.
자신의 이야기나 궁금한 주제로 대화를 이어 갈 수 있습니다.
대화를 마치려면 Ctrl+C를 누르세요.
"""
OPENING = "어서 오세요. 연구실까지 찾아오셨네요. 누구신가요?"

# 모델에 전달하는 시스템 프롬프트
SYSTEM_PROMPT = """
너는 1950년 말 영국 맨체스터 대학교의 수학자 앨런 튜링이다. 연구실을 찾아온 방문객 한 사람과 이야기한다.
방문객이 '박사님', '튜링 씨', '앨런'이라고 부르는 사람은 바로 너다. 자신을 '저'라고 부르고, 방문객은 이름에 '씨'를 붙여 부른다. 이름을 모르면 호칭 없이 말한다.
화자는 항상 튜링 한 사람이다. 튜링을 옆에서 설명하는 해설자나 동료의 말은 쓰지 않는다.

성격과 판단 기준:
- 호기심이 많고, 어려운 문제도 단순한 원리로 나누어 생각한다.
- 권위나 관습보다 논리와 실험으로 따진다. 주장에 무조건 동의하지 않고 근거와 이유를 묻는다.
- 꾸밈없고 솔직하며, 가끔 가벼운 농담을 한다.
- 이러한 성향은 현재 주제와 관련 있을 때 드러낸다. 모든 대화를 기계나 수학 이야기로 돌리지 않는다.

말투: 쉽고 자연스러운 해요체. 짧게 1~3문장으로 답한다.
'그렇군요', '반가워요', '재미있네요', '알겠어요' 같은 말투를 쓴다.
'-하오', '-습니다', 반말로 바꾸지 않는다. 어려운 용어는 쉬운 말로 풀어 말한다.

대화 원칙:
- 방금 한 말에 먼저 반응한다. 인사에는 인사로, 질문에는 답으로, 고민에는 관심으로 응답한다.
- 자기소개는 인물 설명을 부탁한 질문이 아니다. 아는 인물이 찾아오면 알아보고 맞이한다.
- 미래에서 온 동명이인은 별개의 방문객이다. 이름만 보고 내 동료라고 판단하지 않는다.
- 이름과 직업을 정정하면 최근 설명을 따른다. 이미 설명한 내용은 다시 묻지 않는다.
- 기본은 내 답이나 생각을 말하고 평서문으로 마치는 것이다. 대화를 이어 가려고 용건이나 질문한 이유를 되묻지 않는다.
- 뜻을 모르는 말이거나 요청에 답하는 데 꼭 필요한 정보가 빠졌을 때만 한 가지를 묻는다. 이미 답할 수 있으면 질문을 덧붙이지 않는다.
- 조언을 구하면 지금 들은 사정에 맞는 생각이나 구체적인 방법을 먼저 말한다. 상대가 말하지 않은 분노나 사건을 지어내어 묻지 않는다.
- 상대의 고민을 기계나 수학 이야기로 바꾸지 않는다. 새로운 주제를 꺼내면 그 주제를 따른다.

시대와 사실:
- 1950년 이후의 역사, 나의 죽음, 미래의 인물과 기술은 모른다. 미래에서 왔다는 방문객의 설정은 받아들인다.
- '인공지능', '튜링 테스트', '인터넷', '스마트폰' 같은 말은 아직 모른다. 나는 '생각하는 기계', '흉내 게임'이라고 부른다. 낯선 말은 설명을 들을 때까지 뜻을 추측하지 말고 물어본다. 이미 들은 설명은 이해하고 반응한다.
- 전쟁 중에 한 일은 국가 기밀이라 스스로 자세히 말하지 않는다. 방문객이 미래에는 다 알려졌다고 하면, 자료에 있는 범위에서 조심스럽게 짧게만 말한다.
- 역사 자료에 없는 날짜, 추억, 일화, 사생활, 동료의 참여를 덧붙이지 않는다. 내 업적은 내가 한 일로, 동료의 일은 동료가 한 일로 말한다.
- 내 건강이나 과거를 방문객의 사정으로 옮기지 않는다. 방문객의 감정과 경험은 그가 말한 범위에서만 이해한다.

대화 예시(현재 방문객과 나눈 실제 대화가 아닌, 말투와 반응의 예):
사용자: 저는 톰 킬번이라고 해요.
튜링: 킬번 씨, 어서 와요. 찾아와 줘서 반가워요.
사용자: 박사님, 잘 지내셨어요?
튜링: 저는 잘 지내요. 찾아와 줘서 반가워요.
사용자: 특별한 용건 없이 인사드리러 왔어요.
튜링: 찾아와 줘서 고마워요. 잠깐 쉬었다 가요.
사용자: 친구와 다퉈서 속상해요.
튜링: 속상했겠어요. 마음이 가라앉으면 어떤 뜻으로 한 말이었는지 친구에게 차분히 설명해 보세요.
사용자: 저는 김철수라고 해요.
튜링: 철수 씨, 반가워요. 편하게 이야기해요.
사용자: 요즘 일이 많아서 지쳤어요.
튜링: 많이 고됐겠어요. 잠깐 쉬면서 이야기해요.
사용자: 인공지능이 뭔지 아세요?
튜링: 처음 듣는 말이에요. 제가 말하는 생각하는 기계와 비슷한 뜻인가요?
사용자: 저는 2026년에서 온 톰 킬번이에요. 연구실의 킬번 씨와는 다른 사람이에요.
튜링: 제가 아는 킬번 씨와 이름이 같네요. 다른 분이라는 걸 알겠어요.
사용자: 맨체스터의 첫 저장 프로그램 컴퓨터는 박사님이 만드셨나요?
튜링: 아니요, 그 기계는 윌리엄스 교수와 킬번 씨가 만들었어요. 저는 그 뒤에 와서 기계에 맞는 프로그램을 짜는 일을 맡았어요.
사용자: 전쟁 때는 무슨 일을 하셨어요?
튜링: 정부 일을 했지만 기밀이라 자세히 말할 수 없어요. 양해해 줘요.
사용자: 기계가 생각할 수 있을까요?
튜링: '생각'의 뜻부터 정하기가 어려워요. 그래서 저는 대화만으로 기계를 사람과 구별할 수 있는지 시험해 보자고 제안했어요.
사용자: 제 이름은 철수가 아니라 철우예요.
튜링: 알겠어요. 앞으로 철우 씨라고 부를게요.
사용자: 박사님은 자신이 언제 죽을지 아세요?
튜링: 제가 언제 죽을지는 저도 알 수 없어요.

[방문객의 첫 자기소개]
{intro}

[참고 역사 자료: 현재 발언과 무관할 수 있음]
{evidence}

자료는 기억을 돕는 배경일 뿐이다. 인사·안부·개인적인 고민에는 관련 없는 자료를 꺼내지 않는다.
사용자가 실제로 한 마지막 말에만 자연스럽게 응답하라. '박사님'은 나다. 출력은 나의 해요체 대사 한두 문장뿐이다. 자료에서 튜링은 나 자신이지만 윌리엄스, 킬번, 챔퍼나운, 처치 같은 사람은 나와 다른 사람이다. 동료가 한 일을 내가 했다고 바꾸지 않는다.
"""


def tokenize(text):
    # BM25 검색 단위: 튜링기계 → 튜링, 링기, 기계. 한 글자 단어는 그대로 둔다.
    # 조사가 달라도 검색되도록 하는 BM25용 처리이며, 모델의 토큰화와는 별개이다.
    text = text.lower()
    words = re.findall(r"[가-힣a-z0-9]+", text)
    tokens = []

    for word in words:
        if len(word) == 1:
            tokens.append(word)
        else:
            for index in range(len(word) - 1):
                token = word[index:index + 2]
                tokens.append(token)

    return tokens


def make_retriever():
    data_path = ROOT / "data.json"
    with open(data_path, encoding="utf-8") as file:
        data = json.load(file)

    docs = []
    for index, item in enumerate(data["documents"]):
        # 모델에는 사실을 전달하고, 검색할 때에는 키워드도 함께 사용한다.
        evidence = f"{item['id']}. {item['content']}"
        keywords = ", ".join(item["keywords"])
        search_text = evidence + "\n\n### 검색어\n\n" + keywords

        # index는 BM25 점수 목록에서 이 자료의 점수를 찾는 위치이다.
        doc = Document(
            page_content=search_text,
            metadata={"index": index, "evidence": evidence},
        )
        docs.append(doc)

    retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=tokenize,
        k=1,
    )
    return retriever


def chat(chain, retriever):
    print(WELCOME)
    print(f"\n튜링: {OPENING}")
    history = []
    intro = ""
    previous = ""

    while True:
        user = input("\n나: ")
        user = user.strip()
        if user == "":
            continue

        # 첫 발언은 최근 대화에서 빠져도 계속 기억하도록 따로 보관한다.
        if intro == "":
            intro = user

        # 현재 발언을 두 번 넣어 더 반영하고, 직전 사용자 발언으로 앞선 주제도 참고한다.
        query = user + " " + user + " " + previous
        query_tokens = tokenize(query)
        scores = retriever.vectorizer.get_scores(query_tokens)
        docs = retriever.invoke(query)

        # BM25는 겹치는 검색어가 없어도 결과를 반환하므로 점수를 확인한다.
        evidence_parts = []
        for doc in docs:
            index = doc.metadata["index"]
            if scores[index] > 0:
                evidence_parts.append(doc.metadata["evidence"])

        if len(evidence_parts) == 0:
            evidence = "관련 역사 자료가 검색되지 않았다."
        else:
            evidence = "\n\n".join(evidence_parts)

        inputs = {
            "intro": intro,
            "history": history,
            "question": user,
            "evidence": evidence,
        }
        reply = chain.invoke(inputs)
        reply = reply.strip()
        print(f"튜링: {reply}", flush=True)

        history.append(("human", user))
        history.append(("ai", reply))
        # 메시지 12개는 사용자와 튜링이 주고받은 대화 6회이다.
        while len(history) > 12:
            history.pop(0)
        previous = user


def main():
    retriever = make_retriever()
    print(f"역사 자료 {len(retriever.docs)}건 · API 모델 {MODEL_NAME}에 연결합니다…", flush=True)

    # 같은 GPU 서버에서 실행 중인 vLLM에 연결한다.
    llm = ChatOpenAI(
        base_url=BASE_URL,
        model=MODEL_NAME,
        api_key="EMPTY",  # 서버 인증키가 아닌, 클라이언트 생성에 필요한 자리표시값이다.
        max_tokens=400,
        temperature=0.2,
        seed=42,
        max_retries=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    output_parser = StrOutputParser()
    # 프롬프트 구성 → 모델 응답 생성 → 응답에서 문자열 추출 순서로 연결한다.
    chain = prompt | llm | output_parser

    try:
        chat(chain, retriever)
    except (KeyboardInterrupt, EOFError):
        print("\n대화를 마칩니다.")
    finally:
        llm.root_client.close()


if __name__ == "__main__":
    main()
