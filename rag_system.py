import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 설정
load_dotenv()
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001" 
LLM_MODEL = "gemini-2.5-flash"

def run_lawlens_analysis(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not os.path.exists(DB_PATH):
        return {"result": "오류: API 키가 없거나 DB가 없습니다.", "docs": [], "scores": []}
        
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings,
        collection_name="lawlens_cases"
    )
    
    # 1. 넉넉하게 10개 검색
    results = vector_store.similarity_search_with_relevance_scores(query, k=10)
    
    if not results:
        return {"result": "죄송합니다. 유사한 판례를 찾을 수 없습니다.", "docs": [], "scores": []}

    # 2.유죄/무죄 분류
    guilty_cases = []
    other_cases = []
    
    for doc, score in results:
        judgment = doc.metadata.get("judgment", "")
        # 유죄 시그널 확인
        if "유죄" in judgment or "벌금" in judgment or "징역" in judgment or "선고유예" in judgment:
            guilty_cases.append((doc, score))
        else:
            other_cases.append((doc, score))

    # 3. 메인 케이스 선정 및 상황 판단
    is_guilty_found = False
    
    if guilty_cases:
        # 유죄가 있으면 -> 그걸 메인으로 (성공!)
        main_case, main_score = guilty_cases[0]
        remaining = guilty_cases[1:] + other_cases
        is_guilty_found = True
        section_title = "🏆 유사 승소 사례 (유죄 판례)"
        analysis_guide = "이 판례는 유죄가 선고된 사례입니다. 승소(유죄) 요인을 중점적으로 분석하세요."
    else:
        # 유죄가 없으면 -> 무죄 중 제일 비슷한 걸 메인으로 (경고 모드!)
        main_case, main_score = results[0]
        remaining = results[1:]
        is_guilty_found = False
        section_title = "⚠️ 유사 판례 (무죄 사례 주의)"
        analysis_guide = """
        🚨 [중요 경고] 검색 결과, 유사한 유죄 판례가 없습니다. 
        이 사례는 '무죄(혐의 없음)' 판결이 난 사례입니다.
        사용자에게 '유사한 승소 사례를 찾지 못했음'을 명확히 알리고, 
        이 사건은 **어떤 이유 때문에 처벌받지 않았는지(패소 요인)**를 분석하여 사용자에게 주의를 주세요.
        """

    # 최종 문서 리스트 (메인 + 나머지 4개)
    final_docs = [main_case]
    final_scores = [main_score]
    for doc, score in remaining[:4]:
        final_docs.append(doc)
        final_scores.append(score)

    # 4. 프롬프트 구성
    context_text = f"""
    [📌 메인 분석 대상 판례]
    - 판결 결과: {main_case.metadata.get('judgment')} (매우 중요!)
    - 사건번호: {main_case.metadata.get('case_id')}
    - 내용: {main_case.page_content}
    - 유사도: {main_score*100:.1f}%

    [📑 기타 참고 판례]
    """
    for i, doc in enumerate(final_docs[1:]):
        context_text += f"{i+1}. {doc.metadata.get('case_id')} ({doc.metadata.get('judgment')}): {doc.page_content[:100]}...\n"

    template = """
    당신은 대한민국 사이버 범죄 전문 AI 변호사 'LawLens'입니다.
    
    [분석 데이터]
    {context}

    [사용자 상황]
    {question}

    **[AI 분석 가이드]**
    1. 현재 분석 모드: **{section_title}**
    2. 지침: {analysis_guide}
    3. 사용자 상황이 '[분석 모드: 기사/커뮤니티 악플]'이면 작성자별로 나누어 분석하세요.

    ---
    [작성 양식]

    ### 1. 📝 AI 사건 정밀 분석
    * **사건 개요:** (전체적인 상황 요약)
    * **핵심 쟁점:** (모욕성, 공연성, 특정성 충족 여부)
    
    **(다중 악플인 경우에만 작성)**
    | 작성자 | 발언 요약 | 요건 충족(모욕/특정/공연) | 처벌 확률 |
    | :--- | :--- | :--- | :--- |
    | (ID) | (내용 짧게) | 모욕(O), 특정(X), 공연(O) | 낮음 |
    | (ID) | (내용 짧게) | 모욕(O), 특정(O), 공연(O) | **매우 높음** |

    **(다중 악플 상세 분석)**
    * **[작성자 ID 1] 상세 검토:**
      - 판단: (왜 처벌 확률이 높은지/낮은지 구체적인 법적 이유 서술)
    * **[작성자 ID 2] 상세 검토:**
      - 판단: (욕설의 수위, 특정성 성립 여부 등 상세 분석)

    ### 2. {section_title}
    
    | 구분 | 내용 |
    | :--- | :--- |
    | **유사도** | **{main_score_str}** |
    | **사건번호** | {main_case_id} |
    | **판결 결과** | **{main_judgment}** |
    | **사실관계** | (판례 내용 요약) |
    | **승소(유죄) 요인** | (이 사건에서 유죄가 인정된 결정적인 이유 1~2가지 / 무죄라면 무죄 이유) |
    | **내 사건과의 공통점** | (사용자의 상황과 이 판례가 유사한 점을 구체적으로 비교) |
    | **법적 판단 근거** | (법원의 법률적 논리 및 적용 법조항) |
    | **내 사건 조언** | (이 판례를 통해 본 내 사건의 유불리 및 대응 전략) |

    ### 3. 📑 기타 유사 판례 요약
    (참고 판례 4건 요약)
    * **사건 A:** (사건명/번호) - (결과) / (핵심 이유)
    * **사건 B:** (사건명/번호) - (결과) / (핵심 이유)
    * **사건 C:** (사건명/번호) - (결과) / (핵심 이유)
    * **사건 D:** (사건명/번호) - (결과) / (핵심 이유)

    ### 4. 📉 예상 처벌 및 승소 확률
    | 구분 | 예측 결과 |
    | :--- | :--- |
    | **승소 확률** | **{main_score_str}** (유사 판례 기반) |
    | **예상 벌금** | (유죄 판례가 없으면 '예측 불가' 또는 '처벌 가능성 낮음'으로 기재) |
    | **처벌 수위** | (예상되는 처분) |

    ### 5. 🏛️ 고소 절차 안내
    (표준 절차 안내)
    * **표준 절차:** 경찰서 접수(고소장) -> 피의자 특정 및 소환 조사 -> 검찰 송치 -> 기소 -> 법원 판결
    * **예상 소요 시간:** (통상적인 사이버 모욕죄 사건 소요 시간, 예: 3~6개월)
    * **준비물:** 신분증, 증거 자료(캡처, 녹음 등), 고소장

    ---
    **작성 지침:** - 만약 '무죄 사례 주의' 모드라면, 승소 확률이 낮을 수 있음을 솔직하게 말해주세요.
    - 표 형식을 반드시 유지하세요.
    """
    
    prompt = PromptTemplate(template=template, input_variables=[
        "context", "question", "main_score_str", "main_case_id", 
        "section_title", "analysis_guide", "main_judgment"
    ])
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1, google_api_key=api_key)
    chain = prompt | llm | StrOutputParser()
    
    final_response = chain.invoke({
        "context": context_text,
        "question": query,
        "main_score_str": f"약 {main_score*100:.1f}%",
        "main_case_id": main_case.metadata.get('case_id', '정보 없음'),
        "main_judgment": main_case.metadata.get('judgment', '미상'),
        "section_title": section_title,
        "analysis_guide": analysis_guide
    })
    
    return {
        "result": final_response,
        "docs": final_docs,
        "scores": final_scores
    }

# (호환성 유지)
def get_lawlens_advisor(): pass
def get_similarity_scores(query, k=5): pass

def generate_complaint_draft(user_story):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return "API Key Error"
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
    prompt = PromptTemplate(template="[사용자 상황]\n{story}\n\n위 내용을 바탕으로 경찰청 표준 고소장 내용을 작성해줘.", input_variables=["story"])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"story": user_story})