import re
import emoji
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json

load_dotenv()

class LawLensPreprocessor:
    def __init__(self):
        # 분석을 위한 LLM 설정
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # ---------------------------------------------------------
    # 정규화 (Normalization) & 노이즈 제거 (Noise Cleaning)
    # 파이썬 정규식 사용
    # ---------------------------------------------------------
    def clean_text(self, text):
        if not text:
            return ""
        # 1. 날짜/시간 패턴 강화 (다양한 포맷 대응)
        # 예: "2024년 1월 1일 월요일", "---- 2024.01.01 ----", "[오전 10:30]", "14:20:55" 등
        text = re.sub(r'[-=]*\s*\d{4}[년.-]\s*\d{1,2}[월.-]\s*\d{1,2}[일.-]?\s*.*[-=]*', '', text) # 날짜 구분선 제거
        text = re.sub(r'\[?\s*(오전|오후)?\s*\d{1,2}:\d{2}(:\d{2})?\s*\]?', '', text)  # 타임스탬프 통합 제거

        # 2. 시스템 메시지 제거 (줄 단위 처리로 안전성 확보)
        # ".*" 패턴은 자칫 일반 대화까지 지울 수 있으므로, 줄의 시작(^)과 끝($)을 명시하거나 특정 키워드 위주로 삭제
        system_patterns = [
            r'.*님이 입장하셨습니다.*',
            r'.*님이 나갔습니다.*',
            r'.*님이 .*님을 초대했습니다.*',
            r'.*채팅방을 나갔습니다.*'
        ]
        for pattern in system_patterns:
            text = re.sub(pattern, '', text)
        # 3. 개인정보(PII) 마스킹
        # 전화번호 (010-1234-5678, 010 1234 5678) -> [전화번호]
        text = re.sub(r'01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}', '[전화번호]', text)

        # 4. 반복 문자 축약 
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # 5. 이모지 변환 시 띄어쓰기 확보
        # 이모지를 텍스트로 바꿀 때 앞뒤 공백을 넣어 분석기가 단어를 잘 구분하게 함
        text = emoji.demojize(text, language='ko')
        text = text.replace(":", " ") # :smile: -> smile (콜론 제거로 토큰화 용이하게)

        # 6. 다중 공백 및 줄바꿈 정리
        # 타임스탬프 삭제 후 남은 "  " 등을 공백 하나로 통일
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ---------------------------------------------------------
    # 법률적 판단 및 구조화
    # 규칙으로 짜기 어려운 '맥락'은 LLM에게 시킴
    # ---------------------------------------------------------
    def analyze_features(self, cleaned_text):
        prompt = PromptTemplate.from_template("""
        너는 사이버 범죄 전문 법률 분석가야. 아래 텍스트를 분석해서 JSON 형식으로 출력해.
        
        [분석할 텍스트]
        {text}

        [분석 지침]
        1. 대상 특정성 (target_type): 개인(닉네임), 개인(실명/지인), 집단, 불특정 중 선택
        2. 공연성 (space): 1:1대화, 소수단톡방, 다수단톡방, 전체채팅/게시판 중 선택
        3. 표현 유형 (expression): 단순욕설, 인격비하, 성적표현, 패드립, 협박, 사실적시 중 선택 (복수 가능)
        4. 목적성 (sexual_intent): 없음, 분노표출, 성적흥분/만족, 조롱 중 선택 (통매음 판단 핵심)
        5. 범죄 유형 후보 (candidate_crime): 모욕, 통신매체이용음란(통매음), 명예훼손, 협박, 기타 중 선택
        6. 위험도 (risk_level): 높음, 중간, 낮음, 없음
        7. STT/OCR 오타 보정 : "박아" vs "밖에", "보지" vs "보지요" 등 발음이 유사한 오타가 있어도 문맥을 보고 원래 의도를 파악하여 판단하세요.                    

        [출력 형식 - 반드시 JSON만 출력할 것]
        {{
            "features": {{
                "target_type": "...",
                "space": "...",
                "expression": ["...", "..."],
                "sexual_intent": "..."
            }},
            "candidate_crime": "...",
            "risk_level": "...",
            "reason": "간단한 분석 이유 한 줄"
        }}
        """)

        chain = prompt | self.llm
        try:
            response = chain.invoke({"text": cleaned_text})
            # JSON 부분만 깔끔하게 추출
            json_str = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(json_str)
        except Exception as e:
            return {"error": str(e), "candidate_crime": "분석실패"}

    # 전체 파이프라인 실행 함수
    def run_pipeline(self, raw_text):
        # 1단계: 텍스트 정제
        normalized_text = self.clean_text(raw_text)
        
        # 2단계: AI 심층 분석
        analysis_result = self.analyze_features(normalized_text)
        
        # 3단계: 최종 결과 합치기
        final_data = {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "analysis": analysis_result
        }
        return final_data

# 테스트 실행
if __name__ == "__main__":
    processor = LawLensPreprocessor()
    
    # 테스트 케이스
    sample = "[14:20] 김롤붕: 야이 씨%%%%발 개못생긴 년아 ㅋㅋㅋㅋㅋㅋ 니네 엄마한테 가서 젖이나 더 먹고와라 🤬"
    
    result = processor.run_pipeline(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))