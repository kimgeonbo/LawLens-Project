# legal_utils.py
import re

# 💡 법률 용어 사전
LEGAL_DICTIONARY = {
    "공연성": "불특정 또는 다수인이 인식할 수 있는 상태 (인터넷 댓글은 기본적으로 충족됨)",
    "특정성": "제3자가 봤을 때 '이 욕이 누구를 향한 것인지' 알 수 있는 상태",
    "모욕성": "사실 적시 없이 경멸적 감정을 표현하여 사회적 평가를 떨어뜨리는 것",
    "비방할 목적": "공익이 아닌, 오로지 상대방을 깎아내리려는 악의적 의도",
    "전파가능성": "한 사람에게 말했어도, 그 사람이 말을 퍼뜨릴 가능성이 있으면 공연성 인정",
    "송치": "경찰이 '죄가 있다'고 보아 사건을 검찰로 넘기는 것",
    "불송치": "경찰이 '죄가 안 된다'고 보아 사건을 자체 종결하는 것",
    "기소": "검사가 법원에 재판을 청구하는 것",
    "불기소": "검사가 재판에 넘기지 않고 사건을 끝내는 처분",
    "기소유예": "죄는 인정되나, 반성 등을 고려해 검사가 한 번 봐주는(재판 X) 처분",
    "약식명령": "재판 없이 서류 심사만으로 벌금형을 내리는 간소화 절차",
    "구약식": "검사가 판사에게 벌금형 약식명령을 내려달라고 요청하는 것",
    "선고유예": "죄가 가벼워 형 선고를 미루고, 2년 뒤 없던 일로 해주는 판결",
    "집행유예": "형을 선고하되, 감옥에 보내는 것을 일정 기간 미뤄주는 판결",
    "친고죄": "피해자가 직접 고소해야만 처벌 가능한 범죄 (모욕죄)",
    "반의사불벌죄": "피해자가 처벌을 원치 않으면 처벌 못 하는 범죄 (명예훼손)",
    "위법성 조각": "죄의 요건은 갖췄으나 정당방위 등 이유로 처벌하지 않는 것",
    "사실적시": "허위가 아닌 진실한 사실을 말함",
    "고소": "피해자가 처벌을 요구하는 것",
    "고발": "제3자가 처벌을 요구하는 것",
    "합의": "가해자가 보상하고 피해자가 처벌불원 의사를 밝히는 계약"
}

def add_legal_tooltips(text):
    if not text: return ""
    sorted_keys = sorted(LEGAL_DICTIONARY.keys(), key=len, reverse=True)
    pattern = re.compile('|'.join(re.escape(key) for key in sorted_keys))

    def replace_func(match):
        term = match.group(0)
        definition = LEGAL_DICTIONARY[term]
        url = f"https://terms.naver.com/search.naver?query={term}"
        return (
            f'<a href="{url}" target="_blank" style="text-decoration: none; color: inherit;">'
            f'<span style="font-weight: bold; border-bottom: 2px dotted #555; cursor: help;" '
            f'title="💡 {term}: {definition} (클릭 시 백과사전 검색)">{term}</span></a>'
        )
    return pattern.sub(replace_func, text)