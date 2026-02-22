import streamlit as st
import os
import pandas as pd
import altair as alt
import warnings
import re 
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
warnings.filterwarnings("ignore")

# 함수 임포트
from legal_utils import add_legal_tooltips
from rag_system import run_lawlens_analysis, get_lawlens_advisor, generate_complaint_draft
from media_utils import extract_text_from_image, extract_text_from_audio
from data_preprocessor import LawLensPreprocessor

# 페이지 설정
st.set_page_config(page_title="LawLens - AI 법률 진단", page_icon="⚖️", layout="wide")


# --------------------------------------------------------------------------
# ⚠️ 음성 파일 법적 효력 안내 팝업
# --------------------------------------------------------------------------
@st.dialog("⚠️ 음성 녹음 파일 법적 효력 안내")
def show_audio_legal_warning():
    st.markdown("""
    통신비밀보호법 제3조 및 대법원 판례에 따르면:
    
    1. 본인이 대화에 참여하고 있는 경우(당사자 간 대화), 상대방의 동의 없는 녹음도 법적 증거 능력이 인정되며 처벌받지 않습니다.
    2. 단, 타인 간의 대화(본인이 없는 자리)를 몰래 녹음하는 것은 불법(도청)이며 증거로 사용할 수 없습니다.
    
    귀하가 업로드하려는 파일은 본인이 참여한 대화의 녹음 파일입니까?
    """)
    
    col1, col2 = st.columns(2)
    if col1.button("네, 확인했습니다 (동의)"):
        st.session_state['audio_consent'] = True
        st.rerun()
    if col2.button("아니요 (취소)"):
        st.session_state['audio_consent'] = False
        st.rerun()

# --------------------------------------------------------------------------
# 메인 로직
# --------------------------------------------------------------------------
st.title("⚖️ LawLens")
st.markdown("#### 멀티모달 데이터 기반의 사이버 모욕죄 성립요건 진단 시스템")
st.info("💡 사이드바에서 증거 자료를 업로드하거나, 분석 모드를 변경할 수 있습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_msg = "안녕하세요! AI 변호사 LawLens입니다. 어떤 상황인지 말씀해 주세요."
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

if "uploader_key" not in st.session_state: st.session_state["uploader_key"] = 0
if "audio_consent" not in st.session_state: st.session_state["audio_consent"] = False

# ==============================================================================
# 📂 사이드바 (모드 선택 및 파일 업로드)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 설정 및 증거")
    
    analysis_mode = st.radio(
        "분석 모드 선택",
        ["💬 일반 채팅/게임 (General)", "📰 기사/커뮤니티 악플 (Comments)"],
        captions=["1:1 대화, 롤 채팅 등", "여러 명의 댓글 분석"]
    )
    
    st.markdown("---")
    st.subheader("📂 증거 파일 업로드")
    
    uploader_key = st.session_state["uploader_key"]
    
    uploaded_imgs = st.file_uploader(
        "이미지 (스크린샷)", 
        type=["png", "jpg", "jpeg"], accept_multiple_files=True, 
        key=f"img_uploader_{uploader_key}" 
    )
    
    st.subheader("음성 파일")
    if not st.session_state['audio_consent']:
        if st.button("🎤 녹음 파일 업로드 (클릭)"):
            show_audio_legal_warning()
    else:
        st.success("✅ 법적 고지 동의 완료")
        uploaded_audios = st.file_uploader(
            "녹음/영상 파일 (mp3, wav 등)", 
            type=["mp3", "wav", "m4a", "mp4", "avi", "mov"], accept_multiple_files=True, 
            key=f"audio_uploader_{uploader_key}"
        )
        if st.button("동의 취소 (업로더 숨기기)"):
            st.session_state['audio_consent'] = False
            st.rerun()
    
    if 'uploaded_audios' not in locals(): uploaded_audios = None
    
    st.markdown("---")
    if uploaded_imgs: st.success(f"📷 이미지 {len(uploaded_imgs)}장 준비됨")
    if uploaded_audios: st.success(f"🎤 음성파일 {len(uploaded_audios)}개 준비됨")

# ==============================================================================
# 💬 채팅 및 결과 표시 화면
# ==============================================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True) 

        # 그래프 및 대시보드
        if "df" in message and message["df"] is not None:
            df = message["df"]
            st.markdown("---")
            st.subheader("📊 유사 판례 분석 대시보드")
            
            if "scores" in message and message["scores"]:
                scores = message["scores"]
                avg_score = sum(scores) / len(scores)
                st.metric("전체 판례 평균 유사도", f"{avg_score*100:.1f}%")
            else:
                 st.metric("전체 판례 평균 유사도", "0.0%")

            if not df.empty:
                # -------------------------------------------------------
                # [추가] 판결 내용을 '유죄/무죄/기타'로 그룹화하는 로직
                # -------------------------------------------------------
                def categorize_judgment(text):
                    text = str(text)
                    if any(x in text for x in ['유죄', '벌금', '징역', '선고유예', '집행유예']):
                        return '유죄'
                    elif any(x in text for x in ['무죄', '기각', '공소기각', '혐의없음']):
                        return '무죄'
                    else:
                        return '기타'
                
                # 데이터프레임에 '판결_구분' 컬럼 추가
                df['판결_구분'] = df['판결'].apply(categorize_judgment)
                # -------------------------------------------------------

                col1, col2 = st.columns([1.5, 1])
                
                with col1:
                    st.markdown("##### 📏 판례별 유사도 비교")
                    sim_chart = alt.Chart(df).mark_bar(color='#ff9f43', cornerRadius=5).encode(
                        x=alt.X('사건번호:N', sort=None, axis=alt.Axis(labelAngle=-45), title='사건 번호'),
                        y=alt.Y('유사도(%):Q', scale=alt.Scale(domain=[0, 100]), title='유사도(%)'),
                        tooltip=[
                            alt.Tooltip('판례명:N', title='판례명'),
                            alt.Tooltip('유사도(%):Q', title='유사도(%)', format='.1f')
                        ]
                    ).properties(height=250)
                    st.altair_chart(sim_chart, theme="streamlit")

                with col2:
                    st.markdown("##### ⚖️ 판결 결과 비율")
                    # [수정] 도넛 차트 (innerRadius=60) & 색상 고정
                    pie_chart = alt.Chart(df).mark_arc(innerRadius=60).encode(
                        theta=alt.Theta(field="판결_구분", aggregate="count", type='quantitative'),
                        # 색상 매핑: 유죄=빨강, 무죄=파랑, 기타=회색
                        color=alt.Color('판결_구분:N', 
                                        scale=alt.Scale(domain=['유죄', '무죄', '기타'], 
                                                        range=['#e74c3c', '#3498db', "#555455"]),
                                        legend=alt.Legend(title="판결 구분")),
                        tooltip=[
                            alt.Tooltip('판결_구분:N', title='구분'),
                            alt.Tooltip('count():Q', title='건수'),
                            alt.Tooltip('판결:N', title='상세 내용') # 마우스 올리면 원래 판결 내용도 보임
                        ]
                    ).properties(height=250)
                    st.altair_chart(pie_chart, theme="streamlit")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("##### 💰 벌금 액수 비교")
                    bar_chart = alt.Chart(df).mark_bar(cornerRadius=5).encode(
                        x=alt.X('사건번호:N', axis=alt.Axis(labelAngle=-45), title='사건 번호'), 
                        y=alt.Y('벌금(만원):Q', title='벌금(만원)'),
                        color=alt.Color('판결:N'),
                        tooltip=[
                            alt.Tooltip('사건번호:N', title='사건번호'),
                            alt.Tooltip('벌금(만원):Q', title='벌금')
                        ]
                    ).properties(height=200)
                    st.altair_chart(bar_chart, theme="streamlit")

                with col4:
                    st.markdown("##### 📈 연도별 추이")
                    line_chart = alt.Chart(df).mark_line(point=True).encode(
                        x=alt.X('연도:O', title='연도'), 
                        y=alt.Y('벌금(만원):Q', title='벌금(만원)'),
                        color=alt.value("#8b5c49"),
                        tooltip=[
                            alt.Tooltip('연도:O', title='연도'),
                            alt.Tooltip('벌금(만원):Q', title='벌금')
                        ]
                    ).properties(height=200)
                    st.altair_chart(line_chart, theme="streamlit")

                st.markdown("##### 🔎 상세 판례 데이터 (원본 보기)")
                #  use_container_width=True를 width="stretch"로 변경
                st.dataframe(
                    df,
                    column_config={
                        "판례명": st.column_config.TextColumn("판례 제목", width="medium"),
                        "사건번호": st.column_config.TextColumn("사건 번호"),
                        "유사도(%)": st.column_config.ProgressColumn("유사도", format="%.1f%%", min_value=0, max_value=100),
                        "벌금(만원)": st.column_config.NumberColumn("벌금", format="%d 만원"),
                        "판결": st.column_config.TextColumn("결과"),
                        "링크": st.column_config.LinkColumn("판례 원본", display_text="전문 보기 🔗")
                    },
                    hide_index=True,
                    width="stretch" 
                )

        if "complaint" in message:
            with st.expander("📄 생성된 고소장 초안 (클릭하여 펼치기)", expanded=False):
                st.info("💡 아래는 AI가 작성한 초안입니다. 복사해서 공식 양식에 채워 넣으세요.")
                official_url = "https://minwon.police.go.kr/app/common/file/FrontDownloadCustomerCenter?path=/contents/datafiles/investigate/investigate1.hwp&fileName=%EA%B3%A0%EC%86%8C%EC%9E%A5.hwp"
                st.markdown(f"**👉 [경찰청 표준 고소장 양식 다운로드 (HWP)]({official_url})**")
                st.text_area("고소장 내용 (초안)", message["complaint"], height=300)
                st.download_button("💾 초안 텍스트 다운로드 (.txt)", message["complaint"], "고소장_초안.txt")

# ==============================================================================
# 🚀 입력 및 질문 처리 로직
# ==============================================================================
final_query = ""
user_input_trigger = False

# 1. 기사/커뮤니티 악플 모드
if analysis_mode == "📰 기사/커뮤니티 악플 (Comments)":
    with st.container(border=True):
        st.subheader("📰 다중 악플/게시글 진단")
        col_news1, col_news2 = st.columns(2)
        with col_news1:
            post_title = st.text_input("게시글/기사 제목", placeholder="예: OO갤러리 개념글")
        with col_news2:
            victim_info = st.text_input("피해 대상 (누구 욕?)", placeholder="예: 작성자(본인), 혹은 댓글 단 다른 유저")
        
        # 안내 문구 강화
        comment_content = st.text_area(
            "악플 내용 (여러 명일 경우 줄바꿈으로 구분)", 
            placeholder="📸 스크린샷이 있다면 이 칸은 비워두셔도 됩니다.", 
            height=150
        )
        
        # 증거 파일 감지 및 안내 문구
        has_file = bool(uploaded_imgs or uploaded_audios)
        if has_file:
            st.success("✅ 증거 파일(스크린샷/녹음)이 준비되었습니다. 텍스트를 입력하지 않고 바로 '진단하기'를 누르셔도 됩니다.")

        # 버튼은 width parameter를 지원하지 않을 수 있으나 에러 로그엔 없었음.
        # 안전하게 하기 위해 버튼은 use_container_width=True 유지 (보통 버튼은 지원함)
        # 만약 버튼도 에러나면 이 부분도 지워야 함.
        if st.button("🚨 다중 악플 진단하기", use_container_width=True): 
            has_text = bool(comment_content.strip())
            
            if not has_text and not has_file:
                st.warning("⚠️ 악플 내용을 입력하거나 증거 파일(스크린샷)을 업로드해주세요.")
            else:
                content_to_show = comment_content if has_text else "(사용자가 업로드한 스크린샷/녹음 파일 내용을 참조하여 분석하세요)"
                
                final_query = f"""
                [분석 모드: 기사/커뮤니티 악플]
                1. 게시글 제목: {post_title}
                2. 피해 대상: {victim_info}
                3. 악플 내용들 (작성자별 분석 필요):
                {content_to_show}
                """
                user_input_trigger = True

# 2. 일반 채팅 모드
else:
    if chat_input := st.chat_input("상황을 입력하세요..."):
        final_query = chat_input
        user_input_trigger = True

# ------------------------------------------------------------------------------
# 🧠 공통 분석 로직
# ------------------------------------------------------------------------------
if user_input_trigger and final_query:
    processed_files_text = ""
    display_msg = ""
    
    with st.spinner("⏳ 증거 파일(이미지/녹음) 분석 및 텍스트 추출 중..."):

        if uploaded_imgs:
            all_extracted_text = ""
            for idx, img_file in enumerate(uploaded_imgs):
                file_ext = os.path.splitext(img_file.name)[1]
                safe_filename = f"temp_img_{idx}{file_ext}"   
                with open(safe_filename, "wb") as f: f.write(img_file.getbuffer())
                extracted = extract_text_from_image(safe_filename)
                if extracted: all_extracted_text += f"\n[이미지 {idx+1}]\n{extracted}\n"
                if os.path.exists(safe_filename): os.remove(safe_filename)
            if all_extracted_text: processed_files_text += f"\n\n[이미지 내용]\n{all_extracted_text}"

        if uploaded_audios:
            all_audio_text = ""
            for idx, audio_file in enumerate(uploaded_audios):
                file_ext = os.path.splitext(audio_file.name)[1]
                safe_filename = f"temp_audio_{idx}{file_ext}"
                with open(safe_filename, "wb") as f: f.write(audio_file.getbuffer())
                extracted = extract_text_from_audio(safe_filename, hf_token=HF_TOKEN)
                if "❌" not in extracted: all_audio_text += f"\n[음성 {idx+1}]\n{extracted}\n"
                if os.path.exists(safe_filename): os.remove(safe_filename)
            if all_audio_text: processed_files_text += f"\n\n[음성 내용]\n{all_audio_text}"

        full_query = final_query + processed_files_text
        
        display_msg = full_query
        if analysis_mode == "📰 기사/커뮤니티 악플 (Comments)":
            preview = comment_content[:50] + "..." if len(comment_content) > 50 else comment_content
            if not preview and has_file: preview = "(스크린샷/녹음 파일 분석 요청)"
            display_msg = f"**[다중 악플 진단 요청]**\n- 제목: {post_title}\n- 대상: {victim_info}\n- 내용:\n{preview}"
            
        st.session_state.messages.append({"role": "user", "content": display_msg})
        with st.chat_message("user"):
            st.markdown(display_msg)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ 판례 검색 및 법률 분석 중... (유죄 판례 우선 검색)"):
                advisor = get_lawlens_advisor() # (안 쓰지만 임포트 때문에 남김)
                processor = LawLensPreprocessor()
                pre_result = processor.run_pipeline(full_query)
                
                analysis = pre_result["analysis"]
                candidate = analysis.get("candidate_crime", "기타")
                search_query = f"{pre_result['normalized_text']}\n키워드: {candidate}"
                
                # 여기서 run_lawlens_analysis 호출
                retrieval_result = run_lawlens_analysis(search_query)
                
                result_text = retrieval_result["result"]
                final_docs = retrieval_result["docs"]
                final_scores = retrieval_result["scores"]
                
                final_display_text = add_legal_tooltips(result_text)

                data_list = []
                for i, doc in enumerate(final_docs):
                    meta = doc.metadata
                    score = final_scores[i] if i < len(final_scores) else 0
                    data_list.append({
                        "판례명": meta.get("title", "?"), "사건번호": meta.get("case_id", "?"),
                        "벌금(만원)": meta.get("fine", 0), "연도": meta.get("year", 2020),
                        "판결": meta.get("judgment", "기타"), "유사도(%)": score * 100,
                        "링크": f"https://www.law.go.kr/precSc.do?menuId=7&query={meta.get('case_id','')}"
                    })
                df = pd.DataFrame(data_list)

                st.markdown(final_display_text, unsafe_allow_html=True)
                
                complaint_text = ""
                with st.spinner("📄 경찰서 제출용 고소장 초안 작성 중..."):
                    complaint_text = generate_complaint_draft(full_query)
                
                if not df.empty:
                    st.markdown("---")
                    st.subheader("📊 대시보드")
                    
                    avg_score = sum(final_scores) / len(final_scores) if final_scores else 0
                    st.metric("전체 판례 평균 유사도", f"{avg_score*100:.1f}%")

                    col1, col2 = st.columns([1.5, 1])
                    with col1:
                        st.markdown("##### 📏 판례별 유사도 비교")
                        sim_chart = alt.Chart(df).mark_bar(color='#ff9f43', cornerRadius=5).encode(
                            x=alt.X('사건번호:N', sort=None, axis=alt.Axis(labelAngle=-45), title='사건 번호'),
                            y=alt.Y('유사도(%):Q', scale=alt.Scale(domain=[0, 100]), title='유사도(%)'),
                            tooltip=[
                                alt.Tooltip('판례명:N', title='판례명'),
                                alt.Tooltip('유사도(%):Q', title='유사도(%)', format='.1f')
                            ]
                        ).properties(height=250)
                        st.altair_chart(sim_chart, theme="streamlit")
                    with col2:
                        st.markdown("##### ⚖️ 판결 결과 비율")
                        pie_chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta(field="판결", aggregate="count", type='quantitative'),
                            color=alt.Color('판결:N', scale=alt.Scale(domain=['유죄', '무죄'], range=['#d9534f', '#5bc0de'])),
                            tooltip=[
                                alt.Tooltip('판결:N', title='결과'),
                                alt.Tooltip('count():Q', title='건수')
                            ]
                        ).properties(height=250)
                        st.altair_chart(pie_chart, theme="streamlit")
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown("##### 💰 벌금 액수 비교")
                        bar_chart = alt.Chart(df).mark_bar(cornerRadius=5).encode(
                            x=alt.X('사건번호:N', axis=alt.Axis(labelAngle=-45), title='사건 번호'), 
                            y=alt.Y('벌금(만원):Q', title='벌금(만원)'),
                            color=alt.Color('판결:N'),
                            tooltip=[
                                alt.Tooltip('사건번호:N', title='사건번호'),
                                alt.Tooltip('벌금(만원):Q', title='벌금')
                            ]
                        ).properties(height=200)
                        st.altair_chart(bar_chart, theme="streamlit")
                    with col4:
                        st.markdown("##### 📈 연도별 추이")
                        line_chart = alt.Chart(df).mark_line(point=True).encode(
                            x=alt.X('연도:O', title='연도'), 
                            y=alt.Y('벌금(만원):Q', title='벌금(만원)'),
                            color=alt.value("#8b5c49"),
                            tooltip=[
                                alt.Tooltip('연도:O', title='연도'),
                                alt.Tooltip('벌금(만원):Q', title='벌금')
                            ]
                        ).properties(height=200)
                        st.altair_chart(line_chart, theme="streamlit")

                    st.markdown("##### 🔎 상세 판례 데이터 (원본 보기)")
                    st.dataframe(
                        df,
                        column_config={
                            "판례명": st.column_config.TextColumn("판례 제목", width="medium"),
                            "사건번호": st.column_config.TextColumn("사건 번호"),
                            "유사도(%)": st.column_config.ProgressColumn("유사도", format="%.1f%%", min_value=0, max_value=100),
                            "벌금(만원)": st.column_config.NumberColumn("벌금", format="%d 만원"),
                            "판결": st.column_config.TextColumn("결과"),
                            "링크": st.column_config.LinkColumn("판례 원본", display_text="전문 보기 🔗")
                        },
                        hide_index=True,
                        width="stretch" # 데이터프레임 width 수정 완료
                    )
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_display_text,
                    "df": df,
                    "scores": final_scores,
                    "complaint": complaint_text
                })
                
                st.session_state["uploader_key"] += 1
                st.rerun()