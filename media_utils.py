import easyocr
import whisper
import warnings
import os
import torch
import numpy as np
import streamlit as st
import config

# 경고 메시지 무시
warnings.filterwarnings("ignore")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
# 데코레이션 사용으로 처음 한번만 로드
@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    # EasyOCR 모델 메모리에 로드
    return easyocr.Reader(config.OCR_LANGUAGES, gpu=torch.cuda.is_available(), verbose=False)

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    # Whisper 모델 메모리에 로드
    return whisper.load_model(config.WHISPER_MODEL_SIZE)

@st.cache_resource(show_spinner=False)
def load_pyannote_pipeline(hf_token):
    # Pyannote 화자 분리 파이프라인 로드
    if not PYANNOTE_AVAILABLE or not hf_token:
        return None
    
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        return pipeline
    except Exception as e:
        print(f"Pyannote 로드 실패: {str(e)}")
        return None

def group_text_by_line(results, y_threshold=15):
    """
    EasyOCR 결과(단어 단위)를 Y좌표 기준으로 묶어서 '줄(Line)' 단위로 만듦.
    채팅방은 [닉네임] [내용]이 같은 높이에 있으므로 이를 합쳐야 함.
    """
    if not results:
        return ""

    # Y좌표(세로 위치) 기준으로 정렬
    # item 구조: ([[x,y], ...], 'text', conf)
    sorted_results = sorted(results, key=lambda x: x[0][0][1]) 

    lines = []
    current_line = [sorted_results[0]]

    for i in range(1, len(sorted_results)):
        prev_y = current_line[-1][0][0][1] # 이전 글자 Y좌표
        curr_y = sorted_results[i][0][0][1] # 현재 글자 Y좌표

        # 높이 차이가 크지 않으면 같은 줄로 간주
        if abs(curr_y - prev_y) < y_threshold:
            current_line.append(sorted_results[i])
        else:
            # 줄 바꿈 발생 -> 저장된 줄을 X좌표(가로) 순으로 정렬 후 합침
            current_line.sort(key=lambda x: x[0][0][0])
            line_text = " ".join([item[1] for item in current_line])
            lines.append(line_text)
            current_line = [sorted_results[i]]

    # 마지막 줄 처리
    if current_line:
        current_line.sort(key=lambda x: x[0][0][0])
        lines.append(" ".join([item[1] for item in current_line]))

    return "\n".join(lines)

def extract_text_from_image(image_path):
    """
    [EasyOCR] 이미지 파일 경로를 받아서 텍스트로 변환
    """
    print(f"📷 이미지 분석 중... ({image_path})")
    try:
        # GPU가 없으면 gpu=False로 자동 설정됨
        reader = load_easyocr_reader()
        result = reader.readtext(image_path, detail=1)

        return group_text_by_line(result)
    except Exception as e:
        return f"OCR 에러: {str(e)}"
    
def format_time(seconds):
    # 초 단위를 mm:ss 형식으로 변환
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

def extract_text_from_audio(audio_path, hf_token=None):
    """
    [Whisper + Pyannote] 음성 -> 텍스트 (화자 분리 포함)
    hf_token: Hugging Face 토큰 (pyannote 사용 시 필수)
    """
    hf_token = hf_token or config.HF_TOKEN

    print(f"🎤 음성 분석 중... ({audio_path})")
    
    if not os.path.exists(audio_path):
        return "파일이 없습니다."

    try:
        # 1. 캐싱된 Whisper 모델 로드 및 음성 인식 수행
        model = load_whisper_model()
        
        transcription = model.transcribe(audio_path, language="ko")
        segments = transcription["segments"]

        pipeline = load_pyannote_pipeline(hf_token)

        if pipeline is None:
            if not PYANNOTE_AVAILABLE:
                msg = "Pyannote.audio 라이브러리 미설치"
            elif not hf_token:
                msg = "Hugging Face 토큰 누락"
            else:
                msg = "파이프라인 로드 실패"
            
            print(msg)
            result_text = []
            for seg in segments:
                start = format_time(seg['start'])
                end = format_time(seg['end'])
                result_text.append(f"[{start} - {end}] {seg['text']}")
            return "\n".join(result_text)
        
        # 2. 화자 분리 수행
        print("🗣️ 화자 분리(Diarization) 수행 중...")
        diarization = pipeline(audio_path)
        
        # 3. Whisper 세그먼트와 화자 정보 매칭
        final_output = []
        
        for seg in segments:
            w_start, w_end, w_text = seg['start'], seg['end'], seg['text']

            speaker_counts = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(w_start, turn.start)
                overlap_end = min(w_end, turn.end)
                duration = max(0, overlap_end - overlap_start)
                
                if duration > 0:
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + duration
            
            # 가장 오래 말한 화자 선택 (없으면 Unknown)
            if speaker_counts:
                best_speaker = max(speaker_counts, key=speaker_counts.get)
            else:
                best_speaker = "Unknown"

            # 화자 이름 변경 (SPEAKER_00 -> 화자 A)
            speaker_label = f"화자 {int(best_speaker.split('_')[-1]) + 1}" if "SPEAKER" in best_speaker else best_speaker
            
            time_str = f"[{format_time(w_start)}]"
            final_output.append(f"{time_str} {speaker_label}: {w_text}")

        return "\n".join(final_output)

    except Exception as e:
        return f"음성 분석 에러: {str(e)}"

# 테스트 코드
if __name__ == "__main__":
    # 테스트용
    print("모듈 테스트 준비 완료")