import pytesseract
import cv2
import numpy as np
import os
import whisper
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    return whisper.load_model("tiny")

def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='kor+eng')
        return text.strip() if text.strip() else "(텍스트 인식 실패)"
    except Exception as e:
        return f"OCR 에러: {str(e)}"

def extract_text_from_audio(audio_path, hf_token=None):
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"음성 분석 에러: {str(e)}"