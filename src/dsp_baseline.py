# src/dsp_baseline.py
# -*- coding: utf-8 -*-
"""DSPy를 이용한 간단한 구조화 요약 파이프라인(웹 검색/벡터 검색은 외부에서 준비해 컨텍스트로 주입)"""
from typing import Optional, List
from pydantic import BaseModel
import dspy
from dotenv import load_dotenv

class FarmLogFields(BaseModel):
    json: str  # LLM이 JSON 문자열만 출력하도록 강제

class MakeFarmLog(dspy.Signature):
    """STT 텍스트와 보조 컨텍스트(RAG, 웹노트)를 읽고 영농일지 JSON을 만듭니다.
    출력은 반드시 유효한 JSON 문자열이어야 합니다.
    """
    stt: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: FarmLogFields = dspy.OutputField()

class DSPyFarmLog:
    def __init__(self, model_name: str = "gpt-4o"):
        load_dotenv()
        lm = dspy.OpenAI(model=model_name, temperature=0)
        dspy.settings.configure(lm=lm)
        self.predictor = dspy.Predict(MakeFarmLog)

    def run(self, stt: str, context: str) -> str:
        out = self.predictor(stt=stt, context=context)
        return out.answer.json