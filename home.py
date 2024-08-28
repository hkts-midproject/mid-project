# -*- coding:utf-8 -*-
import pandas as pd
from utils import load_eda_data
from utils import load_data
import streamlit as st
from millify import prettify
import seaborn as sns
import matplotlib.pyplot as plt

def run_home():
    total_df = load_eda_data()

    st.markdown(">### 🚩 팀소개 \n"
    "#### 팀 명 : \n"
    "#### 구성원 : \n"
    "#### ⭐ 조장 : 박선애 \n"
    "#### ⭐ 조원 : 신지민 / 오손빈 / 이유리 / 정명훈 \n"
    )



    st.markdown(">### 📊 대시보드 개요 \n"
    "##### **본 프로젝트는 데이터를 이용해 소비자 정보에 따른 소득 구간(금액별) 예측 모델링 대시보드입니다.**  \n"
    "##### **고객 인적 정보, 자산, 부채, 소비 데이터를 사용하여 소득분위에 영향을 미치는 요인들을 분석하고, 이를 통해 소득 분위를 예측하고자 함**  \n"
    )

    st.markdown(">### 📚 사용데이터\n"
                "#### [가계금융복지조사(2023년 Data)](https://mdis.kostat.go.kr/ofrData/selectOfrDataDetail.do?survId=1005641&itmDiv=1&nPage=3&itemId=2005&itemNm=%EC%86%8C%EB%93%9D%C2%B7%EC%86%8C%EB%B9%84%C2%B7%EC%9E%90%EC%82%B0)\n"
                "- **사용자 인적 정보 ( 나이, 결혼, 직업 등)**\n"
                "- **자산 (총금액)**\n"
                "- **부채 (총금액)**\n"
                "- **소비 (총금액)**")

    