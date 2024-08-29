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
    "#### 팀 명 : 잘 풀리는 집 🏡\n"
    "#### 구성원 : \n"
    "#### ⭐ 조장 - 박선애 \n"
    "#### ⭐ 조원 - 신지민 / 오손빈 / 이유리 / 정명훈 \n"
    "***  "
    )



    st.markdown(">### 📊 대시보드 및 웹앱 개요 \n"
                
                "##### **본 프로젝트는 '가계 금융복지조사 2023년' 데이터를 이용해 소비자 정보에 따른 소득 구간(금액별)을 모델링을 통하여 예측하고**  \n"
                "##### **'고객 투자성향분석'을 통하여 '재무건강진단 웹앱'을 Streamlit으로 구현해보았습니다.**  \n"
                "***  \n"
                )

    st.markdown(">### 📚 사용데이터\n"
                "#### [가계금융복지조사(2023년 Data)](https://mdis.kostat.go.kr/ofrData/selectOfrDataDetail.do?survId=1005641&itmDiv=1&nPage=3&itemId=2005&itemNm=%EC%86%8C%EB%93%9D%C2%B7%EC%86%8C%EB%B9%84%C2%B7%EC%9E%90%EC%82%B0)\n"
                "- **사용자 인적 정보 ( 나이, 결혼, 직업 등)**\n"
                "- **자산 (총금액)**  \n"
                "- **부채 (총금액)**  \n"
                "- **소비 (총금액)**  \n"
                "***  \n"
                )

    