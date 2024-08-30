# -*- coding:utf-8 -*-
import pandas as pd
from utils import load_eda_data
from utils import load_data
import streamlit as st
from millify import prettify
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import base64

def run_home():
    total_df = load_eda_data()


    # 프로젝트 주제
    st.markdown(">### 📌 프로젝트 주제 \n"
                
                "##### **- 본 프로젝트는 '가계 금융복지조사 2023년' 데이터를 이용해 소비자 정보에 따른 소득 구간(금액별)을 예측**  \n"
                "##### **- '고객 투자성향분석'결과를 기반으로 '재무건강 진단 서비스'를 구현**  \n"
                )
    # 이미지에 링크를 연결하는 코드 (import base64 해주어야됨)
    st.markdown(
            """
            <a href="https://github.com/hkts-midproject/mid-project.git">
                <img src="data:image/png;base64,{}" width="150"></a>
            """
                .format( base64.b64encode(open("data/img/github-logo.png", "rb").read()).decode() ), unsafe_allow_html=True,
            )
    # img = Image.open('data/img/HAN_TOSS_MID_PROJECT.png')
    # url = "https://github.com/hkts-midproject/mid-project.git"
    # st.markdown("[![프로젝트 github](data/img/github-logo.png)](https://github.com/hkts-midproject/mid-project.git)")
    
    # st.image(img, width=800)
    
    st.markdown("""---""")

    # 팀 소개
    st.markdown(">### 🚩 팀소개 \n")
    st.markdown("#### 팀 명 : 잘 풀리는 집 🏡\n"
    "#### 팀 원 : 이유리 / 오손빈 / 정명훈 / 신지민 / 박선애(팀장) \n"
    )
    img = Image.open('data/img/팀원소개.png')
    st.image(img, width=700)

    st.markdown("""---""")


    st.markdown(">### 📚 사용데이터\n"
                "#### [가계금융복지조사(2023년 Data)](https://mdis.kostat.go.kr/ofrData/selectOfrDataDetail.do?survId=1005641&itmDiv=1&nPage=3&itemId=2005&itemNm=%EC%86%8C%EB%93%9D%C2%B7%EC%86%8C%EB%B9%84%C2%B7%EC%9E%90%EC%82%B0)\n"
                "- **사용자 인적 정보 ( 나이, 결혼, 직업 등)**\n"
                "- **자산 (총금액)**  \n"
                "- **부채 (총금액)**  \n"
                "- **소비 (총금액)**  \n"
                "***  \n"
                )

    