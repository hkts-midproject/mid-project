# -*- coding:utf-8 -*-
import pandas as pd
from utils import load_data
import streamlit as st
from millify import prettify
import seaborn as sns
import matplotlib.pyplot as plt

def run_home():
    total_df = load_data()
    st.markdown("## 대시보드 개요 \n"
    "**본 프로젝트는 데이터를 이용해 소비자 정보에 따른 소득 구간(금액별) 예측 모델링 대시보드입니다.** \n"
    "**고객 인적 정보, 자산, 부채, 소비 데이터를 사용하여 소득분위에 영향을 미치는 요인들을 분석하고,**\n"
    "**이를 통해 소득 분위를 예측하고자 함**")

    st.markdown("### 사용데이터\n"
                "- **사용자 인적 정보 ( 나이, 결혼, 직업 등)**\n"
                "- **자산 (총금액)**\n"
                "- **부채 (총금액)**\n"
                "- **소비 (총금액)**")

    