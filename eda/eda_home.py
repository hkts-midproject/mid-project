# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from eda.viz import showViz

def home():
    st.markdown("### Visualization 개요 \n"
    "- **이상치 확인** \n"
    "- **이상치 제거** \n"
    "- **분석 방법** \n")
    st.markdown("### Statistics 개요 \n")
    st.markdown("### Map 개요 \n")

def run_eda(total_df):
    st.markdown("## 탐색적 자료 분석 개요 \n"
                "👇👇👇 탐색적 자료분석 페이지입니다. 👇👇👇"
                )

    selected = option_menu(None, ["Home", "Visualization", "Statistics", "Team 소개"],
                                icons=['house', 'bar-chart', "file-spreadsheet", 'map'],
                                menu_icon="cast", default_index=0, orientation="horizontal",
                                styles={
                                    "container": {"padding": "0!important", "background-color": "#fafafa"}, #fafafa #6F92F7
                                    "icon": {"color": "orange", "font-size": "18px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                                 "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "blue"},
                                }
                            )

    if selected == 'Home':
        home()
    elif selected == 'Visualization':
        # st.title("Visualization")
        showViz(total_df)
    elif selected == 'Statistics':
        st.title("Statistics")
    elif selected == 'Team 소개':
        st.title("Team 소개")
    else:
        st.warning("Wrong")