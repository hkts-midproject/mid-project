# -*- coding:utf-8 -*-
from eda.statistics import showViz_2
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from eda.viz import showViz

def home():
    st.markdown("### Preprocessing \n"
    "- **이상치 확인** \n"
    "- **이상치 제거** \n"
    "*** \n"
    )
    
    st.markdown("### Analytics \n"
    "- **비소비지출에 따른 소득금액 확인** \n"
    "- **소비지출에 따른 소득금액 확인** \n"
    "- **은퇴상태에 따른 소득금액 확인** \n"
    "*** \n"
    )
    

def run_eda(total_df):
    st.markdown("## 🧬 탐색적 자료 분석 개요 🧬 \n"
                
                )

    selected = option_menu(None, ["Contents", "Preprocessing", "Analytics"],
                                icons=['map', 'bar-chart', "file-spreadsheet"],
                                menu_icon="cast", default_index=0, orientation="horizontal",
                                styles={
                                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                                    "icon": {"color": "orange", "font-size": "18px"},
                                    "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                                 "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "blue"},
                                }
                            )

    if selected == 'Contents':
        st.title("Contents")
        home()
    elif selected == 'Preprocessing':
        st.title("Preprocessing")
        showViz(total_df)
    elif selected == 'Analytics':
        st.title("Analytics")
        showViz_2(total_df)
    else:
        st.warning("Wrong")