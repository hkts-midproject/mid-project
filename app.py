# -*- coding:utf-8 -*-

import streamlit as st
from streamlit_option_menu import option_menu
from home import run_home
from utils import load_data
from eda.eda_home import run_eda

def main():
    total_df = load_data()
    with st.sidebar:
        selected = option_menu("MENU", ['HOME', 'EDA', 'ML Model - 소득 분위 예측', 'ML Model - 고객 투자 성향 분석'], 
                               icons=['house', 'file-bar-graph', 'graph-up-arrow', 'graph-up-arrow'], menu_icon="cast", default_index=0)
    if selected == "HOME":
        run_home()  
    elif selected == "EDA":
        run_eda(total_df)
    elif selected == "ML Model - 소득 분위 예측":
        pass
    elif selected == "ML Model - 고객 투자 성향 분석":
        pass
    else:
        print("error..")
        
if __name__ == "__main__":
    main()