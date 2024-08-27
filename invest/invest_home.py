import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_survey as ss
from .survey import survey_display

def invest_run(): 
    
    

    st.header('고객 투자 성향 분석')
    st.markdown("""---""")


    tab1, tab2 = st.tabs(["클러스터링 분석", "고객 투자성향 분석 예측"])


    with tab1:
        st.markdown('### TBD: 클러스터링 분석')


    with tab2:
        survey_display()
