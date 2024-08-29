import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_survey as ss
from .survey import survey_display
from .clustering import cluster
from .cluster_types import cluster_types
def invest_run(): 
    
    

    st.header('고객 투자 성향 분석')
    st.markdown("""---""")


    tab1, tab2, tab3= st.tabs(["클러스터링 분석", "클러스터 유형 분석", "고객 투자성향 분석 예측"])


    with tab1:
        cluster()

    with tab2:
        cluster_types()

    with tab3:
        survey_display()
