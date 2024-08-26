import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

def invest_run(): 
    st.header('고객 투자 성향 분석')
    st.markdown("""---""")


    tab1, tab2 = st.tabs(["1", "2"])


    with tab1:
        st.markdown('### 1번 탭 입니다.')


    with tab2:
        st.markdown('### 2번 탭 입니다.')
