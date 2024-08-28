# -*- coding:utf-8 -*-
import streamlit as st


from .feature_engineer import feature_engineer
from .distribution import distribution
from .factors import factors
from .modeling import modeling

def income_run(total_df): 
    st.header('Income Decile Prediction')
    st.markdown("""---""")

    # Tab creation for different steps
    tab1, tab2, tab3, tab4 = st.tabs(["특성공학(Feature Engineering)", "데이터 분포도 확인(Data Distribution)", "요인 분석(Factor Analysis)", "모델링(Modeling)"])

    # Feature Engineering
    with tab1:
        feature_engineer(total_df=total_df)

    # PCA 및 t-SNE 분포 분석
    with tab2:
        distribution(total_df)

    # 요인 분석
    with tab3:
        factors(total_df)

    # 모델링
    with tab4:
        modeling(total_df)
    