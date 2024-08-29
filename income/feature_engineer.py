import streamlit as st

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


def feature_engineer(total_df):
        st.title('특성공학(Feature Engineering)')
        st.markdown("""
        - 머신러닝 모델의 성능을 향상시키기 위해 원시 데이터에서 유의미한 특징(feature)을 추출하고 변형하는 과정.
        - 이 과정은 데이터의 도메인 지식과 통계적 기법을 사용하여 데이터를 처리하고, 모델이 학습하기에 적합한 입력 변수를 만들어내는 것을 목표로 한다.""")
        
        st.markdown(""" """)


        st.markdown("""### VIF(Variance Inflation Factor) 진행 """)
        
        st.markdown("""
        - 분산 팽창 인자 VIF는 **다중공선성(multicollinearity)을 판단**하기 위해 사용되는 통계적 지표이다.
        - VIF는 이러한 다중공선성을 판단하기 위해 각 독립 변수가 다른 독립 변수들과 얼마나 상관되어 있는지를 측정한다.
            """)
        
        st.markdown("""
        
        > ❓다중공선성(multicollinearity)란?
            
            - 회귀 분석에서 독립 변수들 간에 강한 상관관계가 존재하는 현상을 뜻함.
            - 이로 인해 회귀 모델에서 각 독립 변수의 개별적인 영향력을 정확하게 측정하기 어려워지며, 모델의 계수 추정이 불안정해지고 해석이 어려워질 수 있음. 
            - 다중공선성이 심하면 모델이 특정 변수에 과도하게 의존하게 되어, 새로운 데이터에 대한 예측 성능이 저하될 수 있음.
        
            """)

        st.subheader('📌VIF 진행 결과', divider='gray')
        st.markdown("""
        - 이 데이터에서 대부분의 변수들은 다중공선성 문제가 심각하지 않으며, VIF 값이 1과 5 사이에 있어 허용이 가능한 수치임을 알 수 있다.
        - VIF 값이 3을 초과하는 몇몇 변수들(Family_num, Master_Retired)은 주의 깊게 다뤄야 한다.
            
            ▶ 다중공선성을 줄이기 위해 이 **변수들을 제외**하거나, **주성분 분석(PCA)** 등 차원 축소 기법을 사용하여 해결하고자 함.""")



        # Split into two columns: VIF Data and VIF Plot
        top_col1, top_col2 = st.columns([3, 5], gap="small", vertical_alignment="center")

        col1, col2 = st.columns([3, 5], gap="small", vertical_alignment="center")
        

        with top_col1:
            st.markdown("#### VIF 분석 결과 순서")

        with top_col2:
            st.markdown("#### VIF 분석 결과 시각화")

        with col1:
            # Prepare data for VIF analysis
            X = total_df.drop('Income_code', axis=1)
            X = add_constant(X)

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data = vif_data.iloc[1:]  # 상수항(constant)을 제거

            # VIF 데이터프레임 출력
            st.write(vif_data)

        with col2:
            # VIF 시각화
            plt.figure(figsize=(4, 4))
            sns.barplot(x='VIF', y='Feature', data=vif_data)
            plt.title('Variance Inflation Factor (VIF)')
            plt.xticks(rotation=90)
            st.pyplot(plt)
