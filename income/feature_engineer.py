import streamlit as st

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


def feature_engineer(total_df):
        st.markdown('### 특성공학(Feature Engineering)')


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
