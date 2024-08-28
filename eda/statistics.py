# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant



target_col = 'Income'

def show_import_features(total_df, target_col):

    st.markdown("#### **타겟 변수와 다른 피처 간의 관계를 탐색**  \n"
                "##### - **`Income`** 이라는 타겟 변수와 다른 피처 간의 관계를 탐색하고, 중요한 피처를 식별하며, 이를 시각적으로 표현  \n"
                "##### - 아래의 Bar 차트를 보면 대부분의 피처들이 비슷하게 낮은 값을 가지며, **`Spend_Consum`** 와 **`Spend_NonConsum`** 가 유난히 높은 값을 나타냄  \n"
                "##### - 해당 피처는 모델링 과정에서 우선적으로 고려해야 할 변수입니다.  \n"
                "***  "
                )
    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]

    # 데이터셋에서 상위 50%의 중요한 피처(feature)를 선택

    from sklearn.feature_selection import SelectPercentile

    X_train = total_df.drop([target_col,'Income_code'], axis=1)
    y_train = total_df[target_col]

    select = SelectPercentile(percentile=50) # 50%의 데이터만 사용. 최고의 상황 원래 데이터 30 + 노이즈 10개
    X_train_selected = select.fit_transform(X_train, y_train)

    X_train.shape, X_train_selected.shape

    cols = zip(X_train.columns, select.get_support())

    index = 0
    for z in cols:
        if z[1]:
            print(z[0], X_train_selected[0][index])
            index= index+1
        


    # Scartter Plot 생성
    plt.figure(figsize=(13,6))
    sns.barplot(x = X_train.columns, y = select.scores_, palette="Set2")
    plt.title('Checking Important Features for ML Modeling ')
    plt.xlabel('Data Columns')
    plt.xticks(rotation=70)

    # Streamlit에서 차트 표시
    st.pyplot(plt)

def show_scatterplot_Con(total_df, target_col):

    st.markdown("#### **Spend_Consum(소비지출) 산점도 시각화**  \n"
                "##### - **`Spend_Consum(소비지출)`** 과 **`Income`** 간의 관계를 산점도로 시각화  \n"
                "##### - 소비지출이 소득에 미치는 영향을 더 잘 이해할 수 있음  \n"
                "##### - 소비지출이 높은 경우 소득이 어떻게 변동하는지, 두 변수 간에 상관관계가 존재하는지 확인  \n"
                "***  "
                )

    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]

    # Scartter Plot 생성
    plt.figure(figsize=(13,6))
    sns.scatterplot(data=total_df, x=total_df['Spend_Consum'], y=total_df[target_col])
    plt.title('Amount of Income from Consumption Data')
    plt.xlabel('Spend_Consum')
    plt.ylabel('Income Amount')

    # Streamlit에서 차트 표시
    st.pyplot(plt)

def show_scatterplot_NonCon(total_df, target_col):

    st.markdown("#### **Spend_NonConsum(비소비지출) 산점도 시각화**  \n"
                "##### - **`Spend_NonConsum(비소비지출)`** 과 **`Income`** 간의 관계를 산점도로 시각화  \n"
                "##### - 비소비지출이 소득과 양의 상관관계 또는 음의 상관관계가 있는지 또는 특이점(outliers)이 있는지 확인  \n"
                "##### - 비소비지출이 소득에 어떻게 기여하는지에 대한 가정을 검증할 수 있습니다.  \n  "
                "***  "
                )
    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]
    


    # Scartter Plot 생성
    plt.figure(figsize=(13,6))
    sns.scatterplot(data=total_df, x=total_df['Spend_NonConsum'], y=total_df[target_col])
    plt.title('Amount of Income from Non-Consumption Data')
    plt.xlabel('Spend_NonConsum')
    plt.ylabel('Income Amount')

    # Streamlit에서 차트 표시
    st.pyplot(plt)



def showViz_2(total_df):
    selected = st.sidebar.radio("차트 메뉴", ['중요도 높은 피쳐확인', '소비 지출에 따른 소득금액','비소비 지출에 따른 소득금액'])
    if selected == "중요도 높은 피쳐확인":
        show_import_features(total_df, target_col)

    elif selected == "소비 지출에 따른 소득금액":     
        show_scatterplot_Con(total_df, target_col)

    elif selected == "비소비 지출에 따른 소득금액":
        show_scatterplot_NonCon(total_df, target_col)

        
    else:
        st.warning("Error")