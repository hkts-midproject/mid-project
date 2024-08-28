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
    plt.figure(figsize=(15,8))
    sns.barplot(x = X_train.columns, y = select.scores_, palette="Set2")
    plt.title('Checking Important Features for ML Modeling ')
    plt.xlabel('Data Columns')
    plt.xticks(rotation=70)

    # Streamlit에서 차트 표시
    st.pyplot(plt)

def show_scatterplot_NonCon(total_df, target_col):

    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]
    


    # Scartter Plot 생성
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=total_df, x=total_df['Spend_NonConsum'], y=total_df[target_col])
    plt.title('Amount of Income from Non-Consumption Data')
    plt.xlabel('Spend_NonConsum')
    plt.ylabel('Income Amount')

    # Streamlit에서 차트 표시
    st.pyplot(plt)

def show_scatterplot_Con(total_df, target_col):

    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]

    # Scartter Plot 생성
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=total_df, x=total_df['Spend_Consum'], y=total_df[target_col])
    plt.title('Amount of Income from Consumption Data')
    plt.xlabel('Spend_Consum')
    plt.ylabel('Income Amount')

    # Streamlit에서 차트 표시
    st.pyplot(plt)

def show_boxplot_retired(total_df, target_col):

    # Filter outliers beyond the 99th percentile
    upper = total_df[target_col].quantile(0.99)
    total_df = total_df[total_df[target_col] < upper]

    # Scartter Plot 생성
    plt.figure(figsize=(15,8))
    sns.boxplot(data=total_df, x=total_df['Master_Retired'], y=total_df[target_col], palette="Set2")
    plt.title('Amount of Income from Master_Retired Data')
    plt.xlabel('Retired')
    plt.ylabel('Income Amount')
    plt.xticks(rotation=45)  # X축 레이블 회전

    # Change X-axis labels to '은퇴' and '은퇴 아님'
    plt.xticks(ticks=[0, 1], labels=['Non-Retired', 'Retired'])

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