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
    selected = st.sidebar.radio("차트 메뉴", ['비소비 지출에 따른 소득분위', '소비 지출에 따른 소득금액','은퇴상태에 따른 소득금액'])
    if selected == "비소비 지출에 따른 소득분위":
        show_scatterplot_NonCon(total_df, target_col)
    elif selected == "소비 지출에 따른 소득금액":     
        show_scatterplot_Con(total_df, target_col)
    elif selected == "은퇴상태에 따른 소득금액":
        show_boxplot_retired(total_df, target_col)
        
    else:
        st.warning("Error")