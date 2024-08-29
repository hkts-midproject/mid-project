# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# import time


columns_to_plot = ['Assets', 'Liabilities', 'Principal_Interest', 'Spend_Consum', 'Spend_NonConsum']

def outlierChart(total_df, columns_to_plot, num_columns=3):
    st.markdown("#### **5개 컬럼에 대한 이상치 확인** \n"
                "##### - 이 그래프들은 모두 극단적인 비대칭성을 보이며, 데이터가 매우 비대칭적으로 분포되어 있음을 나타냄.\n"
                "##### - 대부분의 변수들이 낮은 값에 집중되어 있고, 극단적으로 높은 값들이 존재하여 왜도와 첨도가 매우 높음.\n"
                "##### - 이러한 분포 특성은 데이터 분석과 모델링 시 이상치 처리가 필요하며, 분석 결과에 크게 영향을 미칠 수 있으므로 주의가 필요함.  \n"
                "***  "
                )
    # 행의 개수 설정
    num_rows = (len(columns_to_plot) + num_columns - 1) // num_columns

    # 전체 그래프 크기 설정
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))

    # axes를 1차원 배열로 변환 (1개일 경우도 처리 가능하게)
    axes = axes.flatten() if num_rows * num_columns > 1 else [axes]

    # 컬럼마다 그래프를 출력
    for i, column_to_plot in enumerate(columns_to_plot):
        ax = axes[i]

        # 기본 통계 요약
        stats_summary = total_df[column_to_plot].describe()

        # 분포의 형태(왜도와 첨도) 계산
        skewness = total_df[column_to_plot].skew()
        kurtosis = total_df[column_to_plot].kurt()

        # 히스토그램과 KDE 시각화
        sns.histplot(total_df[column_to_plot], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {column_to_plot}')
        ax.set_xlabel(column_to_plot)
        ax.set_ylabel('Frequency')

        # 텍스트로 통계 요약, 왜도, 첨도 추가
        ax.text(0.95, 0.85, f'Mean: {stats_summary["mean"]:.2f}\n'
                            f'Median: {stats_summary["50%"]:.2f}\n'
                            f'Standard Deviation: {stats_summary["std"]:.2f}\n'
                            f'Skewness: {skewness:.2f}\n'
                            f'Kurtosis: {kurtosis:.2f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))

    # 남은 빈 공간을 비활성화
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 그래프 간 간격 조정
    plt.tight_layout()

    # Streamlit에서 차트 표시
    st.pyplot(fig)

columns_to_check = ['Assets', 'Liabilities', 'Principal_Interest', 'Spend_Consum', 'Spend_NonConsum']

def show_boxplot(total_df, columns_to_check):
    st.markdown("#### **이상치 제거 결과** \n"
                "##### - 원본 데이터 크기 : (18094, 26)  \n"
                "##### - 이상치 제거 후 데이터 크기 : (11939, 26)  \n"  
                )
    # Boxplot 생성
    plt.figure(figsize=(13, 6))
    sns.boxplot(data=total_df[columns_to_check])
    plt.title('Boxplot of Selected Financial Columns Without Outliers')
    plt.xlabel('Columns')
    plt.ylabel('Value')
    plt.xticks(rotation=45)  # X축 레이블 회전

    # Streamlit에서 차트 표시
    st.pyplot(plt)



def showViz(total_df):
    selected = st.sidebar.radio("차트 메뉴", ['이상치 확인', '이상치 제거'])
    if selected == "이상치 확인":
        outlierChart(total_df, columns_to_plot, num_columns=3)
    elif selected == "이상치 제거":
        show_boxplot(total_df, columns_to_check)
    else:
        st.warning("Error")
