# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import time


columns_to_plot = ['Assets', 'Liabilities', 'Principal_Interest', 'Spend_Consum', 'Spend_NonConsum']

def outlierChart(total_df, columns_to_plot, num_columns=3):
    st.markdown("**5개 컬럼에 대한 이상치 확인** \n"
                "- 이 그래프들은 모두 극단적인 비대칭성을 보이며, 데이터가 매우 비대칭적으로 분포되어 있음을 나타냄.\n"
                "- 대부분의 변수들이 낮은 값에 집중되어 있고, 극단적으로 높은 값들이 존재하여 왜도와 첨도가 매우 높음.\n"
                "- 이러한 분포 특성은 데이터 분석과 모델링 시 이상치 처리가 필요하며, 분석 결과에 크게 영향을 미칠 수 있으므로 주의가 필요함."
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
    st.markdown("**이상치 제거 결과** \n"
                "- 원본 데이터 크기 : (18094, 26)\n"
                "- 이상치 제거 후 데이터 크기 : (11939, 26)")
    # Boxplot 생성
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=total_df[columns_to_check])
    plt.title('Boxplot of Selected Financial Columns Without Outliers')
    plt.xlabel('Columns')
    plt.ylabel('Value')
    plt.xticks(rotation=45)  # X축 레이블 회전

    # Streamlit에서 차트 표시
    st.pyplot(plt)


def barChart(total_df):
    st.markdown("### **분석 방법**\n"
                ">#### **다중공선성 확인**\n"
                "- 이 데이터에서 대부분의 변수들은 다중공선성 문제가 심각하지 않으며, VIF 값이 1과 5 사이에 있어 허용 가능한 수치이다.\n"
                "- VIF 값이 3을 초과하는 몇몇 변수들(Family_num, Master_Retired)은 주의 깊게 다뤄야 한다.\n"
                " 다중공선성을 줄이기 위해 이 변수들을 제외하거나, 주성분 분석(PCA) 등 차원 축소 기법을 사용할 수 있음\n"
                ">#### **데이터 분포도 확인**\n"
                "- 예측하고자 하는 소득분위(1~10분위)를 기준으로 데이터의 분포도를 확인.\n"
                "- PCA는 데이터의 전반적인 분포를 이해하고, 주성분이 데이터의 변동성을 어떻게 설명하는지 확인하는 데 유용하고\n"
                "- t-SNE는 데이터의 클러스터링이나 국소적인 데이터 구조를 명확하게 드러내는 데 적합하기 때문에 두 가지의 분포도를 모두 확인\n"
                )
    

    
    # 다중공선성 분석을 위한 데이터프레임 설정 (타겟 변수를 제외한 독립 변수들만 사용)
    X = total_df.drop('Income_code', axis=1)

    # 상수항 추가 (VIF 계산을 위해 필요)
    X = add_constant(X)

    # VIF 계산
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vf = vif_data['Feature'].drop(0)
    vv = vif_data["VIF"].drop(0)
    vif_data = pd.concat([vf, vv], axis=1)


    plt.figure(figsize=(10, 6))
    sns.barplot(x='Feature', y='VIF', data=vif_data)
    plt.title('Variance Inflation Factor (VIF)')
    plt.xticks(rotation=90)

    # Streamlit에서 차트 표시
    st.pyplot(plt)

    st.markdown(">#### **Feature Scaling, PCA 진행** \n "
                "- 데이터 값의 범위를 조정하기 위해 Scaling 진행 \n"
                "   - `StandardScaler` : 데이터의 평균 = 0, 분산 = 1이 되도록 스케일링\n"
                
                "- 다중공선성 최소화, 다수의 Feature 문제를 해결하기 위해 PCA 진행\n "
                "   -" "`pca = PCA(n_components=2)` : 데이터를 2차원 공간으로 축소 \n"
                "   - 이를 통해 데이터의 두 가지 주요 방향(주성분)으로 투영된 데이터를 얻음 \n"
    )

    st.markdown(">#### **데이터 분포도 확인** \n "
                "- 예측하고자 하는 소득분위(1~10분위)를 기준으로 데이터의 분포도를 확인 \n"
                "- PCA는 데이터의 전반적인 분포를 이해하고, 주성분이 데이터의 변동성을 어떻게 설명하는지 확인하는 데 유용하고 데이터들 사이의 차이점이 명확해진다.\n"
                "- t-SNE는 데이터의 클러스터링이나 국소적인 데이터 구조를 명확하게 드러내는 데 적합하기 때문에 두 가지의 분포도를 모두 확인 \n"
                " ->  소득 분위 결정 요인을 분석하기 위해 많은 하이퍼 파라미터 튜닝, 모델링 최적화 등을 해 본 결과 정확도가 좋지 않은 원인을 찾기 위해 시도\n"
    )

    st.markdown("### 📌 **결론**\n" 
                ">***사용하고자 하는 데이터는 그 값들의 경계가 모호하기 때문에 소득분위를 예측하고자 할 때,*** \n"
                 ">***정확하게 각 분위별로 구분을 할 수는 없지만 실제 소득분위와 유사하게 예측할 수 있다는 한계점이 발생했다.*** \n"
                )

def showViz(total_df):
    # total_df['DEAL_YMD'] = pd.to_datetime(total_df['DEAL_YMD'], format="%Y-%m-%d")
    # sgg_nm = st.sidebar.selectbox("자치구명", sorted(total_df['SGG_NM'].unique()))
    selected = st.sidebar.radio("차트 메뉴", ['이상치 확인', '이상치 제거', '분석방법'])
    if selected == "이상치 확인":
        outlierChart(total_df, columns_to_plot, num_columns=3)
    elif selected == "이상치 제거":
        show_boxplot(total_df, columns_to_check)
    elif selected == "분석방법":
        barChart(total_df)
    else:
        st.warning("Error")
