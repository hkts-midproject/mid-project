import streamlit as st 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def distribution(total_df):
    st.title('데이터 분포도 확인')

    st.markdown("""- **예측하고자 하는 소득분위(1~10분위)를 기준으로 데이터의 분포도를 확인.**""")

    st.markdown("""
        
        > 1.**PCA**를 통해 데이터의 전반적인 분포를 이해하고, 주성분이 데이터의 변동성을 어떻게 설명하는지 확인.
            
            ❓ PCA(Principal Component Analysis)란?
            - 데이터의 주요 정보를 유지하면서 차원을 축소하기 때문에 데이터의 구조를 단순화하고 노이즈를 제거하는 데 유용하다.
            - 선형 변환을 사용하여 고차원 데이터를 저차원(주로 2D 또는 3D)으로 표현한다.
        
        > 2. **t-SNE**를 통해 데이터의 클러스터링, 국소적인 데이터 구조를 명확하게 드러내어 두 가지의 분포도를 모두 확인.

            ❓ t-SNE(t-Distributed Stochastic Neighbor Embedding)란?
            - 고차원 데이터의 복잡한 구조를 저차원(주로 2D 또는 3D)으로 시각화하기 위해 개발된 비선형 차원 축소 기법이다.
            - 데이터 포인트 간의 유사성을 보존하려고 하며, 특히 데이터의 클러스터(군집) 구조를 잘 시각화할 수 있다.
            - PCA와 달리 t-SNE는 비선형 방법을 사용하여 데이터의 국소적 구조(데이터 포인트 사이의 가까운 관계)를 잘 유지한다.
            - 이로 인해 데이터의 클러스터링이나 군집 구조를 탐색하는 데 매우 효과적이다.
        
    """)

    st.markdown(""" """)
    st.markdown(""" """)

    st.subheader('📌소득분위를 기준으로 PCA, t-SNE을 진행한 데이터의 분포도 확인 결과', divider='gray')

    st.markdown("""
    - 예측하고자 하는 소득분위(1 ~ 10분위)를 기준으로 하여 PCA, t-SNE을 진행한 데이터의 분포도를 확인했을 때, 아래의 그래프와 같이 각 1 ~ 10분위에 해당하는 데이터의 분포가 정확하게 구분되지 않았음을 알 수 있다.
    - 예를 들어, 소득분위 1분위에 해당하는 데이터는 2분위의 범위에도 포함되어 있는 경우가 발견되며 다른 2, 3, 4, 5, 6, 7, 8, 9, 10 분위 범위에 포함된 데이터들도 경계가 모호하게 섞여 있음을 알 수 있다.
    - **전반적으로 큰 클러스터 내에 모든 소득 코드가 골고루 퍼져 있는 것을 볼 수 있으며, :blue[데이터가 크게 분리되거나 클러스터화되지 않고 전체적으로 연속적으로 분포]해 있음을 나타낸다.**
    """)


    # 한 페이지에 두개의 요소를 출력하도록 분할
    col1, col2 = st.columns(2, gap="large")

    with col1:

        # 데이터 스케일링 및 PCA 수행
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(total_df.drop('Income_code', axis=1))

        # PCA 분포 시각화
        st.subheader("PCA 2D Data Distribution")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=total_df['Income_code'], palette=sns.color_palette("Spectral", 10))
        plt.title('PCA 2D Data Distribution')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)


    with col2:
        # t-SNE 분포 시각화
        st.subheader("t-SNE 2D Data Distribution")
        img = Image.open('data/t-SNE 2D.png')
        st.image(img, caption='t-SNE 2D 결과')

    st.subheader('⭐ 결론', divider='gray')
    st.markdown("""
    - 사용하고자 하는 데이터는 그 값들의 경계가 모호하기 때문에 소득분위를 예측하고자 할 때, 정확하게 각 분위별로 구분을 할 수는 없지만 실제 소득분위와 유사하게 예측할 수 있다는 한계점이 발생했다.
    
    ▶ **소득 분위 결정 요인을 분석하기 위해 많은 하이퍼 파라미터 튜닝, 
         모델링 최적화 등을 해 본 결과 정확도가 좋지 않은 원인이 됨을 알 수 있다.**
    """)