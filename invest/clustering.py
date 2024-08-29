import streamlit as st 
from PIL import Image

def cluster(): 
    st.markdown("""
        ## 📌 주제

        ```
            ❓ 위 분석을 통해 소득을 예측한 뒤 소득이 높은 고객을 선별하여 
        가계 정보 데이터를 통해 투자 성향 분석과 투자 상품 추천을 진행한다.

        ```


        ## 사용 데이터
            - 18095 rows X 74 columns
            - 인적 사항: 나이, 직업, 결혼유무, 가구유형 
            - 금융 정보: 자산, 부채, 소비, 소득 금액 세부 사항
            - 투자 성향: 여유 자금 투자 성향, 금융 자산 투자 방법, 금융 자산 투자시 고려 사항

        ## 분석 Flow

        새로운 고객이 어떤 cluster에 포함되는지 확인 후 해당 cluster 기존 고객들의 자산 비중 비율을 보여준다. 
        > *본 연구에서 제안하는 고객 세분화 기법의 전체 프레임워크는 크게 5단계로 구성된다. 
        > 1단계는 은행 내 각 시스템에서 관리하고 있는 데이터를 수집하는 단계이며, 
        > 2단계에서는 고객변수의 선택 및 변수를 활용하여 개인별 금융프로파일을 만들었다. 
        > 3단계에서 직업군별 연령대, 거주지역, 거래성향들을 고려하여 블록을 구성했으며, 
        > 4단계에서 이 블록을 클러스터링하여 고객 세그먼트를 완성하였다.
        > 마지막 단계는 각 그룹의 고객 특성을 이해하고 기존 세분화 모형과 차별성을 찾는것이다.*

        [참고논문](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO201808962641880) → 논문 참고하여 고객 세분화 및 cluster 별 고객 성향 파악

        - 인구통계
        - 저축성향
        - 소득소비성향
        - 접촉성향
        - 대출성향
        """)
    st.subheader("분석 방법")
    st.markdown("""
        1. **데이터 중 무응답이 50% 이상인 column은 분석에서 제외**
        2. **이상치 제거**
        3. **최종 선택된 Columns 대상으로 clustering 진행**
                
        """)
    st.markdown("""
        ##### Gower Distance 사용
        **💡 Gower Distance란?**
                
            
            서로 다른 데이터 유형 (연속형, 범주형, 이진형 등)을 
            모두 고려할 수 있도록 하는 유사도 측정 방법
            
        """)
    eqcol1, eqcol2, _ = st.columns(3)
    with eqcol1:
        st.markdown("""
            **1. 연속형 데이터 계산**
        """)
        st.latex(
                r"""S_{ij} = 1 - \frac{|x_{ik} - x_{jk}|}{\max(x_k) - \min(x_k)} """
        )
    with eqcol2: 
        st.markdown("""
            **2. 범주형 데이터 계산**
        """)
        st.latex(
                r"""S_{ij} = 
                    \begin{cases} 
                    1 & \text{if } x_{ik} = x_{jk} \\ 
                    0 & \text{if } x_{ik} \neq x_{jk} 
                    \end{cases}
                """
        )
    
    st.subheader("모델 결과")
    st.markdown("##### K-Prototype Clustering")
    st.markdown("**최종 선택 모델: K-Prototype Clustering with cao-10**")
    c1, c2, c3, c4,c5 = st.columns(5)
    _, h2, _, h4, _ = st.columns(5)

    with c1:
        st.markdown('cao-5')
        cao5img = Image.open('data/invest/clustering/cao-5.png')
        st.image(cao5img)
    with c2:
        st.markdown('**⭐cao-10⭐** 최종 선정 모델')
        cao10img = Image.open('data/invest/clustering/cao-10.png')
        st.image(cao10img)
    with c3:
        st.markdown('cao-15')
        cao15img = Image.open('data/invest/clustering/cao-15.png')
        st.image(cao15img)
    with c4:
        st.markdown('cao-20')
        cao20img = Image.open('data/invest/clustering/cao-20.png')
        st.image(cao20img)
    with c5:
        st.markdown('cao-30')
        cao30img = Image.open('data/invest/clustering/cao-30.png')
        st.image(cao30img)

    with h2:
        st.markdown('huang-10')
        hua10img = Image.open('data/invest/clustering/huang-10.png')
        st.image(hua10img)
    with h4:
        st.markdown('huang-25')
        hua25img = Image.open('data/invest/clustering/huang-25.png')
        st.image(hua25img)
    
    
