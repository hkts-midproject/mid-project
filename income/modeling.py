
import streamlit as st
import pandas as pd
from PIL import Image

def modeling():
    st.markdown("# 소득분위 예측 모델링")
    
    st.subheader("📌다양한 모델을 사용하여 성능 비교")
    st.markdown("""
        먼저 여러가지 모델을 사용하여 성능을 비교해보기위해 회귀모델인 Logistic Regression, 앙상블 모델인 XGBoost, 그리고 Random Forest를 사용해보았다. 
        결과는 아래와 같이 나왔다.         
        """)
    col1, col2, col3 = st.columns(3)
    

    
    columns =  ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    scores_df = pd.DataFrame(columns=columns)


    with col1: 
        # Logistic Regression 모델
        st.markdown("#### Logistic Regression")
        
        lr_cm = Image.open('data/income/Logistic Regression_cm.png')
        lr_roc = Image.open('data/income/Logistic Regression_roc.png')
        st.image(lr_cm)
        st.image(lr_roc)
        

    with col2:
        # XGBoost 모델
        st.markdown("#### XGBoost")
        
        xg_cm = Image.open('data/income/XGBoost_cm.png')
        xg_roc = Image.open('data/income/XGBoost_roc.png')
        st.image(xg_cm)
        st.image(xg_roc)

    with col3: 
        # Random Forest 모델
        st.markdown("#### Random Forest")
        
        rfc_cm = Image.open('data/income/Basic Random Forest_cm.png')
        rfc_roc = Image.open('data/income/Basic Random Forest_roc.png')
        st.image(rfc_cm)
        st.image(rfc_roc)


    scores_df = pd.read_csv('data/income/income_model_scores.csv', index_col='Model')
    st.dataframe(data=scores_df, use_container_width=True)
    
    st.markdown("""
        
        - **혼동 행렬 그래프**
            1. **정확한 예측 (대각선 요소들):**
                - 대각선에 위치한 값들은 모델이 정확히 예측한 사례를 나타낸다. 1분위의 경우 500대로 정확히 예측되었고, 2분위의 경우 300대가 정확히 예측되었다.
                - 대부분의 유사한 예측은 대각선상에 위치하게 되며, 대각선의 숫자가 크고 대각선상의 범주가 짙게 표시될 수록 모델의 성능이 좋다고 해석 할 수 있다.
            
            2. **오류 패턴 (비대각선 요소들):**
                - 대각선 이외의 값들은 모델이 잘못 예측한 사례를 나타낸다. 
                - 특정 클래스에서 다른 특정 클래스로의 오차가 많다면, 그 두 클래스 간의 분류가 어려운 경우가 될 수 있는데, 1-3분위 사이에 오차가 다소 많은 것으로 확인했다. 
                - 다만 데이터도 1-3분위에 치중해있음으로 비율상으로는 고분위 데이터의 예측이 더 어려운것으로 보인다.
        
        ##### ⭐ 요약
        결과는 Accuracy score 부터 분류학습 지표인 recall, f1등의 스코어를 비롯해 크게 차이가 나지는 않았다. 다만 10분위라는 큰 카테고리에 본질적으로 연속적인 성격의 데이터를 임의로 10등분한 만큼 적확한 분류에 어려움이 있다고 판단했다. 
        
        또한 0.85정도로 높게 나온 ROC-AUC Score와 혼돈행렬이 정확하게 예측함을 나타내는 대각선 방향에 치중되어있는 모양새로 정확한 값을 맞추는 accuracy score는 떨어지더라도 분위 자체는 타겟값에 근접한 범위로 예측하고 있다고 볼 수 있다.

        위 모델중에서는 ROC-AUC 값이 크게 떨어지지 않으며 Accuracy 와 Precision등이 높게 나온 Random Forest Classifier를 좀더 최적화 해보기로 하였다. 
            
        ----
    """)
    st.header("""
        **📌Random Forest Classifier의 개선**
    """)

    st.markdown("""
        - **문제**:  
            1분위에서 2분위가 각 데이터셋의 15~20%를 구성하고 있는데에 비해 9분위는 0.5% 밖에 되지 않는 데이터셋 불균등의 문제가 영향을 끼칠것으로 보인다. 
            
            결과의 혼돈 매트릭스도 확인 한바 데이터셋의 피처들도 불균형이 크고 예측값도 다소 1분위에 가깝게 쏠려있는것을 확인하여 이부분을 개선해보기로 결정했다. 
        
        - **개선책**
                
            1. RFC 하이퍼 파라미터 class_weight을 'balanced'로 설정: 라벨의 frequency에 반비례하게 웨이트를 설정하는 모드이다.
            2. 훈련 데이터세트에 SMOTE 오버샘플링과 Tomek Links 언더샘플링 적용: 
                - **SMOTE**: frequency가 적은 클래스의 feature들을 사용하여 클래스의 데이터를 생성해내는 기법
                - **Tomek Links**: 다수에 속하는 클래스들중 소수 클래스와 가까운 포인트를 더 적게 샘플링하는 밸런싱 기법
        
        """)


    rfccol1, rfccol2 = st.columns(2)
    rfc_scores = pd.read_csv('data/income/income_model_rfc_scores.csv', index_col='Model')

    with rfccol1:
        # Sampling 적용한 Random Forest 모델
        st.markdown("#### Sampling data Random Forest")
         
        smotek_rfc_cm = Image.open('data/income/Random Forest with SMOTETomek Sampling_cm.png')
        smotek_rfc_roc = Image.open('data/income/Random Forest with SMOTETomek Sampling_roc.png')
        st.image(smotek_rfc_cm)
        st.image(smotek_rfc_roc)

    with rfccol2: 
        # Balanced Random Forest
        st.markdown("#### Balanced Random Forest")

        brfc_cm = Image.open('data/income/Balanced Random Forest_cm.png')
        brfc_roc = Image.open('data/income/Balanced Random Forest_roc.png')
        st.image(brfc_cm)
        st.image(brfc_roc)

    st.dataframe(rfc_scores, use_container_width=True)
    st.markdown("""
        ##### ⭐ 결과
                
        역시나 성능이 크게 늘지는 않았으나 balanced random forest의 ROC-AUC가 높고 SMOTomek 샘플링의 precision이 조금 높게 나오는것으로 확인했다. 
                
        ----
    """)

    st.markdown("""
        ### 📌그리드서치 최적화
                
        보다 최적화하기 위해 다음의 하이퍼파라미터로 GridSearch를 하여 최적의 하이퍼파라미터를 볼드로 표기하였다. 
                
        - `n_estimators`: [100, **200**, 300], 
        - `criterion`: ['gini', **'entropy'**, 'logloss'],
        - `min_samples_split`: [2, 3, 5, **10**, 30]
        - `class_weight`: ['**balanced**', 'balanced_subsample'],
        - `min_weight_fraction_leaf`: [**0**, 0.2, 0.5,  4]
                

        최적화된 Random Forest Tree에 각각 샘플링 기법을 사용한 훈련세트와 사용하지 않은 훈련세트로 성능을 테스트 해보았다.
                
        ##### ⭐결과
    """)
    rfcoptcol1, rfcoptcol2 = st.columns(2)
    optrfc_scores = pd.read_csv('data/income/income_model_opt_rfc_scores.csv', index_col='Model')

    with rfcoptcol1:
        # Optimized RFC with sampling 
        st.markdown("#### Optimized RFC with sampling")

        opt_smote_rfc_cm = Image.open('data/income/Optimized RFC with SMOTE_cm.png')
        opt_smote_rfc_roc = Image.open('data/income/Optimized RFC with SMOTE_roc.png')
        st.image(opt_smote_rfc_cm)
        st.image(opt_smote_rfc_roc)

    with rfcoptcol2:

        # st.markdown("#### Optimized RFC without sampling")
        st.markdown("#### Optimized RFC")
        opt_rfc_cm = Image.open('data/income/Optimized RFC_cm.png')
        opt_rfc_roc = Image.open('data/income/Optimized RFC_roc.png')
        st.image(opt_rfc_cm)
        st.image(opt_rfc_roc)

    st.dataframe(optrfc_scores, use_container_width=True)
    
    st.markdown("""
        그리드서치 실행시 샘플링 하지 않은 데이터로 한 만큼 샘플링 하지 않은 쪽의 Random Forest의 결과값이 전반적으로 더 좋게 나왔다.         
    """)

    st.markdown("""----""")
    st.subheader("📌3분위 분할 예측")

    st.markdown("""
        - **문제**:
            - 소득분위는 이산적인 데이터 이지만 본질적으로는 소득이라는 연속적인 데이터를 임의로 끊어낸 구간을 의미하는 만큼 10분위라는 세부적인 카테고리로 정확하게 예측하는데에는 어려움이 있다고 판단하였다.
            - PCA는 잠재적 요인일 뿐이지만 차원을 단순화하여 시각화 했을때도 명확하게 구분되는 점이 없음을 시사했다.
        """)
    # Optimized RFC without sampling
    tricol1, tricol2 = st.columns(2)

    with tricol1: 
        st.dataframe(pd.DataFrame.from_dict(
            {
                'Accuracy': 0.7909,
                'Precision': 0.7992,
                'Recall' : 0.7909,
                'F1 Score' : 0.7940,
                'ROC-AUC' : 0.9118,
            },
            orient='index',
            columns=['value']
        ), use_container_width=True)
        st.markdown("""
        - **해결책**:
            - 10분위만큼 세분화 하지 않아도 중,상,하의 3분위로 구분하여 정확도를 올리기로 결정하였다.
                    
        - **결과** 
            - 확실히 0.91대로 개선된 ROC-AUC와 0.8에 가까운 스코어를 확인할 수 있었다.

            """)
    with tricol2: 
        tri_cm =  Image.open('data/income/trisect_income_cm.png')
        tri_roc = Image.open('data/income/trisect_income_roc.png')
        st.image(tri_cm)
        st.image(tri_roc)

    st.markdown("""### ⭐ 결론 / 요약""")
    st.markdown("""
        데이터에 불균형이 심한것을 감안하여 이런저런 시행착오를 겪으며 회귀 모델, 앙상블 모델, 트리 모델등부터 시작하여 샘플링기법과 GridSearch를 통하여 결과값을 최적화 해 본 결과 Balanced Random Forest Classifier를 사용하는것이 가장 낫다고 판단했다.
        또한 피처의 분포가 겹치는것이 많은 연속적 데이터 (EDA, 요인분석 탭 참고)를 많은 다수 카테고리로 분류하는데에 한계가 있음을 확인, 
        3분위정도로 낮춰 분류하는것으로 에둘러 성능을 늘리기도 해보았다. 
        
        ---
                
        **참고자료**
        https://www.kaggle.com/code/marcinrutecki/best-techniques-and-metrics-for-imbalanced-dataset/notebook
    """)
    