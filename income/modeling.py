
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def modeling(total_df):
    st.markdown("### 소득분위 예측 모델링")
    col1, col2, col3 = st.columns(3)
    
    X = total_df.drop('Income_code', axis=1)
    y = total_df['Income_code'] - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    columns = columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    scores_df = pd.DataFrame(columns=columns)

    st.subheader("다양한 모델을 사용하여 성능 비교")
    st.markdown("""
        먼저 여러가지 모델을 사용하여 성능을 비교해보기위해 회귀모델인 Logistic Regression, 앙상블 모델인 XGBoost, 그리고 Random Forest를 사용해보았다. 
        결과는 아래와 같이 나왔다.         
        """)

    with col1: 
        # Logistic Regression 모델
        st.markdown("#### Logistic Regression")
        
        lr_cm = Image.open('data\income\Logistic Regression_cm.png')
        lr_roc = Image.open('data\income\Logistic Regression_roc.png')
        st.image(lr_cm)
        st.image(lr_roc)
        

    with col2:
        # XGBoost 모델
        st.markdown("#### XGBoost")
        
        xg_cm = Image.open('data\income\XGBoost_cm.png')
        xg_roc = Image.open('data\income\XGBoost_roc.png')
        st.image(xg_cm)
        st.image(xg_roc)

    with col3: 
        # Random Forest 모델
        st.markdown("#### Random Forest")
        
        rfc_cm = Image.open('data\income\Basic Random Forest_cm.png')
        rfc_roc = Image.open('data\income\Basic Random Forest_roc.png')
        st.image(rfc_cm)
        st.image(rfc_roc)


    scores_df = pd.read_csv('data\income\income_model_scores.csv')
    st.write(scores_df)
    
    st.markdown("""
        결과는 기본적으로 크게 
    """)
    st.markdown("""
        **그래서 우리는 RFC를 쓰기로 했다... 어쩌구**
    """)



    rfccol1, rfccol2 = st.columns(2)
    rfc_scores = pd.read_csv('data\income\income_model_rfc_scores.csv')

    with rfccol1:
        # Sampling 적용한 Random Forest 모델
        st.markdown("#### Sampling data Random Forest")
         
        smotek_rfc_cm = Image.open('data\income\Random Forest with SMOTETomek Sampling_cm.png')
        smotek_rfc_roc = Image.open('data\income\Random Forest with SMOTETomek Sampling_roc.png')
        st.image(smotek_rfc_cm)
        st.image(smotek_rfc_roc)

    with rfccol2: 
        # Balanced Random Forest
        st.markdown("#### Balanced Random Forest")

        brfc_cm = Image.open('data\income\Balanced Random Forest_cm.png')
        brfc_roc = Image.open('data\income\Balanced Random Forest_roc.png')
        st.image(brfc_cm)
        st.image(brfc_roc)

    st.write(rfc_scores)
    

    st.markdown("""
        **그래서 우리는 그리드 서치를 했다...!!**
        **여기에 샘플링기법이랑 밸런스트리도 썼다..!!!!**
    """)

    rfcoptcol1, rfcoptcol2 = st.columns(2)
    optrfc_scores = pd.read_csv('data\income\income_model_opt_rfc_scores.csv')

    with rfcoptcol1:
        # Optimized RFC with sampling 
        st.markdown("#### Optimized RFC with sampling")

        opt_smote_rfc_cm = Image.open('data\income\Optimized RFC with SMOTE_cm.png')
        opt_smote_rfc_roc = Image.open('data\income\Optimized RFC with SMOTE_roc.png')
        st.image(opt_smote_rfc_cm)
        st.image(opt_smote_rfc_roc)

    with rfcoptcol2:

        # st.markdown("#### Optimized RFC without sampling")
        st.markdown("#### Optimized RFC without sampling")
        opt_rfc_cm = Image.open('data\income\Optimized RFC_cm.png')
        opt_rfc_roc = Image.open('data\income\Optimized RFC_roc.png')
        st.image(opt_rfc_cm)
        st.image(opt_rfc_roc)

    st.write(optrfc_scores)
    
    st.markdown("""----""")
    st.subheader("3분위 분할 예측")
    # Optimized RFC without sampling
    tricol1, tricol2 = st.columns(2)

    with tricol1: 
         st.markdown("""
                ``` Accuracy: 0.7909

                    Precision (Weighted): 0.7992

                    Recall (Weighted): 0.7909

                    F1 Score (Weighted): 0.7940

                    ROC-AUC: 0.9118
            """)
    with tricol2: 
        tri_cm =  Image.open('data/income/trisect_income_cm.png')
        tri_roc = Image.open('data/income/trisect_income_roc.png')
        st.image(tri_cm)
        st.image(tri_roc)

    st.markdown("""### 보너스 우리 LSTM도 해봤다!!""")
     # LSTM 모델 결과 시각화 (외부 이미지로 가정)
    st.markdown("#### LSTM 딥러닝 모델")
    img = Image.open('data/LSTM_img.png')
    st.image(img, caption='LSTM 모델 결과')