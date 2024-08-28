
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import plotly.express as px
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

def evaluate_model(model, X_test, y_test, model_name):
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average='weighted')
            recall = metrics.recall_score(y_test, y_pred, average='weighted')
            f1 = metrics.f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

            st.write(f'{model_name} 평가 지표:')
            st.write(f'Accuracy: {accuracy:.4f}')
            st.write(f'Precision (Weighted): {precision:.4f}')
            st.write(f'Recall (Weighted): {recall:.4f}')
            st.write(f'F1 Score (Weighted): {f1:.4f}')
            st.write(f'ROC-AUC: {roc_auc:.4f}')

            # 혼동 행렬 시각화
            st.subheader(f'{model_name}')
            cm = metrics.confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap='Greens', square=True, cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'{model_name} Confusion Matrixs')
            st.pyplot(plt)

            # ROC Curve 시각화
            st.subheader(f'{model_name} ROC Curve')
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob, pos_label=model.classes_[1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            return pd.DataFrame([{
                    'Accuracy': accuracy, 
                    'Precision': precision, 
                    'Recall': recall, 
                    'F1 Score': f1, 
                    'ROC-AUC': roc_auc
                    }])




def modeling(total_df):
    st.markdown("### 소득분위 예측 모델링")
    col1, col2, col3 = st.columns(3)
    
    X = total_df.drop('Income_code', axis=1)
    y = total_df['Income_code'] - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    columns = columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    scores_df = pd.DataFrame(columns=columns)

    with col1: 
        # Logistic Regression 모델
        st.markdown("#### Logistic Regression")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        lr_scores = evaluate_model(lr, X_test, y_test, 'Logistic Regression')
        
        scores_df = pd.concat([scores_df, lr_scores], ignore_index=True)

    with col2:
        # XGBoost 모델
        st.markdown("#### XGBoost")
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        xgb_scores = evaluate_model(xgb, X_test, y_test, 'XGBoost')
        scores_df = pd.concat([scores_df, xgb_scores], ignore_index=True)

    with col3: 
        # Random Forest 모델
        st.markdown("#### Random Forest")
        forest = RandomForestClassifier(n_estimators=200, random_state=42)
        forest.fit(X_train, y_train)
        rfc_scores = evaluate_model(forest, X_test, y_test, 'Random Forest')
        scores_df = pd.concat([scores_df, rfc_scores], ignore_index=True)
    

    st.markdown("""
        **그래서 우리는 RFC를 쓰기로 했다... 어쩌구**
    """)



    rfccol1, rfccol2 = st.columns(2)
    rfc_scores = pd.DataFrame(columns=columns)

    with rfccol1:
        # Sampling 적용한 Random Forest 모델
        st.markdown("#### Sampling data Random Forest")
        rfc = RandomForestClassifier(n_estimators=200, random_state=42)
        smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        rfc.fit(X_resampled, y_resampled)
        evaluate_model(forest, X_test, y_test, 'Random Forest with Sampling')

    with rfccol2: 
        # Balanced Random Forest
        st.markdown("#### Balanced Random Forest")
        balanced_forest = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        balanced_forest.fit(X_train, y_train)
        evaluate_model(balanced_forest, X_test, y_test, 'Balanced Random Forest')

    st.markdown("""
        **그래서 우리는 그리드 서치를 했다...!!**
        **여기에 샘플링기법이랑 밸런스트리도 썼다..!!!!**
    """)

    rfcoptcol1, rfcoptcol2 = st.columns(2)
    with rfcoptcol1:
        # Optimized RFC with sampling 
        rfc_opt = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                        min_samples_split=10, min_weight_fraction_leaf=0,
                        n_estimators=200, random_state=42)

        rfc_opt.fit(X_resampled, y_resampled)
        evaluate_model(rfc_opt, X_test, y_test, 'Optimized RFC with SMOTE')

    with rfcoptcol2:
    # Optimized RFC without sampling
        rfc_opt_nosample = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                        min_samples_split=10, min_weight_fraction_leaf=0,
                        n_estimators=200, random_state=42)

        rfc_opt_nosample.fit(X_train, y_train)
        evaluate_model(rfc_opt_nosample, X_test, y_test, 'Optimized RFC')
        
    
    st.markdown("""### 보너스 우리 LSTM도 해봤다!!""")
     # LSTM 모델 결과 시각화 (외부 이미지로 가정)
    st.markdown("#### LSTM 딥러닝 모델")
    img = Image.open('data/LSTM_img.png')
    st.image(img, caption='LSTM 모델 결과')