# -*- coding:utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 모델링 import
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



from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor


def income_run(total_df): 
    st.header('Income Decile Prediction')
    st.markdown("""---""")

    # Tab creation for different steps
    tab1, tab2, tab3, tab4 = st.tabs(["특성공학(Feature Engineering)", "데이터 분포도 확인(Data Distribution)", "요인 분석(Factor Analysis)", "모델링(Modeling)"])

    # Feature Engineering
    with tab1:
        st.markdown('### 특성공학(Feature Engineering)')


        # Split into two columns: VIF Data and VIF Plot
        top_col1, top_col2 = st.columns([3, 5], gap="small", vertical_alignment="center")

        col1, col2 = st.columns([3, 5], gap="small", vertical_alignment="center")
        

        with top_col1:
            st.markdown("#### VIF 분석 결과 순서")

        with top_col2:
            st.markdown("#### VIF 분석 결과 시각화")

        with col1:
            # Prepare data for VIF analysis
            X = total_df.drop('Income_code', axis=1)
            X = add_constant(X)

            # Calculate VIF
            vif_data = pd.DataFrame()
            vif_data["Feature"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif_data = vif_data.iloc[1:]  # 상수항(constant)을 제거

            # VIF 데이터프레임 출력
            st.write(vif_data)

        with col2:
            # VIF 시각화
            plt.figure(figsize=(4, 4))
            sns.barplot(x='VIF', y='Feature', data=vif_data)
            plt.title('Variance Inflation Factor (VIF)')
            plt.xticks(rotation=90)
            st.pyplot(plt)

    # PCA 및 t-SNE 분포 분석
    with tab2:
        st.markdown('### 데이터 분포도 확인')

        st.markdown("""
        - 예측하고자 하는 소득분위(1~10분위)를 기준으로 데이터의 분포도를 확인.
        - PCA는 데이터의 전반적인 분포를 이해하고, 주성분이 데이터의 변동성을 어떻게 설명하는지 확인하는 데 유용하고
        - t-SNE는 데이터의 클러스터링이나 국소적인 데이터 구조를 명확하게 드러내는 데 적합하기 때문에 두 가지의 분포도를 모두 확인
            
        -> 소득 분위 결정 요인을 분석하기 위해 많은 하이퍼 파라미터 튜닝, 모델링 최적화 등을 해 본 결과 정확도가 좋지 않은 원인을 찾기 위해 시도.
  
        📌소득분위를 기준으로 PCA 2D(2차원으로 축소된 데이터) 데이터의 분포도 확인 결과

        - 예측하고자 하는 소득분위(1 ~ 10분위)를 기준으로 하여 PCA 2D 데이터의 분포도를 확인했을 때, 아래의 그래프와 같이 각 1 ~ 10분위에 해당하는 데이터의 분포가 정확하게 구분되지 않았음을 알 수 있다.
        - 예를 들어, 소득분위 1분위에 해당하는 데이터는 2분위의 범위에도 포함되어 있는 경우가 발견되며 다른 2, 3, 4, 5, 6, 7, 8, 9, 10 분위 범위에 포함된 데이터들도 경계가 모호하게 섞여 있음을 알 수 있다.
        
        ⭐ 결론
        - 사용하고자 하는 데이터는 그 값들의 경계가 모호하기 때문에 소득분위를 예측하고자 할 때, 정확하게 각 분위별로 구분을 할 수는 없지만 실제 소득분위와 유사하게 예측할 수 있다는 한계점이 발생했다.
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


    # 요인 분석
    with tab3:

        st.markdown("### 요인 분석 (Random Forest Regression)")
        X = total_df.drop('Income_code', axis=1)
        y = total_df['Income_code']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # GridSearchCV는 시간이 너무 오래 걸려서 생략
        '''
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        st.write(grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
        '''
        col1, col2 = st.columns(2, gap="large")

        with col1:

            st.markdown("### 최적의 하이퍼 파라미터 값")
            grid_best_param = {
                "max_depth":10,
                "min_samples_leaf":4,
                "min_samples_split":2,
                "n_estimators":300
            }
            st.write(pd.DataFrame.from_dict(grid_best_param, orient='index', columns=['Value']))

        
        with col2:
            evaluation_metrics = {
                'Mean Squared Error (MSE)': 1.4362,
                'Mean Absolute Error (MAE)': 0.869,
                'R^2 Score': 0.7371
            }
            
            st.markdown("### 모델 성능 지표")
            st.write(pd.DataFrame.from_dict(evaluation_metrics, orient='index', columns=['Score']))

    
    
        # 피처 중요도 시각화
        st.subheader("Feature Importances in Random Forest Regression")
        img = Image.open('data/randomforest_regression.png')
        st.image(img, caption='randomforest_regression결과', width=900)
        

    # 모델링
    with tab4:
        st.markdown("### 소득분위 예측 모델링")
        
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
            st.subheader(f'{model_name} Confusion Matrix')
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

        X = total_df.drop('Income_code', axis=1)
        y = total_df['Income_code'] - 1

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Logistic Regression 모델
        st.markdown("#### Logistic Regression")
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        evaluate_model(lr, X_test, y_test, 'Logistic Regression')

        # XGBoost 모델
        st.markdown("#### XGBoost")
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        evaluate_model(xgb, X_test, y_test, 'XGBoost')

        # Random Forest 모델
        st.markdown("#### Random Forest")
        forest = RandomForestClassifier(n_estimators=200, random_state=42)
        forest.fit(X_train, y_train)
        evaluate_model(forest, X_test, y_test, 'Random Forest')

        # Sampling 적용한 Random Forest 모델
        st.markdown("#### Sampling data Random Forest")
        smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        forest.fit(X_resampled, y_resampled)
        evaluate_model(forest, X_test, y_test, 'Random Forest with Sampling')

        # Balanced Random Forest
        st.markdown("#### Balanced Random Forest")
        balanced_forest = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        balanced_forest.fit(X_train, y_train)
        evaluate_model(balanced_forest, X_test, y_test, 'Balanced Random Forest')

        '''
        # LSTM 모델 결과 시각화 (외부 이미지로 가정)
        st.markdown("#### LSTM 딥러닝 모델")
        img = Image.open('data/LSTM_img.png')
        st.image(img, caption='LSTM 모델 결과')
        '''
    