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
from sklearn.metrics import accuracy_score, roc_auc_score
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


def income_run(total_df): 
    st.header('소득 분위 예측')
    st.markdown("""---""")


    tab1, tab2, tab3, tab4 = st.tabs(["특성 공학", "PCA 데이터 분포도 확인", 
                    "요인 분석", "소득분위 예측 모델링"])

    '''
    with tab1:
        st.markdown('### Feature Engineering(특성 공학)')
        st.write("다중공선성 분석")
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

        # VIF 결과 출력
        st.write(vif_data)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature', y='VIF', data=vif_data)
        plt.title('Variance Inflation Factor (VIF)')
        plt.xticks(rotation=90)
        st.pyplot(plt)




    with tab2:
        st.markdown('### 데이터 분포도 확인')
        st.markdown("""
            - 예측하고자 하는 소득분위(1~10분위)를 기준으로 데이터의 분포도를 확인.
            - PCA는 데이터의 전반적인 분포를 이해하고, 주성분이 데이터의 변동성을 어떻게 설명하는지 확인하는 데 유용하고
            - t-SNE는 데이터의 클러스터링이나 국소적인 데이터 구조를 명확하게 드러내는 데 적합하기 때문에 두 가지의 분포도를 모두 확인
            
            -> 소득 분위 결정 요인을 분석하기 위해 많은 하이퍼 파라미터 튜닝, 모델링 최적화 등을 해 본 결과 정확도가 좋지 않은 원인을 찾기 위해 시도
  
            📌소득분위를 기준으로 PCA 2D(2차원으로 축소된 데이터) 데이터의 분포도 확인 결과


            - 예측하고자 하는 소득분위(1 ~ 10분위)를 기준으로 하여 PCA 2D 데이터의 분포도를 확인했을 때, 아래의 그래프와 같이 각 1 ~ 10분위에 해당하는 데이터의 분포가 정확하게 구분되지 않았음을 알 수 있다.
            - 예를 들어, 소득분위 1분위에 해당하는 데이터는 2분위의 범위에도 포함되어 있는 경우가 발견되며 다른 2, 3, 4, 5, 6, 7, 8, 9, 10 분위 범위에 포함된 데이터들도 경계가 모호하게 섞여 있음을 알 수 있다.


            ⭐ 결론

            -  사용하고자 하는 데이터는 그 값들의 경계가 모호하기 때문에 소득분위를 예측하고자 할 때, 정확하게 각 분위별로 구분을 할 수는 없지만 실제 소득분위와 유사하게 예측할 수 있다는 한계점이 발생했다.""")

        # PCA & feature scaling
        # 피처 스케일링
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(total_df)


        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        # 결과 시각화 (PCA 2D 시각화)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=total_df['Income_code'], palette=sns.color_palette("Spectral", 10))
        plt.title('PCA 2D data distribution')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        st.pyplot(plt)

        with st.spinner('Wait for it...'):
    

            # TSNE로 고차원 데이터를 2차원으로 시각화
            tsne = TSNE(n_components=2, random_state=42)
            tsne_data = tsne.fit_transform(scaled_data)

            # 클러스터링 결과 시각화 (TSNE 2D 시각화)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=total_df['Income_code'], palette=sns.color_palette("Spectral", 10))
            plt.title('t-SNE 2D data distribution')
            plt.xlabel('TSNE Component 1')
            plt.ylabel('TSNE Component 2')
            st.pyplot(plt)


    with tab3:
        st.markdown("### 요인 분석")
        st.markdown("""#### 랜덤포레스트 회귀 모델(Random Forest Regression)을 통한 요인 분석""")
        with st.spinner('Wait for it...'):
            # 데이터 준비 (예시로 'Income_code'를 타겟 변수로 설정)
            X = total_df.drop('Income_code', axis=1)
            y = total_df['Income_code']

            # 학습 데이터와 테스트 데이터로 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # 랜덤 포레스트 회귀 모델 초기화
            model = RandomForestRegressor(random_state=42)

            # 하이퍼파라미터 그리드 설정
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            # GridSearchCV 설정 (MSE, R², MAE 등의 회귀 성능 지표 사용 가능)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

            # GridSearchCV로 모델 학습
            grid_search.fit(X_train, y_train)

            # 최적의 모델로 예측 수행
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # 모델 평가
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 평가 수치 출력
            evaluation_metrics = {
                'Mean Squared Error (MSE)': mse,
                'Mean Absolute Error (MAE)': mae,
                'R^2 Score': r2
            }

            evaluation_metrics_df = pd.DataFrame.from_dict(evaluation_metrics, orient='index', columns=['Score'])
            st.write("모델 평가 수치:")
            st.write(evaluation_metrics_df)

            # 피처 중요도 시각화
            feature_importances = pd.DataFrame(best_model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(x=feature_importances['importance'], y=feature_importances.index)
            plt.title('Feature Importances in Random Forest Regression')
            st.pyplot(plt)
    '''

    with tab4: 

        st.markdown("### 소득분위 예측 모델링")
        st.markdown("#### ")

        tc = 'Income_code'

        # 성능 평가 함수
        def evaluate_model(model, X_test, y_test, model_name):
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # 평가 지표 계산
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average='weighted')
            recall = metrics.recall_score(y_test, y_pred, average='weighted')
            f1 = metrics.f1_score(y_test, y_pred, average='weighted')
            roc_auc = metrics.roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

            # 결과 출력
            st.write(f'{model_name} Evaluation Metrics:')
            st.write(f'Accuracy: {accuracy:.4f}')
            st.write(f'Precision (Weighted): {precision:.4f}')
            st.write(f'Recall (Weighted): {recall:.4f}')
            st.write(f'F1 Score (Weighted): {f1:.4f}')
            st.write(f'ROC-AUC: {roc_auc:.4f}')

            # 혼동 행렬 시각화
            cm = metrics.confusion_matrix(y_test, y_pred)
            fig1 = plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap='Greens', square=True, cbar=False)

            # Income_code의 고유 값에 맞게 xticks, yticks 설정
            labels = sorted(y.unique())  # 고유값을 정렬하여 레이블로 사용
            plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45)
            plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45)

            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'{model_name} Confusion Matrix')
            st.pyplot(plt)


            # ROC Curve 시각화
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob, pos_label=model.classes_[1])
            fig2 = plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            return (fig1, fig2)

        x = total_df.drop(tc, axis=1)
        y = total_df[tc] - 1


        # 소득분위를 3개 단위로 구분
        def tri_y(x):
            if x < 4:
                return 0
            elif x < 7:
                return 1
            else:
                return 2


        # 데이터 스케일링
        def scale(x1, x2):
            x1scaler = StandardScaler().set_output(transform="pandas")
            x2scaler = StandardScaler().set_output(transform="pandas")
            x1_s = x1scaler.fit_transform(x1)
            x2_s = x2scaler.fit_transform(x2)
            return x1_s, x2_s

        xb1, xb2, yb1, yb2 = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

        xb1, xb2 = scale(xb1, xb2)


        ## 모델링 시작
        # Logistic Regression으로 클래스 분류(소득 분위 10개)
        
        st.markdown("#### Logistic Regression 모델")
        lr_x1, lr_x2, lr_y1, lr_y2 = train_test_split(x, y, stratify=y, test_size=0.3, random_state=42)

        train_scaler = StandardScaler().set_output(transform="pandas")
        test_scaler = StandardScaler().set_output(transform="pandas")

        lr_x1, lr_x2 = scale(lr_x1, lr_x2)

        kf = StratifiedKFold(n_splits=2, shuffle=False)


        lr = LogisticRegression()

        cross_val_score(lr, lr_x1, lr_y1, cv=kf, scoring='roc_auc_ovr')

        lr.fit(lr_x1, lr_y1)


        evaluate_model(lr, lr_x2, lr_y2, 'Logistic Regression')



        # XGBoost 모델
        st.write("XGBoost 모델")

        xg_x1 , xg_x2 , xg_y1 , xg_y2 = train_test_split(
            x,
            y,
            random_state=42
        )

        xg_x1, xg_x3, xg_y1, xg_y3 = train_test_split(
            xg_x1,
            xg_y1,
            random_state=42
        )




        xgb_clf = XGBClassifier(
            n_estimators=1000, # 학습 횟수
            learning_rate = 0.05, # 학습률(eta)
            max_depth=3,
            eval_metric='mlogloss'
        )
        xgb_clf.fit(xg_x1, xg_y1, verbose=True)

        evals = [
            (xg_x1, xg_y1),
            (xg_x3, xg_y3)
        ]

        xgb_clf.fit(
            xg_x1,
            xg_y1,
            early_stopping_rounds = 50,
            eval_set=evals,
            verbose=False
        )

        xgboost_fig, _ = evaluate_model(xgb_clf, xg_x2, xg_y2, 'XGBoost')


        # Random Forest 모델
        st.write("RandomForest 모델")
        rfc_x1, rfc_x2, rfc_y1, rfc_y2 = train_test_split(x, y, test_size=0.3, random_state=42)

        rfc_x1, rfc_x2 = scale(rfc_x1, rfc_x2)

        forest = RandomForestClassifier(n_estimators=200, random_state=42)
        forest.fit(rfc_x1, rfc_y1)

        y_pred = forest.predict(rfc_x2)

        fig, _  = evaluate_model(forest, rfc_x2, rfc_y2, 'Random Forest')


        st.markdown('''
                ```def gridsearch(estimator, X, Y, params =  {
                    'n_estimators' : [100, 200, 300],
                    'criterion': ['gini', 'entropy', 'logloss'],
                    'min_samples_split': [2, 3, 5, 10, 30],
                    'class_weight': ['balanced', 'balanced_subsample'],
                    'min_weight_fraction_leaf': [0.2, 0,4, 0.5]
                }):

                    grid = GridSearchCV(
                        RandomForestClassifier(random_state=42),
                        param_grid = params,
                        scoring='accuracy',
                        cv=5
                    )

                    grid.fit(X, Y)
                    print(grid.best_estimator_, grid.best_score_, grid.best_params_)```
                    
                    RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       min_samples_split=10, min_weight_fraction_leaf=0,
                       n_estimators=200, random_state=42)
                       0.4228943694741741
                      {
                        'class_weight': 'balanced',
                        'criterion': 'entropy',
                        'min_samples_split': 10,
                        'min_weight_fraction_leaf': 0,
                        'n_estimators': 200
                      } ''')

        # Random Forest 모델 + undersampling / oversampling
        st.write("Random Forest 모델 + undersampling / oversampling")
        kf = StratifiedKFold(n_splits=5, shuffle=False)
        test_rfc = RandomForestClassifier(n_estimators=200, random_state=42)

        ros = RandomOverSampler(random_state=42)
        rus = RandomUnderSampler(random_state=42)
        smote = SMOTE(random_state=42)
        tomekU = TomekLinks()
        smotomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))

        smote_x, smote_y = smotomek.fit_resample(xb1, yb1)
        test_rfc.fit(smote_x, smote_y)

        fig_smot, _ = evaluate_model(test_rfc, xb2, yb2, 'undersampler')

        # Balanced_RandomForest 모델 
        st.write("Balanced_RandomForest")
        balanced_rfc = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        balanced_rfc.fit(xb1, yb1)

        graph = evaluate_model(balanced_rfc, xb2, yb2, 'Balanced RFC')

        rfc_opt_nosample = RandomForestClassifier(class_weight='balanced', criterion='entropy',
                       min_samples_split=10, min_weight_fraction_leaf=0,
                       n_estimators=200, random_state=42)


        rfc_opt_nosample.fit(xb1, yb1)
        opt_fig, _ = evaluate_model(rfc_opt_nosample, xb2, yb2, 'Optimized RFC')
        st.write(cross_val_score(rfc_opt_nosample, smote_x, smote_y, cv=kf, scoring='roc_auc_ovr'))


        # LSTM 딥러닝 모델
        st.write("Accuracy: 0.4267")
        st.write("Precision: 0.4217")
        st.write("Recall: 0.4267")
        st.write("F1 Score")
        st.write("ROC AUC Score: 0.8620")


        img = Image.open('data\LSTM_img.png')
        # 경로에 있는 이미지 파일을 통해 변수 저장
        st.image(img)
        # streamlit를 통해 이미지를 보여준다.

        
            




