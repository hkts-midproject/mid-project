import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def factors(total_df):
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
        
