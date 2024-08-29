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
    st.markdown("""
        예측 결과값에 영향을 끼치는 피쳐의 중요도를 계산하기 위해 Regression모델을 사용하여 분석해 보았다.
        Random Forest Regressor를 사용하여 최적의 파라미터를 그리드서치로 알아보고, MSE, MAE, R²값을 확인하는것으로 진행했다.
    """)

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.markdown("### 최적의 하이퍼파라미터 값")
        grid_best_param = {
            "max_depth":10,
            "min_samples_leaf":4,
            "min_samples_split":2,
            "n_estimators":300
        }
        st.dataframe(pd.DataFrame.from_dict(grid_best_param, orient='index', columns=['Value']), use_container_width=True)

    
    with col2:
        evaluation_metrics = {
            'Mean Squared Error (MSE)': 1.4362,
            'Mean Absolute Error (MAE)': 0.869,
            'R^2 Score': 0.7371
        }
        
        st.markdown("### 모델 성능 지표")
        st.dataframe(pd.DataFrame.from_dict(evaluation_metrics, orient='index', columns=['Score']), use_container_width=True)

    st.markdown("""
        - **MSE: 1.4362**
            - 예측값과 실제값 간의 제곱 오차가 평균적으로 1.44 정도임을 나타내며, MSE가 너무 크지 않다면, 모델이 큰 오차 없이 예측을 수행하고 있음을 의미할 수 있다.
        - **MAE: 0.869**
            - 예측값과 실제값 간의 평균적인 절대 오차가 0.87 정도임을 나타내며, 모델의 예측이 실제값에 비교적 근접하고 있음을 시사한다.
        - **R² Score: 0.7371**
            - 모델이 종속 변수의 변동성을 약 73.7% 설명할 수 있음을 나타내며, 이는 모델이 데이터를 상당히 잘 설명하고 있음을 의미한다.
    """)

    # 피처 중요도 시각화
    st.subheader("Feature Importances in Random Forest Regression")
    st.markdown("""
        Random Forest Regressor에서 나타난 Feature Importance는 아래와 같았다.
                
        가장 높은 요인은 비소비지출 값으로, 소득만큼 세금과 연금이 비례하게 나감을 생각하면 타당한 결과이다. 
        다음으로 높은 소비지출또한 소득 이상으로 소비하는 일이 잘 없음을 생각하면 타당하다.
                
        그 외 요소로는 가구원 수, 자산, 가구주 동거여부등이 있었다.
    """)
    img = Image.open('data/randomforest_regression.png')
    st.image(img, caption='randomforest_regression결과', use_column_width=True)
    st.markdown("""
        
    """)