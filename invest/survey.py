import streamlit as st
import pandas as pd
import streamlit_survey as ss
import pickle
import numpy as np
from PIL import Image
import os

# 
columns = [
 '가구원수',
 '자산',
 '자산_금융자산_저축금액',
 '자산_금융자산_저축_적립예치식저축_주식채권펀드금액',
 '부채',
 '경상소득(보완)',
 '지출_소비지출비',
 '지출_비소비지출(보완)',
 '지출_비소비지출_공적연금사회보험료(보완)',

 '월지출비',
 '자산대비부채비율',
 '자산대비저축금액',
 '자산대비주식채권펀드',
]


intcols_cat = [
    '가구주_성별코드',
    '가구주연령_10세단위코드',
    '가구주_종사상지위(보도용)',
    '가구주_혼인상태코드',
    '가구주_은퇴여부',
]
intcols_num = [
    "가구원수",
    "자산",
    "자산_금융자산_저축금액",
    "자산_금융자산_저축_적립예치식저축_주식채권펀드금액",
    "부채",
    "경상소득(보완)",
    "지출_소비지출비",
    "지출_비소비지출(보완)",
    "지출_비소비지출_공적연금사회보험료(보완)",
]

code_columns = [
 '가구주_성별코드_1',
 '가구주_성별코드_2',
 '가구주연령_10세단위코드_0',
 '가구주연령_10세단위코드_1',
 '가구주연령_10세단위코드_2',
 '가구주연령_10세단위코드_3',
 '가구주연령_10세단위코드_4',
 '가구주_종사상지위(보도용)_0',
 '가구주_종사상지위(보도용)_1',
 '가구주_종사상지위(보도용)_2',
 '가구주_종사상지위(보도용)_3',
 '가구주_혼인상태코드_1',
 '가구주_혼인상태코드_2',
 '가구주_혼인상태코드_3',
 '가구주_혼인상태코드_4',
 '원리금연체여부_1.0',
 '원리금연체여부_2.0',
 '가구주_은퇴여부_1',
 '가구주_은퇴여부_2',
 '가구주_미은퇴_노후준비상황코드_1.0',
 '가구주_미은퇴_노후준비상황코드_2.0',
 '가구주_미은퇴_노후준비상황코드_3.0',
 '가구주_미은퇴_노후준비상황코드_4.0',
 '가구주_미은퇴_노후준비상황코드_5.0',
 '가구주_은퇴_적정생활비충당여부_1.0',
 '가구주_은퇴_적정생활비충당여부_2.0',
 '가구주_은퇴_적정생활비충당여부_3.0',
 '가구주_은퇴_적정생활비충당여부_4.0',
 '가구주_은퇴_적정생활비충당여부_5.0',]

# 코드북 
codebook = {
    '가구주연령_10세단위코드':   {
                        '30세 미만': 0,
                        '30~40세 미만': 1,
                        '40~50세 미만': 2,
                        '50~60세 미만': 3,
                        '60세이상': 4,
                    }, 
    '가구주_종사상지위(보도용)':  {
                        '상용근로자': 0,
                        '임시·일용근로자':1,
                        '자영업자': 2,
                        '그 외 또는 무직': 3
                        
                    },
    '가구주_성별코드': {'남성': 1, '여성': 2},
    '가구주_혼인상태코드': {
        '미혼': 1,
        '배우자있음': 2,
        '사별': 3,
        '이혼': 4
        },
    '가구주_은퇴여부': {'예': 2, '아니오': 1},
    '가구주_은퇴_적정생활비충당여부':  {
                    '아주 잘 되어 있다.': 1.0,
                    '잘 되어 있다.': 2.0,
                    '보통이다.': 3.0,
                    '잘 되어 있지 않다.': 4.0,
                    '전혀 되어 있지 않다.': 5.0,
                },
    '가구주_미은퇴_노후준비상황코드': {
                        '충분히 여유 있다': 1.0,
                        '여유 있다': 2.0,
                        '보통이다': 3.0,
                        '부족하다': 4.0,
                        '매우 부족하다': 5.0,
                    },
    '원리금연체여부': {'예': 1.0, '아니오': 2.0}
}
categorical_columns = ['가구주_성별코드', 
                       '가구주연령_10세단위코드', 
                       '가구주_종사상지위(보도용)', 
                       '가구주_혼인상태코드',
                       '가구주_은퇴여부', 
                       '원리금연체여부',  
                       '가구주_미은퇴_노후준비상황코드', 
                       '가구주_은퇴_적정생활비충당여부']


# encode_inputs의 결과 값(pred 값에 따른 이미지 출력)을 dialog 형태(모달창)로 출력하기
# @st.dialog : encode_inputs 함수 자체를 dialog 형태로 출력하도록 하는 구문.
@st.dialog("당신의 재무건강은?")

def encode_inputs(data):
    col = {}
    for d in data:
        value = codebook.get(d, {}).get(data[d]['value'], data[d]['value'])
        col[d] = [value]
        
    df = pd.DataFrame.from_dict(col, orient='index').T
    if df['가구주_은퇴여부'][0] == 2.0: 
        df['가구주_미은퇴_노후준비상황코드'] = None
    else:
        df['가구주_은퇴_적정생활비충당여부'] = None
    
    # cat/num 으로 나눈다
    # cat number type 정리
    categories = df[categorical_columns]
    numerics = df.drop(categorical_columns, axis=1)
    
    categories[intcols_cat] = categories[intcols_cat].astype(int)

    # cat을 one hot encoding
    categories = pd.get_dummies(categories, columns=categorical_columns, sparse=False )    
    categories = categories.reindex(columns=code_columns).fillna(False)
    
    # num의 누락 계산 
    numerics['월지출비'] = ((numerics['지출_소비지출비'] + numerics['지출_비소비지출(보완)'])/12).round(2)
    numerics['자산대비부채비율'] = (numerics['부채']/numerics['자산']).round(2)
    numerics['자산대비저축금액'] = (numerics['자산_금융자산_저축금액']/numerics['자산']).round(2)
    numerics['자산대비주식채권펀드'] = (numerics['자산_금융자산_저축_적립예치식저축_주식채권펀드금액']/numerics['자산']).round(2)
    
    numerics.fillna(numerics.mean(), inplace=True) 
    
    #합산
    full_features = pd.concat([numerics, categories], axis=1)


    gridsearch = pickle.load(open('data/trained_model.pkl', 'rb'))
    pred = gridsearch.predict(full_features)


    # pred 값에 따른 이미지 표시
    if pred in [0, 1, 2, 3, 4]:
        p= pred[0]
        image_path = f'data/img/{p}.png'  # 이미지 경로 설정
        st.image(Image.open(image_path))
    
    

# 설문조사 질문지 
def survey_display():
    st.write(''' <style>
         
        [data-testid="stNumberInputContainer"]::after {
            content: '만원';
            margin-left: 10px;
            padding-left: 10px;
            width: 100px;


        }
             
        [data-testid="stNumberInputContainer"]:has([aria-label="가구원수"])::after {
            content: '';
             width: 100%;
        }
         
         </style>
        ''', unsafe_allow_html=True)
    
    st.markdown('### ✔ 재무건강 상태 체크하기')
    st.markdown('> 아래의 설문조사를 완료하시고 재무건강 상태를 체크해보세요!')

    survey = ss.StreamlitSurvey()

    with survey.pages(3, progress_bar=True, on_submit=lambda : encode_inputs(survey.data)) as page:
        
        page.submit_button = page.default_btn_submit("완료")
        page.prev_button = page.default_btn_previous("이전")
        page.next_button = page.default_btn_next("다음")

        
        if page.current == 0: 
            st.markdown("""### 가구/ 가구주 정보 """)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(""" 가구주의 연령대는? """)
                survey.radio(
                    label="가구주연령_10세단위코드",
                    options=codebook['가구주연령_10세단위코드'],
                    index=0,
                    label_visibility='collapsed',
                    key='가구주연령_10세단위코드'
                )

                st.markdown( """ 가구원 수는 몇명인가요? """)
                survey.number_input(
                    label="가구원수",
                    label_visibility='collapsed',
                    min_value=1,
                    key="가구원수"
                )
                st.markdown(""" 귀하의 고용 형태를 선택해 주세요.""" )

                survey.selectbox(
                    label="가구주_종사상지위(보도용)",
                    index=0,
                    options=codebook['가구주_종사상지위(보도용)'],
                    label_visibility='collapsed',
                    key='가구주_종사상지위(보도용)'
                )


            with col2: 
                st.markdown(""" 가구주의 성별은? """)
                survey.radio(
                    label="가구주_성별코드",
                    index=0,
                    options=codebook['가구주_성별코드'],
                    label_visibility='collapsed',
                    key='가구주_성별코드'
                )

                st.markdown("""가구주의 혼인 여부를 알려주세요. """)
                
                survey.radio(
                    label="가구주_혼인상태코드",
                    index=0,
                    options=codebook['가구주_혼인상태코드'],
                    label_visibility='collapsed',
                    key='가구주_혼인상태코드'
                )

            

            st.markdown("""귀댁의 가구주(본인)는 은퇴를 하셨습니까?""")
            retired = survey.radio(
                    label="가구주_은퇴여부",
                    options=codebook['가구주_은퇴여부'],
                    label_visibility='collapsed',
                    key='가구주_은퇴여부',
                )
        
            if retired == '예':
                st.markdown("""최근 1년간 귀댁의 가구주와 배우자의 생활비 충당 정도는 어떻습니까?""")
                survey.radio(
                    label="가구주_은퇴_적정생활비충당여부",
                    index=0,
                    options=codebook['가구주_은퇴_적정생활비충당여부'],
                    label_visibility='collapsed',
                    key='가구주_은퇴_적정생활비충당여부'
                )
                
            elif retired == '아니오':
                st.markdown("""귀댁의 노후를 위한 준비 상황은 어떻습니까?""")
                survey.radio(
                    label="가구주_미은퇴_노후준비상황코드",
                    index=0,
                    options=codebook["가구주_미은퇴_노후준비상황코드"],
                    label_visibility='collapsed',
                    key='가구주_미은퇴_노후준비상황코드'
                )
    

        elif page.current == 1:   
            st.markdown("""### 자산 및 부채 정보 """)
            st.markdown("""귀하의 총 자산을 입력해주세요.(10,000₩. 일 만원 단위로 입력)  """)
            asset_amount = survey.number_input(
                label="자산",
                label_visibility='collapsed',
                key='자산',
                step=1,
                min_value=1,
                value=1
            )
            
            st.markdown("""보유 자산중 저축자산의 금액을 입력 해주세요.""" )
            st.caption("""
                저축 자산이란? 입출금 통장, 예금, 적금, 저축식 보험과 펀드, 주식, 채권을 아우르는 자산입니다.
            """, unsafe_allow_html=False,  help=None)

            survey.number_input(
                label="자산_금융자산_저축금액",
                label_visibility='collapsed',
                key='자산_금융자산_저축금액',
                step=1,
                min_value=0,
            )
            
            st.markdown(""" 저축 자산중 적립식 및 예치식 형태의 **펀드**의 금액을 입력해주세요. (만원 단위로 입력)""")
            survey.number_input(
                label="자산_금융자산_저축_적립예치식저축_주식채권펀드금액",
                min_value=0, 
                key='자산_금융자산_저축_적립예치식저축_주식채권펀드금액',
                step=1,
                label_visibility='collapsed'
            )
            st.markdown("""---""")
            st.markdown("""### 부채 """)
            st.markdown("""귀댁의 금융부채 규모를 입력해주세요.""")
            survey.number_input(
                        label="부채",
                        min_value=0, 
                        key='부채',
                        step=1,
                        label_visibility='collapsed'
                    )
            st.markdown("""지난 1년 간, 귀댁에서 원금을 상환하거나 이자 납부 일짜를 지나친 적이 있습니까? """)
            survey.radio(
                    label="원리금연체여부",
                    index=1,
                    options=codebook['원리금연체여부'],
                    label_visibility='collapsed',
                    key='원리금연체여부'
                )
        elif page.current == 2:
            st.markdown("""### 소득 및 지출 """)
            st.markdown("""귀댁의 세금공제 전 총 연간소득을 입력해주세요. (만원 단위)""")
            st.caption("""근로소득 + 사업소득 + 재산소득 + 기타소득을 합친 값을 일 만원(10,000) 단위로 입력해주세요.""")
            survey.number_input(
                        label="경상소득(보완)",
                        min_value=0, 
                        key='경상소득(보완)',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""귀댁의 연간 생활비로 소비하는 비용을 입력해주세요.""")
            st.caption("""주거비(월세, 주거관리비 등) + 식료품 및 외식비 + 통신, 교통비 + 교육비(보육료 포함) (만원 단위)""")
            survey.number_input(
                        label="지출_소비지출비",
                        min_value=0, 
                        key='지출_소비지출비',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""귀댁의 연간 세금, 공적연금 등 경상이전지출 총액을 입력해주세요.""")
            st.caption("""세금 + 연금 + 사회보험료 + 기타(기부금, 경조금 등) (만원 단위)""")
            survey.number_input(
                        label="지출_비소비지출(보완)",
                        min_value=0, 
                        key='지출_비소비지출(보완)',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""그 중, 연금 및 사회보험료의 총액을 입력해주세요""")
            survey.number_input(
                        label="지출_비소비지출_공적연금사회보험료(보완)",
                        min_value=0, 
                        key='지출_비소비지출_공적연금사회보험료(보완)',
                        step=1,
                        label_visibility='collapsed'
                    )
            
        st.markdown("""----- """)