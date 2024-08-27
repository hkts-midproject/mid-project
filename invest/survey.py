import streamlit as st
import pandas as pd
import streamlit_survey as ss


codebook = {
    'Master_Age':   {
                        '30세 미만': 1,
                        '30~40세 미만': 2,
                        '40~50세 미만': 3,
                        '50~60세 미만': 4,
                        '60세이상': 5,
                    }, 
    'Master_laborCode':  {
                        '상용근로자': 1,
                        '임시·일용근로자': 2,
                        '고용원이 있는 자영업자': 3,
                        '고용원이 없는 자영업자': 4,
                        '무급가족종사자': 5,
                        '기타 종사자(실적급의 보험설계사, 대리 운전기사,학습지 방문 교사 등)': 6,
                        '기타(무직자, 가사, 학생 등)': 7,
                    },
    'Master_Gender': {'남성': 1, '여성': 2},
    'Master_MarriCode': {'예': 1, '아니오': 0},
    'Master_Retired': {'예': 1, '아니오': 0},
    'Master_RetirementSavingCode':  {
                    '아주 잘 되어 있다.': 1,
                    '잘 되어 있다.': 2,
                    '보통이다.': 3,
                    '잘 되어 있지 않다.': 4,
                    '전혀 되어 있지 않다.': 5,
                },
    'Master_RetirementReadyCode': {
                        '충분히 여유 있다': 1,
                        '여유 있다': 2,
                        '보통이다': 3,
                        '부족하다': 4,
                        '매우 부족하다': 5,
                    },
    'has_saving': {'예': 1, '아니오': 0},
    'debt_delayed': {'예': 1, '아니오': 0}
}
def encode_inputs(data):
    columns = {}
    for d in data:
        value = codebook.get(d, {}).get(data[d]['value'], data[d]['value'])
        columns[d] = value
        
    df = pd.DataFrame.from_dict(columns, orient='index').T
    st.write(df)
    st.markdown("""여기있는 RESULT DF를 가지고 예측모델을 돌린다. (DF편집 필요할 수 있다.)""")

    return df
    
def survey_display():
    st.write(''' <style>
         
        [data-testid="stNumberInputContainer"]::after {
            content: '만원';
            margin-left: 10px;
            padding-left: 10px;
            width: 100px;


        }
             
        [data-testid="stNumberInputContainer"]:has([aria-label="Family_num"])::after {
            content: '';
             width: 100%;
        }
         
         </style>
        ''', unsafe_allow_html=True)
    
    st.markdown('### 고객 투자성향 분석 예측')

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
                    label="Master_Age",
                    options=codebook['Master_Age'],
                    index=0,
                    label_visibility='collapsed',
                    key='Master_Age'
                )

                st.markdown( """ 가구원 수는 몇명인가요? """)
                survey.number_input(
                    label="Family_num",
                    label_visibility='collapsed',
                    min_value=1,
                    key="Family_num"
                )
                st.markdown(""" 귀하의 고용 형태를 선택해 주세요.""" )

                survey.selectbox(
                    label="Master_laborCode",
                    index=0,
                    options=codebook['Master_laborCode'],
                    label_visibility='collapsed',
                    key='Master_laborCode'
                )


            with col2: 
                st.markdown(""" 가구주의 성별은? """)
                survey.radio(
                    label="Master_Gender",
                    index=0,
                    options=codebook['Master_Gender'],
                    label_visibility='collapsed',
                    key='Master_Gender'
                )

                st.markdown("""가구주의 혼인 여부를 알려주세요. """)
                
                survey.radio(
                    label="Master_MarriCode",
                    index=0,
                    options=codebook['Master_MarriCode'],
                    label_visibility='collapsed',
                    key='Master_MarriCode'
                )

            def changed():
                if 'Master_RetirementSavingCode' in survey.data:
                    del survey.data['Master_RetirementSavingCode']
                if 'Master_RetirementReadyCode' in survey.data:
                    del survey.data['Master_RetirementReadyCode']

            st.markdown("""귀댁의 가구주(본인)는 은퇴를 하셨습니까?""")
            retired = survey.radio(
                    label="Master_Retired",
                    options=codebook['Master_Retired'],
                    label_visibility='collapsed',
                    key='Master_Retired',
                    on_change=changed
                )
        
            if retired == '예':
                st.markdown("""최근 1년간 귀댁의 가구주와 배우자의 생활비 충당 정도는 어떻습니까?""")
                survey.radio(
                    label="Master_RetirementSavingCode",
                    index=0,
                    options=codebook['Master_RetirementSavingCode'],
                    label_visibility='collapsed',
                    key='Master_RetirementSavingCode'
                )
            elif retired == '아니오':
                st.markdown("""귀댁의 노후를 위한 준비 상황은 어떻습니까?""")
                survey.radio(
                    label="Master_RetirementReadyCode",
                    index=0,
                    options=codebook["Master_RetirementReadyCode"],
                    label_visibility='collapsed',
                    key='Master_RetirementReadyCode'
                )
        elif page.current == 1:   
            st.markdown("""### 자산 및 부채 정보 """)
            st.markdown("""귀하의 총 자산을 입력해주세요.(10,000₩. 일 만원 단위로 입력)  """)
            asset_amount = survey.number_input(
                label="asset",
                label_visibility='collapsed',
                key='asset',
                step=1,
                min_value=0,
            )
            if asset_amount > 0 :
                st.markdown("""저축 자산을 보유하고 계십니까?""" )
                st.caption("""
                    저축 자산이란? 입출금 통장, 예금, 적금, 저축식 보험과 펀드, 주식, 채권을 아우르는 자산입니다.
                """, unsafe_allow_html=False,  help=None)


                saving_yn = survey.radio(
                    label="has_saving",
                    index=0,
                    options=codebook['has_saving'],
                    label_visibility='collapsed',
                    key='has_saving'
                )
                if saving_yn == '예':
                    st.markdown(""" 저축 자산중 적립식 및 예치식 형태의 **펀드**의 금액을 입력해주세요. (만원 단위로 입력)""")
                    survey.number_input(
                        label="saving_fund_amount",
                        min_value=0, 
                        key='saving_fund_amount',
                        step=1,
                        label_visibility='collapsed'
                    )
            st.markdown("""---""")
            st.markdown("""### 부채 """)
            st.markdown("""귀댁의 금융부채 규모를 입력해주세요.""")
            survey.number_input(
                        label="liabilities_amount",
                        min_value=0, 
                        key='liabilities_amount',
                        step=1,
                        label_visibility='collapsed'
                    )
            st.markdown("""지난 1년 간, 귀댁에서 원금을 상환하거나 이자 납부 일짜를 지나친 적이 있습니까? """)
            survey.radio(
                    label="debt_delayed",
                    index=1,
                    options=codebook['debt_delayed'],
                    label_visibility='collapsed',
                    key='debt_delayed'
                )
        elif page.current == 2:
            st.markdown("""### 소득 및 지출 """)
            st.markdown("""귀댁의 세금공제 전 총 연간소득을 입력해주세요. (만원 단위)""")
            st.caption("""근로소득 + 사업소득 + 재산소득 + 기타소득을 합친 값을 일 만원(10,000) 단위로 입력해주세요.""")
            survey.number_input(
                        label="income_amount",
                        min_value=0, 
                        key='income_amount',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""귀댁의 연간 생활비로 소비하는 비용을 입력해주세요.""")
            st.caption("""주거비(월세, 주거관리비 등) + 식료품 및 외식비 + 통신, 교통비 + 교육비(보육료 포함) (만원 단위)""")
            survey.number_input(
                        label="consumption_amount",
                        min_value=0, 
                        key='consumption_amount',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""귀댁의 연간 세금, 공적연금 등 경상이전지출 총액을 입력해주세요.""")
            st.caption("""세금 + 연금 + 사회보험료 + 기타(기부금, 경조금 등) (만원 단위)""")
            survey.number_input(
                        label="nonconsumption_amount",
                        min_value=0, 
                        key='nonconsumption_amount',
                        step=1,
                        label_visibility='collapsed'
                    )
            
            st.markdown("""그 중, 연금 및 사회보험료의 총액을 입력해주세요""")
            survey.number_input(
                        label="nonconsumption_pension",
                        min_value=0, 
                        key='nonconsumption_pension',
                        step=1,
                        label_visibility='collapsed'
                    )
            
        st.markdown("""----- """)