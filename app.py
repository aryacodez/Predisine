import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter("ignore")


#DataFrame Generation

data = {'Symptom':['itching', 'skin rash', 'nodal skin eruptions',
       'continuous sneezing', 'shivering', 'chills', 'joint pain',
       'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
       'vomiting', 'burning micturition', 'spotting urination', 'fatigue',
       'weight gain', 'anxiety', 'cold hands and feets', 'mood swings',
       'weight loss', 'restlessness', 'lethargy', 'patches in throat',
       'irregular sugar level', 'cough', 'high fever', 'sunken eyes',
       'breathlessness', 'sweating', 'dehydration', 'indigestion',
       'headache', 'yellowish skin', 'dark urine', 'nausea',
       'loss of appetite', 'pain behind the eyes', 'back pain',
       'constipation', 'abdominal pain', 'diarrhoea', 'mild fever',
       'yellow urine', 'yellowing of eyes', 'acute liver failure',
       'fluid overload', 'swelling of stomach', 'swelled lymph nodes',
       'malaise', 'blurred and distorted vision', 'phlegm',
       'throat irritation', 'redness of eyes', 'sinus pressure',
       'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
       'fast heart rate', 'pain during bowel movements',
       'pain in anal region', 'bloody stool', 'irritation in anus',
       'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen legs', 'swollen blood vessels', 'puffy face and eyes',
       'enlarged thyroid', 'brittle nails', 'swollen extremeties',
       'excessive hunger', 'extra marital contacts',
       'drying and tingling lips', 'slurred speech', 'knee pain',
       'hip joint pain', 'muscle weakness', 'stiff neck',
       'swelling joints', 'movement stiffness', 'spinning movements',
       'loss of balance', 'unsteadiness', 'weakness of one body side',
       'loss of smell', 'bladder discomfort', 'foul smell ofurine',
       'continuous feel of urine', 'passage of gases', 'internal itching',
       'toxic look (typhos)', 'depression', 'irritability', 'muscle pain',
       'altered sensorium', 'red spots over body', 'belly pain',
       'abnormal menstruation', 'dischromic patches',
       'watering from eyes', 'increased appetite', 'polyuria',
       'family history', 'mucoid sputum', 'rusty sputum',
       'lack of concentration', 'visual disturbances',
       'receiving blood transfusion', 'receiving unsterile injections',
       'coma', 'stomach bleeding', 'distention of abdomen',
       'history of alcohol consumption', 'fluid overload',
       'blood in sputum', 'prominent veins on calf', 'palpitations',
       'painful walking', 'pus filled pimples', 'blackheads', 'scurring',
       'skin peeling', 'silver like dusting', 'small dents in nails',
       'inflammatory nails', 'blister', 'red sore around nose',
       'yellow crust ooze', 'prognosis'],'weight':[1, 3, 4, 4, 5, 3, 3, 5, 3, 4, 3, 5, 6, 6, 4, 3, 4, 5, 3, 3, 5, 2,
       6, 5, 4, 7, 3, 4, 3, 4, 5, 3, 3, 4, 5, 4, 4, 3, 4, 4, 6, 5, 4, 4,
       6, 6, 7, 6, 6, 5, 5, 4, 5, 4, 5, 5, 7, 7, 5, 5, 6, 5, 6, 5, 4, 4,
       4, 4, 5, 5, 5, 6, 5, 5, 4, 5, 4, 4, 3, 2, 2, 4, 5, 5, 6, 4, 4, 4,
       3, 4, 5, 6, 5, 4, 5, 3, 2, 2, 2, 3, 4, 6, 6, 4, 5, 4, 5, 4, 4, 3,
       3, 5, 2, 7, 6, 4, 5, 4, 5, 6, 4, 2, 2, 2, 2, 3, 2, 2, 2, 4, 2, 3,
       5]}
df_s = pd.DataFrame(data)
# *********************************************************************

# model work

model = pickle.load(open('RFC_symptoms.pkl','rb'))
def predict(s1,s2,s3,s4,s5='vomiting',s6='vomiting',s7='vomiting'):
    l = [s1,s2,s3,s4,s5,s6,s7]
    print(l)
    
    x= np.array(df_s['Symptom'])
    y= np.array(df_s['weight'])
    for i in range(len(l)):
        for j in range(len(x)):
            if l[i]==x[j]:
                l[i]=y[j]
    res = [l]
    pred = model.predict(res)
    return pred[0]

header1 = '<h2 style="text-align: center;font-size:65px;color:#002b80"><b>Predisine - Disease Detection System<b></h2>'
st.markdown(header1,unsafe_allow_html=True)
st.sidebar.header('Selector')
option = st.sidebar.selectbox('What Type',('Normal Disease','Contributor'))
if option =='Normal Disease':
    

    subheader = '<h4 style="text-align:center;font-size:18px;">Detects Probable Disease based on Superficial and Visible Symptoms Diagnosed.</h4>'
    st.markdown(subheader,unsafe_allow_html=True)
    
    x = ['itching', 'skin rash', 'nodal skin eruptions',
       'continuous sneezing', 'shivering', 'chills', 'joint pain',
       'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
       'vomiting', 'burning micturition', 'spotting urination', 'fatigue',
       'weight gain', 'anxiety', 'cold hands and feets', 'mood swings',
       'weight loss', 'restlessness', 'lethargy', 'patches in throat',
       'irregular sugar level', 'cough', 'high fever', 'sunken eyes',
       'breathlessness', 'sweating', 'dehydration', 'indigestion',
       'headache', 'yellowish skin', 'dark urine', 'nausea',
       'loss of appetite', 'pain behind the eyes', 'back pain',
       'constipation', 'abdominal pain', 'diarrhoea', 'mild fever',
       'yellow urine', 'yellowing of eyes', 'acute liver failure',
       'fluid overload', 'swelling of stomach', 'swelled lymph nodes',
       'malaise', 'blurred and distorted vision', 'phlegm',
       'throat irritation', 'redness of eyes', 'sinus pressure',
       'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
       'fast heart rate', 'pain during bowel movements',
       'pain in anal region', 'bloody stool', 'irritation in anus',
       'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity',
       'swollen legs', 'swollen blood vessels', 'puffy face and eyes',
       'enlarged thyroid', 'brittle nails', 'swollen extremeties',
       'excessive hunger', 'extra marital contacts',
       'drying and tingling lips', 'slurred speech', 'knee pain',
       'hip joint pain', 'muscle weakness', 'stiff neck',
       'swelling joints', 'movement stiffness', 'spinning movements',
       'loss of balance', 'unsteadiness', 'weakness of one body side',
       'loss of smell', 'bladder discomfort', 'foul smell ofurine',
       'continuous feel of urine', 'passage of gases', 'internal itching',
       'toxic look (typhos)', 'depression', 'irritability', 'muscle pain',
       'altered sensorium', 'red spots over body', 'belly pain',
       'abnormal menstruation', 'dischromic patches',
       'watering from eyes', 'increased appetite', 'polyuria',
       'family history', 'mucoid sputum', 'rusty sputum',
       'lack of concentration', 'visual disturbances',
       'receiving blood transfusion', 'receiving unsterile injections',
       'coma', 'stomach bleeding', 'distention of abdomen',
       'history of alcohol consumption', 'blood in sputum',
       'prominent veins on calf', 'palpitations', 'painful walking',
       'pus filled pimples', 'blackheads', 'scurring', 'skin peeling',
       'silver like dusting', 'small dents in nails',
       'inflammatory nails', 'blister', 'red sore around nose',
       'yellow crust ooze']
    
    fullName = st.text_input('Enter your Full Name',' ')    
    if fullName:
      st.write('Patient Name: ',fullName)
      multi = st.multiselect('Select the Symptoms ',[*x])
      if len(multi)<4:
        st.write('Enter atleast four Symptoms and maximum upto seven symptoms to check what kind of disease do you have!!!')
        st.write('Error will get removed once atleast four symptoms are entered')
      if len(multi)==4:
        res=predict(str(multi[0]),str(multi[1]),str(multi[2]),str(multi[3]))
      if len(multi)==5:
        res=predict(str(multi[0]),str(multi[1]),str(multi[2]),str(multi[3]),str(multi[4]))
      if len(multi)==6:
        res=predict(str(multi[0]),str(multi[1]),str(multi[2]),str(multi[3]),str(multi[4]),str(multi[5]))
      if len(multi)==7:
        res=predict(str(multi[0]),str(multi[1]),str(multi[2]),str(multi[3]),str(multi[4]),str(multi[5]),str(multi[6]))
      if st.button('Predict'):
        with st.spinner('Model Predicting....'):
          time.sleep(3)
        st.success('Done!!!')
        st.info('Disease Predicted is : {}'.format(res))
        st.snow()
        if st.button('Reset'):
          st.experimental_rerun()
if option == 'Contributor':
  st.title('Contributors')
  st.subheader('Name: Arya Sarkar')
  st.write("Github Profile: [Click Here](https://github.com/aryacodez)")
  st.subheader('Name: Surya Sarkar')
  st.write("Github Profile: [Click Here](https://github.com/Suryageeks)")
  st.subheader('Name: Faizan Anwar')
  st.write("Github Profile: [Click Here](https://github.com/FaizanAnwar2801)")
