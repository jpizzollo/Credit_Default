import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# Read in data:
data = pd.read_csv('raw_df.csv')
data = data.drop(columns='Unnamed: 0')
st.dataframe(data)

# Read in model
with open('rf_model','rb') as read_file:
    rf = pickle.load(read_file)

st.write('''## Default in light recent debt, filtering for age''')



age_input = st.slider('Age Filter', int(25), int(75), 25,5 )
age_filter = data['Age'] < age_input


groups = data[age_filter].groupby('Default')
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group['Debt5'], group['Debt4'],s=2, label=name)
ax.set_xlim([0,100000])
ax.set_ylim([0,100000])
plt.xlabel('Month 5 Debt')
plt.ylabel('Month 4 Debt')
plt.legend(labels=('no-default','default'))
st.pyplot(plt)





st.write('''## Random forest credit default prediction''')


sex = st.selectbox('Sex', ('Male', 'Female'))
education = st.selectbox('Education Level', ('High School', 'University', 'Graduate','Other'))
marriage = st.selectbox('Marital Status', ('Married','Single','Other'))
age = st.slider('Age', int(25), int(75), 25,1 )
limit = st.slider('Credit Limit', int(1000), int(100000), 1000,1000 )
d1 = st.slider('Month 1 Debt', int(0), int(100000), 0,1000 )
d2 = st.slider('Month 2 Debt', int(0), int(100000), 0,1000 )
d3 = st.slider('Month 3 Debt', int(0), int(100000), 0,1000 )
d4 = st.slider('Month 4 Debt', int(0), int(100000), 0,1000 )
d5 = st.slider('Month 5 Debt', int(0), int(100000), 0,1000 )

# Process some categoricals
Sex_Female = 0
Sex_Male = 0
Education_Graduate = 0
Education_High_School = 0
Education_Others = 0
Education_University = 0
Marriage_Married = 0
Marriage_Others = 0
Marriage_Single = 0


if sex == 'Male':
	Sex_Male = 1
else:
	Sex_Female = 1

if education == 'Graduate':
	Education_Graduate = 1
elif education == 'High School':
	Education_High_School = 1
elif education == 'Other':
	Education_Others = 1
else:
	Education_University = 1

if marriage == 'Married':
	Marriage_Married = 1
elif marriage == 'Single':
	Marriage_Single = 1
else:
	Marriage_Others = 1

input_data = pd.DataFrame({\
'Limit': [limit],\
'Age': [age],\
'Debt1': [d1],\
'Debt2': [d2],\
'Debt3': [d3],\
'Debt4': [d4],\
'Debt5': [d5],\
'Sex_Female': [Sex_Female],\
'Sex_Male': [Sex_Male],\
'Education_Graduate': [Education_Graduate],\
'Education_High_School': [Education_High_School],\
'Education_Others': [Education_Others],\
'Education_University': [Education_University],\
'Marriage_Married': [Marriage_Married],\
'Marriage_Others': [Marriage_Others],\
'Marriage_Single': [Marriage_Single]})


pred = rf.predict_proba(input_data)[0,1]

st.write(f'Predicted probability of defalt: {pred:.2f}')
