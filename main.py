import streamlit as st
import time
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import lightgbm as lgb



html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">Credit Card Fraud Detection</h1>
    <h3 style="color:white;text-align:center;">Data Mining Project</h3>
    <h3 style="color:white;text-align:center;">Enosh Nyarige</h3>
    <p style="color:white;text-align:center;">Jan-April, 2021</p>
	</div>
	"""


st.markdown(html_temp,unsafe_allow_html=True)


# Capture the amount and time of transaction
amount = st.number_input('Enter the transaction amount')
time_txn = st.time_input('Enter the transaction time')


# Load the model (keras)
# model_file = "models/fraud_model.h5"
# bilstm_model = load_model(model_file, compile= True)

model = lgb.Booster(model_file='baseline.model')


with st.beta_expander('Add V_values'):
    # Capture the V_values

    V_values = {

    }
    i = 1
    count = 29
    while i <= count:
        V_value = st.number_input('What is the V_'+ str(i) +' value?')
        st.json({'V_'+str(i): V_value})
        i += 1


# with st.spinner(text='Analyzing the provided dataset'):
#     time.sleep(5)

#     st.balloons()
#     st.success('The results are ready for viewing')

#     time.sleep(3)


#     my_dataframe = pd.read_csv('dataset/creditcard.csv', nrows=20)
#     st.dataframe(my_dataframe)
#     # st.table(data.iloc[0:10])


#     d_copy = my_dataframe.copy()
#     d_copy.drop(d_copy.iloc[:, 1:-2], inplace = True, axis = 1)
#     st.dataframe(d_copy)

#     st.area_chart(d_copy)
#     st.bar_chart(d_copy)
#     st.altair_chart(d_copy)


# y_pred = model.predict(pad_sequences)

# y_pred = np.argmax(y_pred, axis=1)

if st.button("Predict This Transaction"):
    st.write('Normal transaction')
# if st.button("Predict This Transaction"):
    # if y_pred[0] == 0:
    #     st.write("Fraud")
    # elif y_pred[0] == 1:
    #     st.write("Normal")