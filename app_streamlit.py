import streamlit as st
import pickle
import os
import numpy as np
import yaml
import joblib
 
# loading the trained model
params_path = 'params.yaml'
 
@st.cache()

class NotANumber(Exception):
    def __init__(self, message = "Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def read_params(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path)
    model_dir_path = config['model_webapp_dir']
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction   

def validate_input(dict_request):
    for _, val in dict_request.items():
        try:
            val = float(val)
        except Exception as e:
            raise NotANumber

    return True

def form_response(dict_request):
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [list(map(float, data))]
            response = predict(data)
            return response
    except NotANumber as e:
        response = str(e)
        return response



# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Churn Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    vmail_msg = st.slider('Number vmail messages',min_value = 1, max_value = 30)
    tot_day_calls = st.slider('Total day calls',min_value = 1, max_value = 30)
    tot_eve_min = st.slider('Total eve minutes',min_value = 1, max_value = 30) 
    tot_eve_chr = st.slider('Total eve charge',min_value = 1, max_value = 30)
    tot_int_min = st.slider('Total Intl minutes',min_value = 1, max_value = 30)
    cust_sev_calls = st.slider('Customer service calls',min_value = 1, max_value = 30)
    dict_request = {'vmail_msg': vmail_msg, 
                'tot_day_calls': tot_day_calls, 
                'tot_eve_min': tot_eve_min, 
                'tot_eve_chr': tot_eve_chr, 
                'tot_int_min': tot_int_min, 
                'cust_sev_calls': cust_sev_calls}
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result =  form_response(dict_request)
        st.success('Prediction is {}'.format(result))
     
if __name__=='__main__': 
    main()
