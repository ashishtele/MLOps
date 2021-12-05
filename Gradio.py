import gradio as gr
import pickle
import os
import numpy as np
import yaml
import joblib
 
# loading the trained model
params_path = 'params.yaml'
 
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

def form_response(vmail_msg, tot_day_calls, tot_eve_min, tot_eve_chr, tot_int_min, cust_sev_calls):
    dict_request = {'vmail_msg': vmail_msg, 
                'tot_day_calls': tot_day_calls, 
                'tot_eve_min': tot_eve_min, 
                'tot_eve_chr': tot_eve_chr, 
                'tot_int_min': tot_int_min, 
                'cust_sev_calls': cust_sev_calls}
    try:
        if validate_input(dict_request):
            data = dict_request.values()
            data = [list(map(float, data))]
            response = predict(data)
            return response
    except NotANumber as e:
        response = str(e)
        return response

# Slider creation for Gradio
vmail_msg = gr.inputs.Slider(label = 'Number vmail messages',minimum = 1, maximum = 30)
tot_day_calls = gr.inputs.Slider(label = 'Total day calls',minimum = 1, maximum = 30)
tot_eve_min = gr.inputs.Slider(label = 'Total eve minutes',minimum = 1, maximum = 30) 
tot_eve_chr = gr.inputs.Slider(label = 'Total eve charge',minimum = 1, maximum = 30)
tot_int_min = gr.inputs.Slider(label = 'Total Intl minutes',minimum = 1, maximum = 30)
cust_sev_calls = gr.inputs.Slider(label = 'Customer service calls',minimum = 1, maximum = 30)


gr.Interface(form_response,
            inputs = [vmail_msg, tot_day_calls, tot_eve_min, tot_eve_chr, tot_int_min, cust_sev_calls],
            outputs = 'label',
            live = True).launch()

    