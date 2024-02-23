import streamlit as st
import pandas as pd
from utils import load_transformation_pipeline, load_model

st.set_page_config(
    page_title="Acme Loan Default Risk | Model Interface",
    page_icon="👋",
)

st.header("Model Interface")

st.text("Input the data to get a result:")

ordered_columns = ["person_age","person_emp_length",
                   "loan_amnt","loan_int_rate","loan_percent_income",
                   "cb_person_cred_hist_length","person_home_ownership_OTHER",
                   "person_home_ownership_OWN","person_home_ownership_RENT",
                   "loan_intent_EDUCATION","loan_intent_HOMEIMPROVEMENT",
                   "loan_intent_MEDICAL","loan_intent_PERSONAL","loan_intent_VENTURE",
                   "loan_grade_B","loan_grade_C","loan_grade_D","loan_grade_E",
                   "loan_grade_F","loan_grade_G","cb_person_default_on_file_Y",
                   "income_group_low-middle","income_group_middle","income_group_high"]

def get_loan_grade_fields(loan_grade):
    fields_values = {'B': [0], 'C': [0], 'D': [0], 'E': [0], 'F': [0], 'G':[0]}  # Initialize with zeros
    if loan_grade is not None:
        if loan_grade in fields_values:
            fields_values[loan_grade] = [1] 
    else:
        fields_values = {'B':[None], 'C':[None], 'D':[None], 'E':[None], 'F':[None], 'G':[None]}
    result = dict()
    for key in fields_values.keys():
        result[f'loan_grade_{key}'] = fields_values[key]
    return result

def get_person_home_ownership_fields(home_ownership_status):
    field_value = {'RENT':[0], "OWN":[0], "OTHER":[0]}
    if home_ownership_status is not None:
        if home_ownership_status in field_value:
            field_value[home_ownership_status]=[1]
    else:
        field_value = {'person_home_ownership_RENT':[None], "person_home_ownership_OWN":[None], "person_home_ownership_OTHER":[None]}
    result = dict()
    for key in field_value.keys():
        result[f'person_home_ownership_{key}'] = field_value[key]
    return result

def get_loan_intent_fields(loan_intent_value):
    fields_values = {"EDUCATION":[0],"MEDICAL":[0],"PERSONAL":[0], "HOMEIMPROVEMENT":[0],"VENTURE":[0]}
    if loan_intent_value is not None:
        if loan_intent_value in fields_values:
            fields_values[loan_intent_value]=[1]
    else:
        fields_values = {"MEDICAL":[None], "EDUCATION":[None], "PERSONAL":[None], "HOMEIMPROVEMENT":[None],"VENTURE":[None]}
    
    result = dict()
    for key in fields_values.keys():
        result[f'loan_intent_{key}'] = fields_values[key]
    return result

def default_history_field(default_history):
    if default_history is not None:
        if default_history == 'Y':
            return {"cb_person_default_on_file_Y": [1]}
        return {"cb_person_default_on_file_Y": [0]}
    else:
        return {"cb_person_default_on_file_Y": [None]}


def income_group_field(income):
    if income is None:
        return {"income_group_high": [None], "income_group_low-middle":[None],"income_group_middle":[None]}
    result = {"income_group_high": [0], "income_group_low-middle":[0],"income_group_middle":[0]}
    if income <= int(float("3.854200e+04")):
        return  result
    if income >int(float("3.854200e+04")) and income <= int(float("5.500000e+04")):
        result["income_group_low-middle"] = [1]
    if income >int(float("5.500000e+04")) and income <= int(float("7.921800e+04")):
        result['income_group_middle']=[1]
    else:
        result["income_group_high"] = [1]
    return result
    

with st.form("entry"):
    client_age = st.slider("How old is the client?",0, 130, None, key="person_age")

    client_income = st.number_input("What is the client's annual income",value=None, placeholder="Type a number...", key="person_income")
    client_empl_len = st.number_input("Insert the client's employment length (in years)", value=None, placeholder="Type a number...", key="person_emp_length")
    loan_amt = st.number_input("Insert the loan amount", value=None, placeholder="Type a number...", key="loan_amnt")
    loan_percent_income= st.number_input("Insert the percentage of the salary going to the loan", value=None, placeholder="Type a number...", key="loan_per_sal")
    loan_int_rate = st.number_input("Insert the loan interest rate", value=None, placeholder="Type a number...", key="loan_int_rate")
    client_credit_history_lengh = st.number_input("Insert the client's Credit history length", value=None, placeholder="Type a number...", key="cb_person_cred_hist_length")
    client_home_ownership_options = st.selectbox(
        "What is the client Home ownership status?",
        ("Rent", "Mortgage", "Own", "Other"),
        index=None,
        placeholder="Select an option...",
        )

    client_loan_intent = st.selectbox(
        "What is the client intent for this loan?",
        ("Medical", "Debt Consolidation", "Personal", "Home Improvement","Venture"),
        index=None,
        placeholder="Select an option...",
        )
    client_loan_grade = st.selectbox(
        "What is the loan grade?",
        ("A", "B", "C", "D", "E", "F","G"),
        index=None,
        placeholder="Select an option...",
        )
    client_default_history = st.selectbox(
        "What is the loan grade?",
        ("Yes", "No"),
        index=None,
        placeholder="Select an option...",
        )
    
    loan_threshold = st.slider("What is the probability threshold for loan approval ?",0.5, 0.99, 0.5, key="threshold")
    
    submited = st.form_submit_button("Predict")

    
    if submited:
        
        data = {"person_age": [client_age],
                #    "person_income":[client_income],
                #    "person_home_ownership":[str(client_home_ownership_options.capitalize) if client_home_ownership_options is not None else None],
                   "person_emp_length":[client_empl_len],
                #    "loan_intent":[str(client_loan_intent.capitalize).replace(" ","") if client_loan_intent is not None else None],
                #    "loan_grade":[str(client_loan_grade)],
                   "loan_amnt":[loan_amt],
                   "loan_int_rate":[loan_int_rate],
                   "loan_percent_income":[loan_percent_income],
                #    "cb_person_default_on_file":[str(client_default_history.capitalize)[0] if client_default_history is not None else None],
                   "cb_person_cred_hist_length": [client_credit_history_lengh]
        }

        data = data | get_loan_grade_fields(str(client_loan_grade))\
            | get_person_home_ownership_fields(str(client_home_ownership_options.capitalize) if client_home_ownership_options is not None else None)|\
            get_loan_intent_fields(str(client_loan_intent.capitalize).replace(" ","") if client_loan_intent is not None else None)|\
            default_history_field(str(client_default_history.capitalize)[0] if client_default_history is not None else None)|\
            income_group_field(client_income)
        
        sample = pd.DataFrame(data)[ordered_columns]

        scalled_sample = load_transformation_pipeline().transform(sample)
        
        st.dataframe(sample)

        st.write("Scalled sample")
        st.dataframe(scalled_sample)

        models  = load_model()
        if models is not None:
            model,_,__ = models
            
            prediction = model.predict_proba(sample)
            if prediction[0][1] >= loan_threshold:
                st.write("Based on the model, this user will likely default on his loan")
                st.write(f"With a probability {prediction[0][1]}")
            else:
                st.write("Based on the model, this user will likely reimburse his loan")
                st.write(f"With a probability {prediction[0][0]}")



