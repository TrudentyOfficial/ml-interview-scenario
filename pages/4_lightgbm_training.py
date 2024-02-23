import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_improved_dataset, check_improved_dataset, get_train_test,\
      train_and_evaluate_model, load_model, save_model

import warnings, gc
warnings.filterwarnings("ignore")
import re
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from streamlit_shap import st_shap
import shap

shap.initjs()

dataset = load_improved_dataset()


def set_up_gbm_classifier()-> LGBMClassifier:
    params = {
    'application': 'binary', # for binary classification
    'boosting': 'gbdt', # traditional gradient boosting decision tree
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'device': 'cpu', # you can use GPU to achieve faster learning
    'max_depth': -1, # <0 means no limit
    'max_bin': 510, # Small number of bins may reduce training accuracy but can deal with over-fitting
    'lambda_l1': 5, # L1 regularization
    'lambda_l2': 10, # L2 regularization
    'metric' : 'binary_error',
    'subsample_for_bin': 200, # number of samples for constructing bins
    'subsample': 1, # subsample ratio of the training instance
    'colsample_bytree': 0.8, # subsample ratio of columns when constructing the tree
    'min_split_gain': 0.5, # minimum loss reduction required to make further partition on a leaf node of the tree
    'min_child_weight': 1, # minimum sum of instance weight (hessian) needed in a leaf
    'min_child_samples': 5# minimum number of data needed in a leaf
    }

    return LGBMClassifier(boosting_type= 'gbdt', 
          objective = 'regression', 
          n_jobs = 5, 
          silent = True,
          verbose= -1,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'], 
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'], 
          min_split_gain = params['min_split_gain'], 
          min_child_weight = params['min_child_weight'], 
          min_child_samples = params['min_child_samples'])

def get_grid()->dict:
    return  {
    'learning_rate': [0.005,0.01,0.05, 0.02, 0.1,],
    'n_estimators': [8,16,24],
    'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    'objective' : ['regression'],
    'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }


def train_model():
    df_train, df_test = get_train_test(dataset, 'loan_status')
    X_train, y_train = df_train.drop(columns=['loan_status']), df_train['loan_status']
    X_test, y_test = df_test.drop(columns=['loan_status']), df_test['loan_status']

    st.markdown(f" - X train: {X_train.shape}")     
    st.markdown(f"- y train: {y_train.shape}")
    st.markdown(f"- X test: {X_test.shape}")
    st.markdown(f"- y test:  {y_test.shape}")


    lgbm_model = LGBMClassifier()

    model , acc, precision, recall, f1, roc_auc, class_report, figs = train_and_evaluate_model(lgbm_model, X_train=X_train, X_test=X_test, y_test=y_test, y_train=y_train)
    save_model(model, df_train=df_train, df_test=df_test)
    
    st.subheader("Classification Report and Model Evaluation:")
    st.write(f"Model accuracy: {acc}")
    st.write(f"Model Precision: {precision}")
    st.write(f"Model recall: {recall}")
    st.write(f"Model f1: {f1}")
    for fig in figs:
        st.pyplot(fig)

    return model, df_train, df_test

st.set_page_config(
    page_title="Acme Loan Default Risk | Model Training",
    page_icon="👋",
)

st.header("Model Training")


if not check_improved_dataset or dataset is None:
    st.error("The feature engineering was not completed.\n\
              Please return to feature engineering page and press the button")

else:
    models = load_model()
    if st.button("Train model" if models is None else "Retrain Model"):
        models= train_model()
    
    if models is not None:
        model, df_train, df_test = models
        st.subheader("Using SHAP values to understand feature importances")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_train.drop(columns=['loan_status']))
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_train.drop(columns=['loan_status']).iloc[0, :], list(df_train.drop(columns=['loan_status']).columns)))
        st_shap(shap.summary_plot(shap_values, df_train.drop(columns=['loan_status'])))
