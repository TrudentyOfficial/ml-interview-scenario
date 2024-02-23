import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_improved_dataset, check_improved_dataset, get_train_test, train_and_evaluate_model

import warnings, gc
warnings.filterwarnings("ignore")
import re
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = load_improved_dataset()
models = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

def train_model(df_train, df_test, model):
    X_train, y_train = df_train.drop(columns=['loan_status']), df_train['loan_status']
    X_test, y_test = df_test.drop(columns=['loan_status']), df_test['loan_status']

    st.write(f"For model {str(model)}")
    st.markdown(f" - X train: {X_train.shape}")     
    st.markdown(f"- y train: {y_train.shape}")
    st.markdown(f"- X test: {X_test.shape}")
    st.markdown(f"- y test:  {y_test.shape}")
    model , acc, precision, recall, f1, roc_auc, _, __ = train_and_evaluate_model(model, X_train=X_train, X_test=X_test, y_test=y_test, y_train=y_train)
    models.append(str(model).strip())
    accuracy_scores.append(acc)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)

    del acc, precision, recall, f1, roc_auc, _, __
    gc.collect()

st.set_page_config(
    page_title="Acme Loan Default Risk | Model Achitecture Selection",
    page_icon="👋",
)

st.header("Model Achitecture Selection")

if not check_improved_dataset or dataset is None:
    st.error("The feature engineering was not completed.\n\
              Please return to feature engineering page and press the button")

else:
    df_train, df_test = get_train_test(dataset,y_col='loan_status')

    for model in [LGBMClassifier(verbose=-1),RandomForestClassifier(), DecisionTreeClassifier()]:
        train = df_train
        test = df_test
        if not re.search('LGBMClassifier', str(model)): ## Lighgbm works well with missing values
            train = df_train.dropna()
            test = df_test.dropna()
        train_model(df_train=train, df_test=train, model=model)
    
    model_perfs = pd.DataFrame({'Model': models, 
                            'Accuracy': accuracy_scores, 
                            'Precision': precision_scores,
                            'Recall': recall_scores,
                            'F1': f1_scores,
                            'ROC-AUC': roc_auc_scores}).sort_values('Accuracy',ascending=False).reset_index(drop=True)
    st.dataframe(model_perfs)

    st.write("Out of the 3 architectures only LGBMClassifier is not overfitting")
    st.write("So the goal will be to improve the hyperparameters to have a better LGBMClassifier")

