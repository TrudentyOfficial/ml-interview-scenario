import streamlit as st
import pandas as pd
import plotly.express as px
from utils import load_dataset, save_dataset, check_improved_dataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score,\
      roc_auc_score, accuracy_score, ConfusionMatrixDisplay, RocCurveDisplay,\
          PrecisionRecallDisplay
import warnings, gc
warnings.filterwarnings("ignore")
import re
from yellowbrick.classifier import ClassPredictionError
from matplotlib import pyplot as plt

dataset = load_dataset().drop_duplicates()

st.subheader("Feature set improvement")

st.markdown("""
            At this stage, with the current data we can:\n\
             - Split the people into income groups based on person_income\
             distribution below the 25th percentile, from 25th percentil through the median, above the median and above the 75th percentile: low income, low-middle income, middle, and high\n
             - Convert categorical columns into dummy columns 
             - Remove duplicates 
            """)

st.write("Based on the person_income distribution we have the following categories: ")

df_distribution = dataset['person_income'].describe()

dataset['income_group'] = pd.cut(dataset['person_income'],
                              bins=[0, df_distribution['25%'], df_distribution['50%'], df_distribution['75%'], float('inf')],
                              labels=['low', 'low-middle', 'middle', 'high']).astype('category')

dfx = dataset.drop(columns=['person_income']).copy()

fig = px.histogram(dataset['income_group'])
st.plotly_chart(fig)


categorical_columns = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file','income_group']

dfx = pd.get_dummies(dfx, columns=categorical_columns, drop_first=True,dtype=int)
st.dataframe(dfx.drop(columns=[col for col in dataset.columns if col in dfx.columns]))

if st.button('Save improv. dataset' if check_improved_dataset else 'Update improv. dataset'):
    save_dataset(dfx)