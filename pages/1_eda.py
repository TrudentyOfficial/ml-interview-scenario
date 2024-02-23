import streamlit as st
import pandas as pd
import io
from plotly import express as px
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import altair as alt
from vega_datasets import data
from sklearn.neighbors import  LocalOutlierFactor
from utils import load_dataset


def getDataframeInfo(df:pd.DataFrame)->pd.DataFrame:
    buf = io.StringIO()
    df.info(buf=buf)
    lines = buf.getvalue().splitlines()
    return (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split())
       .drop('Count',axis=1)
       .rename(columns={'Non-Null':'Non-Null Count'}))

def grab_col_names(dataframe:pd.DataFrame, target_col:str, cat_th=10):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.drop(columns=[target_col]).columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.drop(columns=[target_col]).columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_cols = cat_cols + num_but_cat

    num_cols = [col for col in dataframe.drop(columns=[target_col]).columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    st.write(f"Unique Observations: {dataframe.drop_duplicates().shape[0]}")
    st.write(f"Variables: {dataframe.drop(columns=[target_col]).shape[1]}")
    st.write(f"Target Variable: {dataframe[[target_col]].shape[1]}")

    st.write(f'Categorical columns: {len(cat_cols)}')
    st.write(f'Numerical columns: {len(num_cols)}')
    return cat_cols, num_cols

def high_correlated_cols(dataframe):
    # Select only the numeric columns from the DataFrame
    numeric_dataframe = dataframe.select_dtypes(include=['number'])
    
    corr = numeric_dataframe.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot= True, fmt='.2f', mask=mask)
    plt.title('Confusion Matrix')
    return fig
    

def get_target_distribution(dataset:pd.DataFrame):
    temp=dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), 
                           height=500, width=1000))
    target=dataset.loan_status.value_counts(normalize=True)
    target.rename(index={1:'Loan Default',0:'No Default'},inplace=True)
    pal, color=['#016CC9','#DEB078'], ['#8DBAE2','#EDD3B3']
    fig=go.Figure()
    fig.add_trace(go.Pie(labels=target.index, values=target*100, hole=.45, 
                        showlegend=True,sort=False, 
                        marker=dict(colors=color,line=dict(color=pal,width=2.5)),
                        hovertemplate = "%{label} Accounts: %{value:.2f}%<extra></extra>"))
    fig.update_layout(template=temp, title='Target Distribution', 
                    legend=dict(traceorder='reversed',y=1.05,x=0),
                    uniformtext_minsize=15, uniformtext_mode='hide',width=700)
    return fig


def get_numeric_distribution(data:pd.DataFrame, target_column:str):
    names = data.columns.drop(target_column)
    figs =[]
    fig1, axes = plt.subplots(1,2, squeeze=False)
    sns.boxplot(y=names[0], x= target_column, data=data, orient='v', ax=axes[0,0], hue='loan_status')
    sns.boxplot(y=names[1], x= target_column, data=data, orient='v', ax=axes[0,1], hue='loan_status')
    figs.append(fig1)

    fig2, axes2 = plt.subplots(1,2, squeeze=False)
    sns.boxplot(y=names[2], x= target_column, data=data, orient='v', ax=axes2[0,0], hue='loan_status')
    sns.boxplot(y=names[3], x= target_column, data=data, orient='v', ax=axes2[0,1], hue='loan_status')
    figs.append(fig2)

    fig3, axes = plt.subplots(1,2, squeeze=False)
    sns.boxplot(y=names[4], x= target_column, data=data, orient='v', ax=axes[0,0], hue='loan_status')
    sns.boxplot(y=names[5], x= target_column, data=data, orient='v', ax=axes[0,1], hue='loan_status')
    figs.append(fig3)
    
    fig4, axes = plt.subplots(1,1, squeeze=False)
    sns.boxplot(y=names[6], x= target_column, data=data, orient='v', ax=axes[0,0], hue='loan_status')
    figs.append(fig4)

    return figs

def get_categorical_feature_distribution(dataset:pd.DataFrame, target_column:str):
    figs = []

    for col in dataset.columns:
        trace0 = go.Bar(
        x = dataset[dataset["loan_status"]== 1][col].value_counts().index.values,
        y = dataset[dataset["loan_status"]== 1][col].value_counts().values,
        name='Loan status = 1'
        )

        #Second plot
        trace1 = go.Bar(
            x = dataset[dataset["loan_status"]== 0][col].value_counts().index.values,
            y = dataset[dataset["loan_status"]== 0][col].value_counts().values,
            name="Loan status = 0"
        )

        data = [trace0, trace1]

        layout = go.Layout(
            title=f'{col} Distribuition'
        )


        fig = go.Figure(data=data, layout=layout)
        figs.append(fig)

    return figs



st.set_page_config(
    page_title="Acme Loan Default Risk | EDA",
    page_icon="👋",
)

st.write("# Exploratory Data Analysis")


st.markdown("### The dataset")

dataset = load_dataset()

st.dataframe(dataset)

st.write("Detailed data description of Credit Risk dataset:")

st.markdown(""" 
| Feature Name               | Description                     |
|----------------------------|---------------------------------|
| person_age                 | Age                             |
| person_income              | Annual Income                   |
| person_home_ownership      | Home ownership                  |
| person_emp_length          | Employment length (in years)    |
| loan_intent                | Loan intent                     |
| loan_grade                 | Loan grade                      |
| loan_amnt                  | Loan amount                     |
| loan_int_rate              | Interest rate                   |
| loan_status                | Loan status (0 is non default 1 is default) |
| loan_percent_income        | Percent income                  |
| cb_person_default_on_file  | Historical default              |
| cb_preson_cred_hist_length | Credit history length           |
""")

st.write(f"Dataset size: {len(dataset)}" )
st.dataframe(getDataframeInfo(dataset))

st.write("Null Values:")
st.write(dataset.isnull().sum())

cat_cols, num_cols = grab_col_names(dataframe=dataset, target_col='loan_status')

dataset_no_dup = dataset.drop_duplicates().copy()

st.subheader("Data distribution")

st.write("Numerical Data")
st.dataframe(dataset_no_dup.describe())

# fig = px.histogram(dataset_no_dup[num_cols])
# st.plotly_chart(fig)

fig, axs = plt.subplots(4, 2,squeeze=False)
axs[0,0].hist(dataset_no_dup[num_cols[0]],label= num_cols[0])
axs[0,1].hist(dataset_no_dup[num_cols[1]],label= num_cols[1])
axs[1,0].hist(dataset_no_dup[num_cols[2]],label= num_cols[2])
axs[1,1].hist(dataset_no_dup[num_cols[3]],label= num_cols[3])
axs[2,0].hist(dataset_no_dup[num_cols[4]],label= num_cols[4])
axs[2,1].hist(dataset_no_dup[num_cols[5]],label= num_cols[5])
axs[3,0].hist(dataset_no_dup[num_cols[6]],label= num_cols[6])


# ax.hist(dataset_no_dup[num_cols], bins=20)

# st.pyplot(fig)

# fig_boc =  plt.boxplot(dataset_no_dup[cat_cols])
# st.pyplot(fig_boc)


st.write(high_correlated_cols(dataset_no_dup))

st.plotly_chart(get_target_distribution(dataset_no_dup))

st.subheader("Oulier Analysis")


figs = get_numeric_distribution(dataset_no_dup[num_cols+['loan_status']], target_column='loan_status')
for fig in figs:
    st.write(fig)

cat_figs = get_categorical_feature_distribution(dataset_no_dup[cat_cols+['loan_status']], target_column='loan_status')

for fig in cat_figs:
    st.write(fig)
