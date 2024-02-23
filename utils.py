import pandas as pd
import plotly.express as px
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
from lightgbm import LGBMClassifier
import os
import pickle


def load_dataset()->pd.DataFrame:
    data = pd.read_csv('credit_risk_dataset.csv')
    return data.copy()

def load_improved_dataset()->pd.DataFrame:
    if check_improved_dataset:
        try:
            data = pd.read_csv('improved_credit_risk_dataset.csv')
            return data.copy()
        except:
            pass

def check_improved_dataset()->bool:
    return os.path.isfile('improved_credit_risk_dataset.csv')

def save_dataset(dataset:pd.DataFrame):
    dataset.to_csv('improved_credit_risk_dataset.csv', index=False)

def get_train_test(df:pd.DataFrame, y_col:str):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[y_col]):
        df_train= df.iloc[train_index].copy()
        df_test = df.iloc[test_index].copy()
    return df_train, df_test

def save_model(model:LGBMClassifier, df_train:pd.DataFrame, df_test:pd.DataFrame):
    pickle.dump(model, open('final_model.pkl', 'wb'))
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)

def load_model():
    try:
        model = pickle.load(open('final_model.pkl', 'rb'))
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
        return model, train_df, test_df
    except:
        pass

def load_transformation_pipeline():
    try:
        return pickle.load(open('transformer_pipeline.pkl','rb'))
    except:
        pass


def train_and_evaluate_model(model:LGBMClassifier, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('scaler', StandardScaler())])
    transformed_X_train = pipeline.fit_transform(X_train)
    transformed_X_test = pipeline.transform(X_test)
    model.fit(transformed_X_train,y_train)
    y_pred = model.predict(transformed_X_test)
    class_report = classification_report(y_test,y_pred)
    figs = []
    conf_mat_fig, conf_mat_ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred, ax=conf_mat_ax)
    figs.append(conf_mat_fig)

    prec_rec_fig, prec_rec_ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test,y_pred, ax= prec_rec_ax)
    figs.append(prec_rec_fig)

    roc_fig, roc_ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test,y_pred, ax= roc_ax)
    figs.append(roc_fig)

    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,average='macro')
    recall = recall_score(y_test,y_pred,average='macro')
    f1 = f1_score(y_test,y_pred,average='macro')
    roc_auc = roc_auc_score(y_test,y_pred,average='macro')

    pickle.dump(pipeline, open('transformer_pipeline.pkl','wb'))    

    # del acc, precision, recall, f1, roc_auc
    # gc.collect()
    return model, acc, precision, recall, f1, roc_auc,class_report ,figs
