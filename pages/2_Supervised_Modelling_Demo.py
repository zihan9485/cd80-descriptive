# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import streamlit as st
from streamlit.hello.utils import show_code
from streamlit.logger import get_logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

import optuna


def supervised_modelling_demo() -> None:
    df = pd.read_csv("data.csv")
    df.rename(columns={"Bankrupt?": "Risk"}, inplace=True)

    with st.sidebar:
        company_id = st.selectbox(
            'Select Company',
            tuple(range(df.shape[0])))

    st.title("Modelling")
    st.caption("Company: " + str(company_id))

    with st.spinner("Training model"):
        X = df.drop(columns=['Risk'])
        y = df['Risk']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        scaler = StandardScaler()

        X_train = pd.DataFrame(scaler.fit_transform(
            X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5,
                                        max_depth=5, random_state=0
                                        ).fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

    # Hyperparameter optimisation takes very long
    # def objective(trial):

    #     n_estimators = trial.suggest_int("n_estimators", 10, 500)
    #     learning_rate = trial.suggest_int("learning_rate", 0.01, 0.5)
    #     max_depth = trial.suggest_int("max_depth", 2, 10)
    #     clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
    #                                      max_depth=max_depth, random_state=0)

    #     score = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    #     f1 = score.mean()
    #     return f1


    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=20)
    # st.write(study.best_trial)

    with st.expander("Model test metrics"):
        st.metric("Balanced accuracy",
                balanced_accuracy_score(y_test, y_pred))
        st.metric("F1 score", f1_score(y_test, y_pred))

    company_data = pd.DataFrame(df.drop(columns=["Risk"]).iloc[company_id]).T
    st.metric("Company predicted risk", clf.predict(company_data))

st.set_page_config(page_title="Supervised Modelling Demo", page_icon="📹")

supervised_modelling_demo()

show_code(supervised_modelling_demo)
