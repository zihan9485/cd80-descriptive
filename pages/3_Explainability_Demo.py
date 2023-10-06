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
import shap
import plotly.express as px
import plotly.graph_objs as go

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)



def explainability_demo() -> None:
    df = pd.read_csv("data.csv")
    df.rename(columns={"Bankrupt?": "Risk"}, inplace=True)

    with st.sidebar:
        company_id = st.selectbox(
            'Select Company',
            tuple(range(df.shape[0])))

    st.title("Model Explainability")
    st.caption("Company: " + str(company_id))

    X = df.drop("Risk", axis=1)  # Independent variables
    y = df["Risk"]  # Dependent variable

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("Risk Classification Model")
    st.write(clf)

    explainer = shap.Explainer(clf)

    shap_values = explainer.shap_values(X_test)

    st.subheader("Overall Feature Impact on Risk prediction")
    with st.expander("How to Interpret"):
        st.info("The SHAP value is the impact of that feature value on the model output")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test)
    st.pyplot(fig)

    st.subheader("Feature Impact on High Risk prediction")
    with st.expander("How to Interpret"):
        st.info("Blue Points -> Low Feature Values, Red Points -> High Feature Values")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], X_test)
    st.pyplot(fig)

    st.subheader("Company Specific Impact (Force Plot)")
    with st.expander("How to Interpret"):
        st.info("Red Features -> Higher Risk Output, Blue Features -> Lower Risk Output")

    st.pyplot(shap.plots.force(
        explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :], matplotlib=True))



st.set_page_config(page_title="Explainability Demo", page_icon="📹")

explainability_demo()

show_code(explainability_demo)
