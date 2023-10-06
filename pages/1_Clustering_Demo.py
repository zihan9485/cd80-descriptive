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

import umap
import hdbscan



def clustering_demo() -> None:
    df = pd.read_csv("data.csv")
    df.rename(columns={"Bankrupt?": "Risk"}, inplace=True)

    df["Risk"].replace({0: "Low", 1: "High"}, inplace=True)

    with st.sidebar:
        company_id = st.selectbox(
            'Select Company',
            tuple(range(df.shape[0])))

    st.title("Unsupervised Clustering")
    st.caption("Company: " + str(company_id))

    mapper = umap.UMAP().fit(df.drop(columns=["Risk"]))

    st.header("Overall Risk")
    st.caption("Reduced with umap")
    # fig, ax = plt.subplots()
    # fig = umap.plot.points(mapper, labels=df["Risk"]).figure
    # st.pyplot(fig)

    reducer = umap.UMAP(n_neighbors=30, n_components=2)
    embedding = reducer.fit_transform(df.drop(columns=["Risk"]))
    df["x"] = embedding[:, 0]
    df["y"] = embedding[:, 1]

    fig = px.scatter(df, x='x', y='y', color="Risk",
                    color_discrete_sequence=["red", "green"])
    fig.update_traces(marker=dict(size=2),
                    selector=dict(mode='markers'))
    fig.add_traces(
        px.scatter(pd.DataFrame(df.iloc[company_id]).T, x='x', y='y').update_traces(
            marker_size=20, marker_color="purple").data
    )
    st.plotly_chart(fig)

    st.header("Cluster Risk")
    st.caption("Clustered with HDBScan")
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(df.drop(columns=["Risk"]))
    df["cluster"] = clusterer.labels_
    company_cluster = df["cluster"][company_id]

    fig = px.scatter(df[df["cluster"] == company_cluster], x='x', y='y', color="Risk",
                    color_discrete_sequence=["red", "green"])
    fig.update_traces(marker=dict(size=4),
                    selector=dict(mode='markers'))
    fig.add_traces(
        px.scatter(pd.DataFrame(df.iloc[company_id]).T, x='x', y='y').update_traces(
            marker_size=20, marker_color="purple").data
    )
    st.plotly_chart(fig)

    st.subheader("Peer Benchmarking")
    df_cluster = df[df["cluster"] == company_cluster]
    df_cluster["Risk"].replace({"Low": 0, "High": 1}, inplace=True)
    col1, col2 = st.columns(2)
    company_risk = df_cluster["Risk"].iloc[company_id].round(3)

    # Can come up with something better than mean
    cluster_risk = df_cluster["Risk"].mean().round(3)

    # Hardcoded delta, but can be retrieved with synthetic data (see descriptive_streamlit.py)
    col1.metric("Company Risk", company_risk, 1, delta_color='inverse')
    col2.metric('Cluster Risk', cluster_risk, -0.02, delta_color='inverse')



st.set_page_config(page_title="Clustering Demo", page_icon="📹")

clustering_demo()

show_code(clustering_demo)
