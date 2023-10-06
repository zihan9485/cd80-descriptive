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

import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import random
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from statsmodels.tsa.seasonal import seasonal_decompose

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Chengdu80 Streamlit Demo",
        page_icon="👋",
    )

    st.write("# Welcome to Chengdu80 Streamlit Demo! 👋")

    st.sidebar.success("Select a demo module above.")

    st.markdown(
        """
        **👈 Select a demo from the sidebar**
        """
    )

    st.markdown(
        """
        **There are 4 modules:**
        1. Descriptive Analysis
        1. Clustering
        1. Supervised Modelling
        1. Model Explainability
        """
    )


if __name__ == "__main__":
    run()
