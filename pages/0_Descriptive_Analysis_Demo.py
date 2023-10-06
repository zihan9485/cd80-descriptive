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
import plotly.express as px
import plotly.graph_objs as go
import random
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mean
from statsmodels.tsa.seasonal import seasonal_decompose


def descriptive_demo() -> None:

  df = pd.read_csv("data.csv")
  df.rename(columns={"Bankrupt?": "Risk"}, inplace=True)


  def generate_historical_data(company_id):
      # Retrieve the company
      company_data = pd.DataFrame(df.iloc[company_id]).transpose()
      company_data["Year"] = 2020
      company_data.reset_index(drop=True, inplace=True)

      # Generate historical data
      for i in range(10):
          new_row = []
          for (columnName, columnData) in company_data.items():
              prev = columnData.values[i]
              if columnName in ["Risk", " Net Income Flag", " Liability-Assets Flag"]:
                  value = random.choice([0, 1])
              elif columnName == "Year":
                  value = prev - 1
              else:
                  if prev == 0:
                      prev = 0.1
                  value = random.uniform(
                      max(0, prev - (0.5*prev)), prev + (0.5*prev))
              new_row.append(value)

          company_data.loc[company_data.shape[0]] = new_row

      company_data.index = company_data["Year"]
      company_data.drop(columns=["Year"], inplace=True)

      return company_data


  # For simplicity, key financial metrics are the features with the highest corr with Bankrupt
  key_financial_metrics = df.corr()['Risk'].sort_values(
      ascending=False).drop("Risk").dropna().iloc[:3]

  with st.sidebar:
      company_id = st.selectbox(
          'Select Company',
          tuple(range(df.shape[0])))

  st.title("Descriptive analysis")
  st.caption("Company: " + str(company_id))

  company_data = generate_historical_data(company_id)
  with st.expander("Historical data"):
      st.dataframe(company_data)


  st.header("Key Financial Metrics")
  col1, col2, col3 = st.columns(3)
  metric1, metric2, metric3 = key_financial_metrics.index
  current_m1 = company_data.iloc[0][metric1].round(3)
  current_m2 = company_data.iloc[0][metric2].round(3)
  current_m3 = company_data.iloc[0][metric3].round(3)

  diff_m1 = (current_m1 - company_data.iloc[1][metric1]).round(5)
  diff_m2 = (current_m2 - company_data.iloc[1][metric2]).round(5)
  diff_m3 = (current_m3 - company_data.iloc[1][metric3]).round(5)

  col1.metric(metric1, current_m1, diff_m1, delta_color='inverse')
  col2.metric(metric2, current_m2, diff_m2, delta_color='inverse')
  col3.metric(metric3, current_m3, diff_m3, delta_color='inverse')

  st.subheader("Similar Companies (Aggregated Data)")
  cos_sims = {}
  for i in range(df.shape[0]):
      # Skip self
      # if i == company_id:
      #     continue
      cos_sim = cosine_similarity(
          np.array(company_data.iloc[0]).reshape(1, -1), np.array(df.iloc[i]).reshape(1, -1))
      cos_sims[i] = cos_sim[0][0]

  cos_sim_threshold = 0.99
  similar_companies = [k for k, v in cos_sims.items() if v >= cos_sim_threshold]
  df_similar_companies = df.iloc[similar_companies]
  st.write("Number of similar companies: " + str(len(similar_companies)))

  all_similar_company_data = []
  for similar_company_id in similar_companies:
      if similar_company_id == company_id:
          similar_company_data = company_data
      else:
          similar_company_data = generate_historical_data(similar_company_id)
      all_similar_company_data.append(similar_company_data)

  sim_com1, sim_com2, sim_com3 = st.columns(3)
  current_m1 = df_similar_companies[metric1].mean().round(3)
  current_m2 = df_similar_companies[metric2].mean().round(3)
  current_m3 = df_similar_companies[metric3].mean().round(3)

  prev_m1 = mean(
      list(map(lambda x: x.iloc[1][metric1], all_similar_company_data)))
  prev_m2 = mean(
      list(map(lambda x: x.iloc[1][metric2], all_similar_company_data)))
  prev_m3 = mean(
      list(map(lambda x: x.iloc[1][metric3], all_similar_company_data)))

  diff_m1 = (current_m1 - prev_m1).round(5)
  diff_m2 = (current_m2 - prev_m2).round(5)
  diff_m3 = (current_m3 - prev_m3).round(5)

  sim_com1.metric(metric1, current_m1, diff_m1, delta_color='inverse')
  sim_com2.metric(metric2, current_m2, diff_m2, delta_color='inverse')
  sim_com3.metric(metric3, current_m3, diff_m3, delta_color='inverse')

  st.header("Individual Features over Time")
  time_plot_feature = st.selectbox(
      'Select Feature',
      tuple(company_data.columns))


  def SetColor(x):
      if(x == 0):
          return "green"
      elif(x == 1):
          return "red"


  similar_companies_feature = []
  for i in range(company_data.shape[0]):
      similar_companies_feature.append(
          list(map(lambda x: x.iloc[i][time_plot_feature], all_similar_company_data)))

  similar_companies_feature_mean = list(
      map(lambda x: mean(x), similar_companies_feature))
  similar_companies_feature_max = list(
      map(lambda x: max(x), similar_companies_feature))
  similar_companies_feature_min = list(
      map(lambda x: min(x), similar_companies_feature))

  sim_comp_upper_bound = go.Scatter(
      name='Similar Companies Upper Bound',
      x=company_data.index.astype(dtype=str),
      y=similar_companies_feature_max,
      mode='lines',
      line=dict(width=0.5,
                color="rgb(255, 188, 0)"),
      fillcolor='rgba(68, 68, 68, 0.1)',
      fill='tonexty')

  sim_comp_ave = go.Scatter(
      name='Similar Companies Average',
      x=company_data.index.astype(dtype=str),
      y=similar_companies_feature_mean,
      mode='lines',
      line=dict(color='rgb(31, 119, 180)'),
  )

  company_feature = go.Scatter(
      name='Company',
      x=company_data.index.astype(dtype=str),
      y=company_data[time_plot_feature],
      line=dict(color='rgb(246, 23, 26)'),
      marker=dict(color=list(map(SetColor, np.array(company_data['Risk'])))))

  sim_comp_lower_bound = go.Scatter(
      name='Similar Companies Lower Bound',
      x=company_data.index.astype(dtype=str),
      y=similar_companies_feature_min,
      mode='lines',
      line=dict(width=0.5, color="rgb(141, 196, 26)"),)

  time_plot_data = [sim_comp_lower_bound,
                    sim_comp_upper_bound, sim_comp_ave, company_feature]

  fig = go.Figure(data=time_plot_data)

  fig.update_layout({"title": time_plot_feature + " from 2010 to 2020",
                    "xaxis": {"title": "Year"},
                    "yaxis": {"title": time_plot_feature},
                    "showlegend": True})

  st.plotly_chart(fig)
  st.warning("Green marker -> Non-Risk Year, Red marker -> Risk Year")

  st.subheader("Outlier Detection")
  decompose_result = seasonal_decompose(
      company_data[time_plot_feature].values, model='additive', period=2)

  residuals = pd.Series(decompose_result.resid)
  residuals.index = company_data.index
  residuals.dropna(inplace=True)
  Q1 = np.percentile(residuals.values, 25)
  Q3 = np.percentile(residuals.values, 75)
  IQR = Q3 - Q1
  ul = Q3+1.5*IQR
  ll = Q1-1.5*IQR
  outliers = residuals[(residuals > ul) | (residuals < ll)]


  fig, ax = plt.subplots()
  ax.plot(residuals)
  if len(outliers):
      ax.scatter(outliers.index, outliers.values, c='r', label="Outlier")
  plt.title("Residuals of feature: " + time_plot_feature)
  plt.xlabel("Years")
  plt.ylabel("Residual")
  plt.legend()
  st.pyplot(fig)

  if len(outliers):
      st.write("Outliers")
      st.write(outliers)
  else:
      st.write("No Outlier Years")

  st.header("Feature Statistics")
  time_range = st.slider(
      'Select a time range',
      int(company_data.index[-1]), int(company_data.index[0]), (int(company_data.index[-1]), int(company_data.index[0])))
  st.write("From {} to {}".format(time_range[0], time_range[1]))
  st.dataframe(company_data.loc[time_range[1]:time_range[0]].describe())

  st.subheader("Compared to Similar Companies")
  box_plot_feature = st.selectbox(
      'Select Feature',
      (" Debt ratio %", " Borrowing dependency"))

  similar_companies_feature = []
  for i in range(company_data.shape[0]):
      similar_companies_feature.append(
          list(map(lambda x: x.iloc[i][box_plot_feature], all_similar_company_data)))
  df_similar_companies_feature = pd.DataFrame(similar_companies_feature).T
  # st.dataframe(df_similar_companies_feature)
  # fig = px.box(df_similar_companies_feature)

  fig = go.Figure()
  for i, col in enumerate(reversed(list(df_similar_companies_feature.columns))):
      fig.add_trace(
          go.Box(y=df_similar_companies_feature[col], name=str(2010 + i)))

  fig.add_trace(go.Scatter(
      x=[str(2020-i) for i in range(company_data.shape[0])],
      y=company_data[box_plot_feature],
      mode='markers',
      marker=dict(color="red"),
      showlegend=True,
      name="Company"
  ))

  st.plotly_chart(fig)


st.set_page_config(page_title="Descriptive Analysis Demo", page_icon="📹")

descriptive_demo()

show_code(descriptive_demo)
