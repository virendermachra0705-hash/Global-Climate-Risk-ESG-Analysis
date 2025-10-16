
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


@st.cache_data
def load_data():
    df = pd.read_csv("final_dashboard.csv")

    # Drop duplicate / unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Handle population strings like '20M', '15.5K', etc.
    def parse_population(val):
        if isinstance(val, str):
            val = val.strip().upper()
            if 'M' in val:
                return float(val.replace('M', '')) * 1e6
            elif 'K' in val:
                return float(val.replace('K', '')) * 1e3
            elif 'B' in val:
                return float(val.replace('B', '')) * 1e9
            else:
                try:
                    return float(val)
                except:
                    return np.nan
        return val

    if 'POP_2020' in df.columns:
        df['POP_2020'] = df['POP_2020'].apply(parse_population)

    # Drop rows with nulls
    df = df.dropna(subset=['ESG_Final', 'GDP_2020', 'SDI_2019', 'Temperature'])

    # Normalize selected columns
    scaler = MinMaxScaler()
    cols_to_normalize = ['ESG_Final', 'GDP_2020', 'SDI_2019', 'Temperature']
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

    # Create composite indexes
    df['Sustainability_Index'] = df[['ESG_Final', 'SDI_2019']].mean(axis=1)
    df['Economic_Index'] = df[['GDP_2020', 'POP_2020']].mean(axis=1)
    df['Climate_Risk_Index'] = df['Temperature']

    return df


# Load data
df = load_data()

st.set_page_config(
    page_title="Global ESG & Climate Risk Dashboard", layout="wide")

st.title("🌍 Global Climate Risk & ESG Data Analysis Dashboard")
st.markdown("""
This interactive dashboard explores the relationship between **Environmental, Social, and Governance (ESG)** performance,
**economic indicators (GDP, SDI, Population)**, and **climate risk (temperature impact)** across countries.
""")


st.sidebar.header(" Filter Options")
countries = st.sidebar.multiselect(
    "Select Countries", df['Country Name'].unique())
if countries:
    df = df[df['Country Name'].isin(countries)]

st.subheader("Global Overview Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg ESG Score", f"{df['ESG_Final'].mean():.2f}")
col2.metric("Avg GDP (normalized)", f"{df['GDP_2020'].mean():.2f}")
col3.metric("Avg SDI", f"{df['SDI_2019'].mean():.2f}")
col4.metric("Avg Temperature", f"{df['Temperature'].mean():.2f}")


st.subheader("Correlation Analysis")

corr = df[['ESG_Final', 'GDP_2020', 'SDI_2019', 'Temperature',
           'Sustainability_Index', 'Economic_Index', 'Climate_Risk_Index']].corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="Correlation Heatmap of ESG, Economic, and Climate Indicators"
)
st.plotly_chart(fig_corr, use_container_width=True)


st.subheader("🔍 Relationship Explorer")

x_axis = st.selectbox(
    "Select X-axis", df.select_dtypes(include=np.number).columns, index=1)
y_axis = st.selectbox(
    "Select Y-axis", df.select_dtypes(include=np.number).columns, index=0)

fig_scatter = px.scatter(
    df, x=x_axis, y=y_axis,
    color='Sustainability_Index',
    hover_name='Country Name',
    size='Economic_Index',
    title=f"{y_axis} vs {x_axis}"
)
st.plotly_chart(fig_scatter, use_container_width=True)


st.subheader("🗺️ Global Sustainability & Climate Map")

selected_metric = st.selectbox("Select Metric for Map",
                               ['ESG_Final', 'Sustainability_Index', 'Economic_Index', 'Climate_Risk_Index'])

fig_map = px.choropleth(
    df, locations="Country Name", locationmode="country names",
    color=selected_metric,
    hover_name="Country Name",
    color_continuous_scale="Viridis",
    title=f"Global Distribution of {selected_metric}"
)
st.plotly_chart(fig_map, use_container_width=True)


st.subheader("🌡️ Scenario Analysis: Climate Risk Impact on ESG")

temp_increase = st.slider(
    "Simulate Temperature Increase (°C)", 0.0, 3.0, 1.0, 0.1)
df['Adjusted_Climate_Risk'] = df['Temperature'] + (temp_increase * 0.05)
df['Adjusted_Sustainability'] = df['Sustainability_Index'] - \
    (temp_increase * 0.03)

fig_scenario = px.scatter(
    df, x='Adjusted_Climate_Risk', y='Adjusted_Sustainability',
    color='Economic_Index', hover_name='Country Name',
    title=f"Impact of +{temp_increase}°C on Sustainability vs Climate Risk"
)
st.plotly_chart(fig_scenario, use_container_width=True)


st.subheader("🤖 Predictive ESG Modeling (Linear Regression)")

# Drop rows with NaN
df_model = df.dropna(
    subset=['ESG_Final', 'GDP_2020', 'SDI_2019', 'Temperature'])
X = df_model[['GDP_2020', 'SDI_2019', 'Temperature']]
y = df_model['ESG_Final']

model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)

df_model['Predicted_ESG'] = preds

fig_pred = px.scatter(
    df_model, x='ESG_Final', y='Predicted_ESG',
    trendline='ols', color='Sustainability_Index',
    hover_name='Country Name',
    title="Actual vs Predicted ESG Performance"
)
st.plotly_chart(fig_pred, use_container_width=True)

st.markdown(
    "*This simple model estimates ESG scores based on economic and climate indicators.*")


st.subheader("Top & Bottom 10 Countries by Sustainability Index")

col1, col2 = st.columns(2)
top10 = df.nlargest(10, 'Sustainability_Index')[
    ['Country Name', 'Sustainability_Index']]
bottom10 = df.nsmallest(10, 'Sustainability_Index')[
    ['Country Name', 'Sustainability_Index']]

with col1:
    st.markdown("### Top 10 Countries")
    st.dataframe(top10)

with col2:
    st.markdown("### Bottom 10 Countries")
    st.dataframe(bottom10)


st.markdown("---")
st.markdown("**Project:** Global Climate Risk & ESG Data Analysis (2025)")
st.markdown("**Developed by**: Virender | Data Science Portfolio Project")
st.markdown(
    "*Analyzing how sustainability, economic strength, and climate risk interact globally.*")
