import streamlit as st
import numpy as np
import pickle
import pandas as pd
import altair as alt
import gzip

# Load the compressed model
with gzip.open("best_model_compressed.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Streamlit app title and description
st.title("🌍 Life Expectancy Prediction 🌍")

st.markdown("""
This application predicts **Life Expectancy** based on key health and economic indicators.
Provide the required inputs and click **Predict** to see the expected life expectancy. ⬇️
""")

# Sidebar for user inputs
st.sidebar.header("Input Features 🏥")

# Input fields
hiv_aids = st.sidebar.slider("HIV/AIDS", 0.0, 10.0, 0.2, step=0.01)
adult_mortality = st.sidebar.slider("Adult Mortality Rate", 0, 500, 180, step=1)
income_composition = st.sidebar.slider("Income Composition of Resources", 0.0, 1.0, 0.5, step=0.01)
schooling = st.sidebar.slider("Years of Schooling", 0, 20, 10, step=1)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0, step=0.1)
under_five_deaths = st.sidebar.slider("Under-Five Deaths", 0, 300, 20, step=1)
thinness_5_9_years = st.sidebar.slider("Thinness (5-9 years)", 0.0, 30.0, 5.0, step=0.1)
infant_deaths = st.sidebar.slider("Infant Deaths", 0, 300, 20, step=1)
thinness_1_19_years = st.sidebar.slider("Thinness (1-19 years)", 0.0, 30.0, 5.0, step=0.1)
diphtheria = st.sidebar.slider("Diphtheria (%)", 0.0, 100.0, 80.0, step=1.0)
polio = st.sidebar.slider("Polio (%)", 0.0, 100.0, 80.0, step=1.0)
gdp = st.sidebar.slider("GDP per Capita", 100, 100000, 5000, step=100)
status = st.sidebar.selectbox("Country Status", ["Developing", "Developed"])

# Convert categorical 'Status' to numeric (Assume: Developing=0, Developed=1)
status = 1 if status == "Developed" else 0

alcohol = st.sidebar.slider("Alcohol Consumption (liters)", 0.0, 20.0, 5.0, step=0.1)
total_expenditure = st.sidebar.slider("Total Health Expenditure (% of GDP)", 0.0, 20.0, 5.0, step=0.1)
hepatitis_b = st.sidebar.slider("Hepatitis B (%)", 0.0, 100.0, 80.0, step=1.0)
percentage_expenditure = st.sidebar.slider("Health Expenditure per Capita", 100, 100000, 5000, step=100)
measles = st.sidebar.slider("Measles Cases", 0, 100000, 500, step=10)

# Collect all features into an array
features = np.array([
    status, adult_mortality, infant_deaths, alcohol, percentage_expenditure,
    hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure,
    diphtheria, hiv_aids, gdp, thinness_1_19_years, thinness_5_9_years,
    income_composition, schooling
]).reshape(1, -1)

# Initialize prediction variable
prediction = None

# Prediction button
if st.button("🔮 Predict Life Expectancy"):
    with st.spinner("✨ Calculating..."):
        try:
            # Make prediction
            prediction = model.predict(features)[0]
            st.success(f"✨ The predicted Life Expectancy is: **{prediction:.2f} years** ✨")

            # Visualization of input features
            st.markdown("### 🌟 Feature vs Prediction Visualization")

            df_viz = pd.DataFrame({
                'Feature': [
                    'Status', 'Adult Mortality', 'Infant Deaths', 'Alcohol Consumption',
                    'Health Expenditure per Capita', 'Hepatitis B', 'Measles', 'BMI',
                    'Under-Five Deaths', 'Polio', 'Total Expenditure', 'Diphtheria',
                    'HIV/AIDS', 'GDP', 'Thinness (1-19 years)', 'Thinness (5-9 years)',
                    'Income Composition', 'Schooling'
                ],
                'Value': features[0]
            })
            chart = alt.Chart(df_viz).mark_bar().encode(
                x='Feature:N',
                y='Value:Q',
                color='Feature:N',
                tooltip=['Feature', 'Value']
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Confidence Interval
            st.markdown("### 📈 Confidence Interval")
            lower_bound = prediction * 0.95  # 5% lower
            upper_bound = prediction * 1.05  # 5% higher
            st.write(f"Confidence Interval: **{lower_bound:.2f} to {upper_bound:.2f} years**")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Download Prediction Section - Only show if prediction exists
if prediction is not None:
    st.markdown("### 📂 Download Prediction")

    # Prepare results for download
    results = pd.DataFrame({
        'Feature': [
            'Status', 'Adult Mortality', 'Infant Deaths', 'Alcohol Consumption',
            'Health Expenditure per Capita', 'Hepatitis B', 'Measles', 'BMI',
            'Under-Five Deaths', 'Polio', 'Total Expenditure', 'Diphtheria',
            'HIV/AIDS', 'GDP', 'Thinness (1-19 years)', 'Thinness (5-9 years)',
            'Income Composition', 'Schooling'
        ],
        'Input Value': features[0],
        'Predicted Life Expectancy (Years)': [prediction] * len(features[0])
    })

    st.download_button("📂 Download Prediction as CSV", results.to_csv(index=False), "prediction.csv", "text/csv")

# Footer
st.markdown("""
---
Made with ❤️ for global health research. 🌍
""")
