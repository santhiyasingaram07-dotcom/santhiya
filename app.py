import streamlit as st

st.set_page_config(
    page_title="Insurance Risk Prediction",
    page_icon="🏥"
)

st.title("🏥 Insurance Risk Prediction")

st.write(
    "Enter the customer information below."
)

st.divider()

st.header("Customer Information")

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

region = st.selectbox(
    "Region",
    ["North", "South", "East", "West", "Central"]
)

education = st.selectbox(
    "Education",
    [
        "No HS",
        "HS",
        "Some College",
        "Bachelor",
        "Master",
        "Doctorate"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced",
        "Widowed"
    ]
)

employment_status = st.selectbox(
    "Employment Status",
    [
        "Employed",
        "Unemployed",
        "Self-employed",
        "Retired"
    ]
)

household_size = st.number_input(
    "Household Size",
    min_value=1,
    max_value=20,
    value=2
)

st.divider()

if st.button("🔍 Predict Risk", use_container_width=True):

    st.success("Information submitted successfully!")

    st.subheader("Prediction Result")

    st.info(
        "The Logistic Regression model will be connected here."
    )
