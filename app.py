import streamlit as st

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Insurance Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🏥 Insurance System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📝 Apply for Insurance",
        "📊 Risk Analysis",
        "📈 Model Evaluation",
        "📋 Column Analysis",
        "⚖ Fairness Analysis"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "🏠 Dashboard":

    st.title("🏥 Insurance Risk Prediction")

    st.write(
        "Medical Insurance Risk Prediction using Logistic Regression"
    )

    st.divider()

    st.header("Insurance Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applicants",
        "0"
    )

    col2.metric(
        "High Risk",
        "0"
    )

    col3.metric(
        "Low Risk",
        "0"
    )

    col4.metric(
        "Model",
        "Logistic Regression"
    )

    st.divider()

    st.subheader("Welcome")

    st.info(
        "Use the menu on the left to apply for insurance, "
        "view risk analysis, evaluate the model, and analyze data."
    )


# --------------------------------------------------
# ONLINE INSURANCE APPLICATION
# --------------------------------------------------

elif page == "📝 Apply for Insurance":

    st.title("📝 Online Insurance Application")

    st.write(
        "Enter the applicant information below."
    )

    st.divider()

    st.header("Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    st.header("Financial Information")

    col1, col2 = st.columns(2)

    with col1:

        income = st.number_input(
            "Annual Income",
            min_value=0.0,
            value=50000.0
        )

    with col2:

        employment_status = st.selectbox(
            "Employment Status",
            [
                "Employed",
                "Unemployed",
                "Self-employed",
                "Retired"
            ]
        )

    st.header("Other Information")

    col1, col2 = st.columns(2)

    with col1:

        region = st.selectbox(
            "Region",
            [
                "North",
                "South",
                "East",
                "West",
                "Central"
            ]
        )

    with col2:

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

    household_size = st.number_input(
        "Household Size",
        min_value=1,
        max_value=20,
        value=2
    )

    st.divider()

    if st.button(
        "🔍 Predict Insurance Risk",
        use_container_width=True
    ):

        st.success(
            "Application submitted successfully!"
        )

        st.subheader("Prediction Result")

        st.info(
            "Logistic Regression will be connected in the next step."
        )


# --------------------------------------------------
# RISK ANALYSIS
# --------------------------------------------------

elif page == "📊 Risk Analysis":

    st.title("📊 Insurance Risk Analysis")

    st.write(
        "Risk distribution and applicant analysis."
    )

    st.info(
        "Risk graphs will be connected after the model is trained."
    )


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "📈 Model Evaluation":

    st.title("📈 Logistic Regression Model Evaluation")

    st.subheader("Classification Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "Not calculated")
    col2.metric("Precision", "Not calculated")
    col3.metric("Recall", "Not calculated")
    col4.metric("F1 Score", "Not calculated")

    st.subheader("Error Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MSE", "Not calculated")
    col2.metric("MAE", "Not calculated")
    col3.metric("RMSE", "Not calculated")
    col4.metric("R²", "Not calculated")


# --------------------------------------------------
# COLUMN ANALYSIS
# --------------------------------------------------

elif page == "📋 Column Analysis":

    st.title("📋 Column Analysis")

    st.write(
        "Numerical and categorical column analysis."
    )

    st.info(
        "Your CSV column analysis will be connected in the next step."
    )


# --------------------------------------------------
# FAIRNESS ANALYSIS
# --------------------------------------------------

elif page == "⚖ Fairness Analysis":

    st.title("⚖ Fairness Analysis")

    st.write(
        "Comparison of model predictions across applicant groups."
    )

    st.info(
        "Fairness analysis will be connected after the model is trained."
    )
