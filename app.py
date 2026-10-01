import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Insurance Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-weight: 700;
    }

    h2 {
        font-weight: 650;
    }

    h3 {
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #dddddd;
        padding: 15px;
        border-radius: 12px;
        background-color: #fafafa;
    }

    .risk-high {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #ffe6e6;
        border: 2px solid #ff4b4b;
    }

    .risk-low {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        background-color: #e6ffe6;
        border: 2px solid #21a366;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DATASET
# ============================================================

CSV_FILE = "medical_insurance (2).csv"

try:

    df = pd.read_csv(CSV_FILE)

except FileNotFoundError:

    st.error(
        f"""
        ❌ Dataset not found.

        Please make sure that:

        1. app.py
        2. {CSV_FILE}

        are in the same GitHub repository folder.
        """
    )

    st.stop()

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# ============================================================
# TARGET DETECTION
# ============================================================

target_candidates = [
    "is_high_risk",
    "high_risk",
    "highrisk",
    "risk",
    "risk_level",
    "risk_class",
    "risk_classification",
    "risk_category",
    "target",
    "outcome",
    "label"
]

target_column = None

for candidate in target_candidates:

    if candidate in df.columns:

        target_column = candidate
        break

# Automatic binary target detection
if target_column is None:

    binary_columns = []

    for col in df.columns:

        unique_values = df[col].dropna().unique()

        if len(unique_values) == 2:

            binary_columns.append(col)

    risk_columns = [
        col
        for col in binary_columns
        if "risk" in col.lower()
    ]

    if risk_columns:

        target_column = risk_columns[0]

    elif binary_columns:

        target_column = binary_columns[-1]

# ============================================================
# MODEL VARIABLES
# ============================================================

model = None

X = None
y = None

X_train = None
X_test = None
y_train = None
y_test = None

y_pred = None
y_probability = None

model_error = None

target_mapping = {}

# ============================================================
# MODEL TRAINING
# ============================================================

if target_column is not None:

    try:

        model_df = df.dropna(
            subset=[target_column]
        ).copy()

        X = model_df.drop(
            columns=[target_column]
        )

        y = model_df[target_column]

        unique_target = list(
            y.dropna().unique()
        )

        # ----------------------------------------------------
        # BINARY TARGET
        # ----------------------------------------------------

        if len(unique_target) == 2:

            target_mapping = {
                unique_target[0]: 0,
                unique_target[1]: 1
            }

            y = y.map(target_mapping)

            # ------------------------------------------------
            # REMOVE ID COLUMNS
            # ------------------------------------------------

            id_columns = []

            for col in X.columns:

                name = col.lower()

                if (
                    name == "id"
                    or name.endswith("_id")
                    or name.startswith("id_")
                ):

                    id_columns.append(col)

            X = X.drop(
                columns=id_columns,
                errors="ignore"
            )

            # ------------------------------------------------
            # FEATURE TYPES
            # ------------------------------------------------

            numeric_features = (
                X
                .select_dtypes(
                    include=["number"]
                )
                .columns
                .tolist()
            )

            categorical_features = (
                X
                .select_dtypes(
                    exclude=["number"]
                )
                .columns
                .tolist()
            )

            # ------------------------------------------------
            # NUMERIC PIPELINE
            # ------------------------------------------------

            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # ------------------------------------------------
            # CATEGORICAL PIPELINE
            # ------------------------------------------------

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            # ------------------------------------------------
            # PREPROCESSOR
            # ------------------------------------------------

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        numeric_pipeline,
                        numeric_features
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_features
                    )
                ]
            )

            # ------------------------------------------------
            # LOGISTIC REGRESSION
            # ------------------------------------------------

            model = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000
                        )
                    )
                ]
            )

            # ------------------------------------------------
            # TRAIN TEST SPLIT
            # ------------------------------------------------

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )
            )

            # ------------------------------------------------
            # TRAIN MODEL
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )

            y_probability = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

        else:

            model_error = (
                "The selected target column does not "
                "contain exactly two classes."
            )

    except Exception as e:

        model_error = str(e)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Insurance System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📝 Online Insurance Application",
        "📊 Risk Analysis",
        "📈 Model Evaluation",
        "📋 Column Analysis",
        "⚖️ Fairness Analysis"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Dataset Information")

st.sidebar.write(
    f"Rows: **{len(df):,}**"
)

st.sidebar.write(
    f"Columns: **{len(df.columns):,}**"
)

if target_column:

    st.sidebar.success(
        f"Target: {target_column}"
    )

else:

    st.sidebar.warning(
        "Risk target not detected."
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Logistic Regression Insurance Risk Prediction"
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🏥 Insurance Risk Prediction System"
    )

    st.write(
        "A machine-learning based insurance risk "
        "prediction and analysis platform."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    total_rows = len(df)
    total_columns = len(df.columns)

    if target_column:

        target_values = (
            df[target_column]
            .astype(str)
            .str.lower()
        )

        high_risk_count = target_values.isin(
            [
                "1",
                "high",
                "high risk",
                "true",
                "yes"
            ]
        ).sum()

        low_risk_count = (
            total_rows - high_risk_count
        )

    else:

        high_risk_count = 0
        low_risk_count = total_rows

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Total Applicants",
        f"{total_rows:,}"
    )

    c2.metric(
        "📋 Total Columns",
        f"{total_columns:,}"
    )

    c3.metric(
        "🔴 High Risk",
        f"{high_risk_count:,}"
    )

    c4.metric(
        "🟢 Not High Risk",
        f"{low_risk_count:,}"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.subheader(
        "🤖 Machine Learning Model"
    )

    if model is not None:

        st.success(
            "Logistic Regression model trained successfully."
        )

        mc1, mc2, mc3 = st.columns(3)

        mc1.metric(
            "Training Records",
            f"{len(X_train):,}"
        )

        mc2.metric(
            "Testing Records",
            f"{len(X_test):,}"
        )

        mc3.metric(
            "Features",
            f"{len(X.columns):,}"
        )

    else:

        st.error(
            "Model is not available."
        )

        if model_error:

            st.warning(
                model_error
            )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "📊 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    if target_column:

        st.subheader(
            "📊 Risk Distribution"
        )

        risk_distribution = (
            df[target_column]
            .astype(str)
            .value_counts()
            .rename_axis("Risk")
            .reset_index(
                name="Applicants"
            )
        )

        st.bar_chart(
            risk_distribution.set_index(
                "Risk"
            )
        )

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📈 Numerical Data Summary"
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

# ============================================================
# ONLINE INSURANCE APPLICATION
# ============================================================

elif page == "📝 Online Insurance Application":

    st.title(
        "📝 Online Insurance Application"
    )

    st.write(
        "Complete the applicant information below "
        "to estimate insurance risk."
    )

    if model is None:

        st.error(
            "The prediction model is unavailable."
        )

        if model_error:

            st.warning(
                model_error
            )

        st.stop()

    # --------------------------------------------------------
    # IMPORTANT COLUMN SELECTION
    # --------------------------------------------------------

    all_features = X.columns.tolist()

    keyword_groups = {

        "👤 Personal Information": [
            "age",
            "gender",
            "sex",
            "marital",
            "region",
            "state",
            "city",
            "depend",
            "household",
            "family"
        ],

        "💰 Financial & Employment Information": [
            "income",
            "salary",
            "employment",
            "occupation",
            "education",
            "job",
            "work"
        ],

        "❤️ Health Information": [
            "bmi",
            "smok",
            "alcohol",
            "diabetes",
            "hypertension",
            "blood_pressure",
            "heart",
            "cholesterol",
            "disease",
            "condition",
            "health",
            "exercise",
            "physical"
        ],

        "🏥 Medical History": [
            "hospital",
            "surgery",
            "medication",
            "medical",
            "family_history",
            "claim",
            "doctor",
            "diagnosis",
            "treatment"
        ],

        "🛡️ Insurance Information": [
            "insurance",
            "coverage",
            "premium",
            "policy",
            "previous",
            "plan",
            "deductible",
            "benefit"
        ]
    }

    selected_columns = []

    # Automatically identify important columns
    for keywords in keyword_groups.values():

        for col in all_features:

            col_lower = col.lower()

            if any(
                keyword in col_lower
                for keyword in keywords
            ):

                if col not in selected_columns:

                    selected_columns.append(col)

    # If too few columns were detected
    if len(selected_columns) < 5:

        selected_columns = all_features[:15]

    # Prevent extremely large forms
    selected_columns = selected_columns[:30]

    user_input = {}

    # --------------------------------------------------------
    # CREATE FORM
    # --------------------------------------------------------

    for section_name, keywords in (
        keyword_groups.items()
    ):

        section_columns = []

        for col in selected_columns:

            if any(
                keyword in col.lower()
                for keyword in keywords
            ):

                section_columns.append(col)

        if section_columns:

            st.subheader(
                section_name
            )

            form_columns = st.columns(2)

            for i, col in enumerate(
                section_columns
            ):

                with form_columns[
                    i % 2
                ]:

                    series = df[col]

                    label = (
                        col
                        .replace("_", " ")
                        .title()
                    )

                    # ----------------------------------------
                    # CATEGORICAL INPUT
                    # ----------------------------------------

                    if (
                        series.dtype == "object"
                        or str(
                            series.dtype
                        ).startswith("category")
                        or series.nunique(
                            dropna=True
                        ) <= 10
                    ):

                        values = (
                            series
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )

                        values = values[:50]

                        if values:

                            user_input[col] = (
                                st.selectbox(
                                    label,
                                    values,
                                    key=f"input_{col}"
                                )
                            )

                    # ----------------------------------------
                    # NUMERIC INPUT
                    # ----------------------------------------

                    else:

                        median_value = (
                            series.median()
                        )

                        if pd.isna(
                            median_value
                        ):

                            median_value = 0.0

                        user_input[col] = (
                            st.number_input(
                                label,
                                value=float(
                                    median_value
                                ),
                                key=f"input_{col}"
                            )
                        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Insurance Risk",
        type="primary",
        use_container_width=True
    ):

        application = {}

        # Fill every model feature
        for col in all_features:

            series = X[col]

            if col in user_input:

                application[col] = (
                    user_input[col]
                )

            elif pd.api.types.is_numeric_dtype(
                series
            ):

                value = series.median()

                if pd.isna(value):

                    value = 0

                application[col] = value

            else:

                mode = series.mode()

                if len(mode) > 0:

                    application[col] = (
                        mode.iloc[0]
                    )

                else:

                    application[col] = "Unknown"

        application_df = pd.DataFrame(
            [application]
        )

        try:

            prediction = model.predict(
                application_df
            )[0]

            probability = (
                model.predict_proba(
                    application_df
                )[0][1]
            )

            probability_percent = (
                probability * 100
            )

            st.markdown("---")

            st.subheader(
                "🏥 Prediction Result"
            )

            # ------------------------------------------------
            # HIGH RISK
            # ------------------------------------------------

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="risk-high">

                    <h2>🔴 HIGH RISK</h2>

                    <h3>
                    Estimated Risk:
                    {probability_percent:.2f}%
                    </h3>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # LOW RISK
            # ------------------------------------------------

            else:

                st.markdown(
                    f"""
                    <div class="risk-low">

                    <h2>🟢 NOT HIGH RISK</h2>

                    <h3>
                    Estimated High-Risk Probability:
                    {probability_percent:.2f}%
                    </h3>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # ------------------------------------------------
            # PROBABILITY METRICS
            # ------------------------------------------------

            p1, p2 = st.columns(2)

            p1.metric(
                "🟢 Not High Risk",
                f"{(1 - probability) * 100:.2f}%"
            )

            p2.metric(
                "🔴 High Risk",
                f"{probability * 100:.2f}%"
            )

            # ------------------------------------------------
            # PROBABILITY GRAPH
            # ------------------------------------------------

            probability_df = pd.DataFrame(
                {
                    "Category": [
                        "Not High Risk",
                        "High Risk"
                    ],
                    "Probability": [
                        1 - probability,
                        probability
                    ]
                }
            )

            st.subheader(
                "📊 Risk Probability"
            )

            st.bar_chart(
                probability_df.set_index(
                    "Category"
                )
            )

            st.info(
                """
                This prediction is generated by a
                Logistic Regression machine-learning model.
                It is intended for project/demo purposes and
                should not be used as an automatic medical
                diagnosis or insurance underwriting decision.
                """
            )

        except Exception as e:

            st.error(
                "Prediction could not be generated."
            )

            st.exception(e)

# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "📊 Risk Analysis":

    st.title(
        "📊 Insurance Risk Analysis"
    )

    if target_column is None:

        st.warning(
            "No risk target column was detected."
        )

    else:

        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "📊 Risk Distribution"
        )

        risk_counts = (
            df[target_column]
            .astype(str)
            .value_counts()
        )

        st.bar_chart(
            risk_counts
        )

        # ----------------------------------------------------
        # RISK PERCENTAGE
        # ----------------------------------------------------

        st.subheader(
            "📈 Risk Percentages"
        )

        risk_percentage = (
            df[target_column]
            .astype(str)
            .value_counts(
                normalize=True
            )
            * 100
        )

        risk_percentage = (
            risk_percentage.round(2)
        )

        st.dataframe(
            risk_percentage.rename(
                "Percentage"
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # NUMERICAL ANALYSIS
        # ----------------------------------------------------

        numeric_columns = (
            X
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if numeric_columns:

            st.subheader(
                "🔢 Numerical Variable Analysis"
            )

            selected_numeric = (
                st.selectbox(
                    "Select a numerical column",
                    numeric_columns
                )
            )

            st.write(
                f"Analysis of **{selected_numeric}**"
            )

            numeric_stats = pd.DataFrame(
                {
                    "Statistic": [
                        "Mean",
                        "Median",
                        "Minimum",
                        "Maximum",
                        "Missing Values"
                    ],
                    "Value": [
                        df[selected_numeric].mean(),
                        df[selected_numeric].median(),
                        df[selected_numeric].min(),
                        df[selected_numeric].max(),
                        df[selected_numeric].isna().sum()
                    ]
                }
            )

            st.dataframe(
                numeric_stats,
                use_container_width=True
            )

            st.line_chart(
                df[selected_numeric]
                .value_counts()
                .sort_index()
            )

        # ----------------------------------------------------
        # CATEGORICAL ANALYSIS
        # ----------------------------------------------------

        categorical_columns = (
            X
            .select_dtypes(
                exclude="number"
            )
            .columns
            .tolist()
        )

        if categorical_columns:

            st.subheader(
                "🔤 Categorical Variable Analysis"
            )

            selected_category = (
                st.selectbox(
                    "Select a categorical column",
                    categorical_columns
                )
            )

            category_counts = (
                df[selected_category]
                .astype(str)
                .value_counts()
            )

            st.bar_chart(
                category_counts
            )

# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.title(
        "📈 Logistic Regression Model Evaluation"
    )

    if (
        model is None
        or y_test is None
        or y_pred is None
    ):

        st.error(
            "Model evaluation is not available."
        )

    else:

        # ----------------------------------------------------
        # CLASSIFICATION METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        c4.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # ERROR METRICS
        # ----------------------------------------------------

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        st.subheader(
            "📐 Error Metrics"
        )

        e1, e2, e3, e4 = st.columns(4)

        e1.metric(
            "MSE",
            f"{mse:.4f}"
        )

        e2.metric(
            "MAE",
            f"{mae:.4f}"
        )

        e3.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

        e4.metric(
            "R²",
            f"{r2:.4f}"
        )

        st.markdown("---")

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.subheader(
            "🔲 Confusion Matrix"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual 0",
                "Actual 1"
            ],
            columns=[
                "Predicted 0",
                "Predicted 1"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.bar_chart(
            cm_df
        )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.subheader(
            "📋 Classification Report"
        )

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = (
            pd.DataFrame(report)
            .transpose()
        )

        st.dataframe(
            report_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        st.subheader(
            "🔍 Actual vs Predicted Sample"
        )

        sample_size = min(
            100,
            len(y_test)
        )

        comparison = pd.DataFrame(
            {
                "Actual": y_test.values[
                    :sample_size
                ],
                "Predicted": y_pred[
                    :sample_size
                ]
            }
        )

        st.dataframe(
            comparison,
            use_container_width=True
        )

# ============================================================
# COLUMN ANALYSIS
# ============================================================

elif page == "📋 Column Analysis":

    st.title(
        "📋 Complete Column Analysis"
    )

    st.write(
        "Detailed information about every column "
        "in the insurance dataset."
    )

    analysis = []

    for column in df.columns:

        data = df[column]

        if pd.api.types.is_numeric_dtype(
            data
        ):

            analysis.append(
                {
                    "Column": column,
                    "Type": "Numerical",
                    "Count": int(
                        data.count()
                    ),
                    "Missing": int(
                        data.isna().sum()
                    ),
                    "Different Values": int(
                        data.nunique()
                    ),
                    "Average": round(
                        data.mean(),
                        3
                    ),
                    "Minimum": data.min(),
                    "Maximum": data.max()
                }
            )

        else:

            analysis.append(
                {
                    "Column": column,
                    "Type": "Categorical",
                    "Count": int(
                        data.count()
                    ),
                    "Missing": int(
                        data.isna().sum()
                    ),
                    "Different Values": int(
                        data.nunique()
                    ),
                    "Average": "-",
                    "Minimum": "-",
                    "Maximum": "-"
                }
            )

    analysis_df = pd.DataFrame(
        analysis
    )

    st.dataframe(
        analysis_df,
        use_container_width=True,
        height=600
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INDIVIDUAL COLUMN
    # --------------------------------------------------------

    st.subheader(
        "🔎 Individual Column Analysis"
    )

    selected_column = st.selectbox(
        "Select a column",
        df.columns
    )

    selected_data = df[
        selected_column
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Number of Values",
        f"{selected_data.count():,}"
    )

    c2.metric(
        "Different Values",
        f"{selected_data.nunique():,}"
    )

    c3.metric(
        "Missing Values",
        f"{selected_data.isna().sum():,}"
    )

    st.subheader(
        "📊 Value Distribution"
    )

    st.bar_chart(
        selected_data
        .astype(str)
        .value_counts()
        .head(30)
    )

# ============================================================
# FAIRNESS ANALYSIS
# ============================================================

elif page == "⚖️ Fairness Analysis":

    st.title(
        "⚖️ Fairness Analysis"
    )

    st.write(
        """
        This section compares the model's predicted
        high-risk rates across selected demographic
        groups.
        """
    )

    if model is None:

        st.warning(
            "The Logistic Regression model is unavailable."
        )

    else:

        # ----------------------------------------------------
        # FIND DEMOGRAPHIC COLUMNS
        # ----------------------------------------------------

        fairness_candidates = []

        for col in X.columns:

            name = col.lower()

            if any(
                keyword in name
                for keyword in [
                    "gender",
                    "sex",
                    "age",
                    "region",
                    "race",
                    "ethnicity",
                    "marital",
                    "state"
                ]
            ):

                fairness_candidates.append(
                    col
                )

        if fairness_candidates:

            fairness_column = (
                st.selectbox(
                    "Select demographic column",
                    fairness_candidates
                )
            )

            fairness_df = X_test.copy()

            fairness_df["actual"] = (
                y_test.values
            )

            fairness_df["predicted"] = (
                y_pred
            )

            fairness_df["group"] = (
                fairness_df[
                    fairness_column
                ]
                .astype(str)
            )

            fairness_summary = (
                fairness_df
                .groupby("group")
                .agg(
                    Applicants=(
                        "predicted",
                        "count"
                    ),
                    Predicted_High_Risk=(
                        "predicted",
                        "sum"
                    )
                )
            )

            fairness_summary[
                "High_Risk_Rate_%"
            ] = (
                fairness_summary[
                    "Predicted_High_Risk"
                ]
                /
                fairness_summary[
                    "Applicants"
                ]
                * 100
            ).round(2)

            st.subheader(
                "📊 Group Comparison"
            )

            st.dataframe(
                fairness_summary,
                use_container_width=True
            )

            st.subheader(
                "📈 Predicted High-Risk Rate"
            )

            st.bar_chart(
                fairness_summary[
                    "High_Risk_Rate_%"
                ]
            )

            st.info(
                """
                Fairness analysis is provided for
                educational and model-auditing purposes.
                Differences between groups do not by
                themselves prove that a model is unfair;
                they indicate areas that may require
                further investigation.
                """
            )

        else:

            st.info(
                """
                No obvious demographic columns were
                automatically detected in the dataset.
                """
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🏥 Insurance Risk Prediction System | "
    "Logistic Regression | Streamlit"
)
