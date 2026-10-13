from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        st.error("Model file nahi mili!")
        st.write("Expected path:", str(MODEL_PATH))
        st.write("App directory:", str(BASE_DIR))

        if (BASE_DIR / "models").exists():
            st.write(
                "Models folder files:",
                [p.name for p in (BASE_DIR / "models").iterdir()]
            )
        else:
            st.write("GitHub repository mein models folder nahi mila.")

        st.stop()

    return joblib.load(MODEL_PATH)


model = load_model()



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fb;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #0b2d5c, #1266a8);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        background: white;
        border: 1px solid #e1e7ef;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        text-align: center;
        margin-top: 20px;
    }

    .risk-title {
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #0b2d5c;
        margin-top: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background: #eef5ff;
        border-left: 5px solid #1266a8;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>💳 Credit Scoring Model</h1>
        <p>
            AI-powered credit risk assessment using financial and credit history.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRO
# ============================================================

st.markdown(
    """
    <div class="info-box">
        <b>How it works:</b>
        Enter the applicant's financial information below.
        The trained machine learning pipeline will estimate whether
        the applicant represents a <b>Good</b> or <b>Bad</b> credit risk.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-title">📋 Applicant Financial Information</div>',
    unsafe_allow_html=True
)

with st.form("credit_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        checking_account_status = st.selectbox(
            "Checking Account Status",
            ["A11", "A12", "A13", "A14"],
            format_func=lambda x: {
                "A11": "< 0 DM",
                "A12": "0 - 200 DM",
                "A13": "≥ 200 DM",
                "A14": "No checking account"
            }[x]
        )

        duration_months = st.number_input(
            "Loan Duration (months)",
            min_value=1,
            max_value=120,
            value=24
        )

        credit_history = st.selectbox(
            "Credit History",
            ["A30", "A31", "A32", "A33", "A34"],
            format_func=lambda x: {
                "A30": "No credits / paid duly",
                "A31": "All credits at this bank paid duly",
                "A32": "Existing credits paid duly",
                "A33": "Delay in paying off",
                "A34": "Critical account / other credits"
            }[x]
        )

        purpose = st.selectbox(
            "Loan Purpose",
            ["A40", "A41", "A42", "A43", "A44",
             "A45", "A46", "A47", "A48", "A49", "A410"],
            format_func=lambda x: {
                "A40": "New car",
                "A41": "Used car",
                "A42": "Furniture / equipment",
                "A43": "Radio / television",
                "A44": "Domestic appliances",
                "A45": "Repairs",
                "A46": "Education",
                "A47": "Vacation",
                "A48": "Retraining",
                "A49": "Business",
                "A410": "Other"
            }[x]
        )

        credit_amount = st.number_input(
            "Credit Amount",
            min_value=1,
            value=2500,
            step=100
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        savings_account_bonds = st.selectbox(
            "Savings Account",
            ["A61", "A62", "A63", "A64", "A65"],
            format_func=lambda x: {
                "A61": "< 100 DM",
                "A62": "100 - 500 DM",
                "A63": "500 - 1000 DM",
                "A64": "≥ 1000 DM",
                "A65": "Unknown / no savings"
            }[x]
        )

        present_employment_since = st.selectbox(
            "Employment Since",
            ["A71", "A72", "A73", "A74", "A75"],
            format_func=lambda x: {
                "A71": "Unemployed",
                "A72": "< 1 year",
                "A73": "1 - 4 years",
                "A74": "4 - 7 years",
                "A75": "≥ 7 years"
            }[x]
        )

        installment_rate = st.slider(
            "Installment Rate (% of disposable income)",
            min_value=1,
            max_value=4,
            value=2
        )

        personal_status_sex = st.selectbox(
            "Personal Status",
            ["A91", "A92", "A93", "A94", "A95"],
            format_func=lambda x: {
                "A91": "Male - divorced/separated",
                "A92": "Female - divorced/married",
                "A93": "Male - single",
                "A94": "Male - married/widowed",
                "A95": "Female - single"
            }[x]
        )

        other_debtors_guarantors = st.selectbox(
            "Other Debtors / Guarantors",
            ["A101", "A102", "A103"],
            format_func=lambda x: {
                "A101": "None",
                "A102": "Co-applicant",
                "A103": "Guarantor"
            }[x]
        )

        present_residence_since = st.slider(
            "Present Residence Since (years)",
            min_value=1,
            max_value=4,
            value=2
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        property_value = st.selectbox(
            "Property",
            ["A121", "A122", "A123", "A124"],
            format_func=lambda x: {
                "A121": "Real estate",
                "A122": "Building society / life insurance",
                "A123": "Car or other property",
                "A124": "Unknown / no property"
            }[x]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        other_installment_plans = st.selectbox(
            "Other Installment Plans",
            ["A141", "A142", "A143"],
            format_func=lambda x: {
                "A141": "Bank",
                "A142": "Stores",
                "A143": "None"
            }[x]
        )

        housing = st.selectbox(
            "Housing",
            ["A151", "A152", "A153"],
            format_func=lambda x: {
                "A151": "Rent",
                "A152": "Own",
                "A153": "For free"
            }[x]
        )

        existing_credits = st.number_input(
            "Existing Credits at This Bank",
            min_value=1,
            max_value=10,
            value=1
        )

        job = st.selectbox(
            "Job",
            ["A171", "A172", "A173", "A174"],
            format_func=lambda x: {
                "A171": "Unemployed / unskilled non-resident",
                "A172": "Unskilled resident",
                "A173": "Skilled employee / official",
                "A174": "Management / self-employed / highly qualified"
            }[x]
        )

    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    col4, col5, col6 = st.columns(3)

    with col4:

        people_liable_for_maintenance = st.number_input(
            "People Liable for Maintenance",
            min_value=1,
            max_value=10,
            value=1
        )

    with col5:

        telephone = st.selectbox(
            "Telephone",
            ["A191", "A192"],
            format_func=lambda x: {
                "A191": "No",
                "A192": "Yes - registered under customer's name"
            }[x]
        )

    with col6:

        foreign_worker = st.selectbox(
            "Foreign Worker",
            ["A201", "A202"],
            format_func=lambda x: {
                "A201": "Yes",
                "A202": "No"
            }[x]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍 Assess Credit Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    input_data = pd.DataFrame(
        [{
            "checking_account_status": checking_account_status,
            "duration_months": duration_months,
            "credit_history": credit_history,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "savings_account_bonds": savings_account_bonds,
            "present_employment_since": present_employment_since,
            "installment_rate": installment_rate,
            "personal_status_sex": personal_status_sex,
            "other_debtors_guarantors": other_debtors_guarantors,
            "present_residence_since": present_residence_since,
            "property": property_value,
            "age": age,
            "other_installment_plans": other_installment_plans,
            "housing": housing,
            "existing_credits": existing_credits,
            "job": job,
            "people_liable_for_maintenance": people_liable_for_maintenance,
            "telephone": telephone,
            "foreign_worker": foreign_worker
        }]
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    good_probability = probability[0] * 100
    bad_probability = probability[1] * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Credit Risk Assessment</div>',
        unsafe_allow_html=True
    )

    if prediction == 0:

        st.success("### 🟢 GOOD CREDIT RISK")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="risk-title">Good Credit Risk</div>
                <p>The model classified this application as Good.</p>
                <h3>Good Probability: {good_probability:.2f}%</h3>
                <p>Bad Probability: {bad_probability:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.error("### 🔴 BAD CREDIT RISK")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="risk-title">Bad Credit Risk</div>
                <p>The model classified this application as Bad.</p>
                <h3>Bad Probability: {bad_probability:.2f}%</h3>
                <p>Good Probability: {good_probability:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown("### 📈 Prediction Probability")

    probability_df = pd.DataFrame(
        {
            "Credit Risk": ["Good", "Bad"],
            "Probability": [
                good_probability,
                bad_probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Credit Risk")
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("📋 View Applicant Information"):

        display_data = input_data.T.reset_index()

        display_data.columns = [
            "Feature",
            "Value"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ This application is an educational machine-learning prototype. "
    "It should not be used as the sole basis for real-world lending decisions."
)