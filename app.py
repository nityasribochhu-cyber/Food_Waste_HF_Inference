
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Food Waste AI Management",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "model_food_waste.pkl"

# =====================================================
# CUSTOM DASHBOARD THEME
# =====================================================


st.markdown("""
<style>

.stApp {
    background-color: #F4F2FA;
    color: #24213D;
}

[data-testid="stSidebar"] {
    background-color: #17152F;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 38px;
    border-radius: 20px;
    background: linear-gradient(120deg, #17152F, #7047D7);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
}

.hero p {
    color: #EAE5FF;
    font-size: 17px;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E4DDF7;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(40, 30, 80, 0.06);
}

.stButton button,
.stFormSubmitButton button {
    background-color: #7047D7;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #5631B5;
    color: white;
    border: none;
}

h1, h2, h3 {
    color: #30205E;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD PRETRAINED MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("model_food_waste.pkl file not found.")
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# =====================================================
# MODEL FEATURES AND SECTOR ENCODING
# =====================================================

features = [
    "sector_enc",
    "month",
    "is_ramadan",
    "is_eid",
    "household_size",
    "income_tier",
    "expat_pct",
    "buffet_style",
    "portion_control",
    "cold_chain_quality",
    "food_literacy",
    "temperature_c",
    "storage_quality",
    "purchase_planning",
    "delivery_spoilage",
    "menu_diversity",
    "overbuying_idx",
    "awareness_campaign",
    "penalty_regime",
    "composting_avail"
]

sector_encoding = {
    "Catering": 0,
    "Hospital": 1,
    "Household": 2,
    "Hotel": 3,
    "Restaurant": 4,
    "School": 5,
    "Supermarket": 6
}

expected_features = getattr(model, "feature_names_in_", None)

if expected_features is not None:
    if list(expected_features) != features:
        st.error("Model feature names or order do not match the application.")
        st.stop()


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🍃 FOOD WASTE AI")
st.sidebar.caption("Smart Food Management System")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home Dashboard",
        "Food Waste Prediction",
        "Smart Donation",
        "Analytics & Insights",
        "Waste Reduction Tips"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### SYSTEM INFORMATION")
st.sidebar.success("ML Model Loaded")
st.sidebar.caption("Model: Food Waste Regression")
st.sidebar.caption("Framework: Scikit-learn")
st.sidebar.caption("Interface: Streamlit")

st.sidebar.divider()

st.sidebar.caption("AI Food Waste Management System")


# =====================================================
# HOME DASHBOARD
# =====================================================

if page == "Home Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🍽️ AI Food Waste Management</h1>
        <p>
        Intelligent food waste prediction, smart donation
        recommendations and sustainable food management.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("System Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Model", "Active", "Loaded")
    c2.metric("Prediction Type", "Regression")
    c3.metric("Model Features", "20")

    st.divider()

    st.subheader("Explore Dashboard")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
        ### 📊 Food Waste Prediction

        Predict estimated food waste using
        a pretrained Machine Learning regression model.
        """)

        st.info("""
        ### 🤝 Smart Donation Recommendation

        Identify potential donation channels based on
        food type, quantity, expiry and storage details.
        """)

    with col2:

        st.success("""
        ### 📈 Analytics & Insights

        Visualize the latest prediction and
        understand food waste estimates.
        """)

        st.success("""
        ### 🌱 Waste Reduction Tips

        Explore practical recommendations for
        sustainable food management.
        """)

    st.divider()

    st.subheader("About This Project")

    st.write("""
    This project combines Machine Learning and
    rule-based recommendation logic to support
    food waste management and food redistribution.

    The prediction model estimates food waste
    in kilograms per person per day.
    """)

    st.warning(
        "Educational prototype: the original prediction model "
        "was trained using synthetic data. Results are not "
        "validated real-world measurements."
    )


# =====================================================
# FOOD WASTE PREDICTION
# =====================================================

elif page == "Food Waste Prediction":

    st.title("📊 Food Waste Prediction")

    st.write(
        "Enter sector and food management information "
        "to estimate food waste."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sector Information")

            sector = st.selectbox(
                "Select Sector",
                list(sector_encoding.keys())
            )

            month = st.slider("Month", 1, 12, 6)

            is_ramadan = st.selectbox(
                "Ramadan Season",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            is_eid = st.selectbox(
                "Eid Season",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            household_size = st.number_input(
                "Household Size", 1, 11, 4
            )

            income_tier = st.slider(
                "Income Tier", 1, 5, 3
            )

            expat_pct = st.slider(
                "Expat Percentage", 0.30, 0.75, 0.50
            )

            buffet_style = st.selectbox(
                "Buffet Style",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            portion_control = st.selectbox(
                "Portion Control",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            cold_chain_quality = st.slider(
                "Cold Chain Quality", 1, 5, 3
            )

        with col2:

            st.subheader("Food Management Information")

            food_literacy = st.slider(
                "Food Literacy", 1, 5, 3
            )

            temperature_c = st.selectbox(
                "Temperature (°C)",
                [22, 28, 35, 40, 44, 46]
            )

            storage_quality = st.slider(
                "Storage Quality", 1, 5, 3
            )

            purchase_planning = st.slider(
                "Purchase Planning", 1, 5, 3
            )

            delivery_spoilage = st.slider(
                "Delivery Spoilage", 0.0, 1.0, 0.15
            )

            menu_diversity = st.slider(
                "Menu Diversity", 5, 49, 20
            )

            overbuying_idx = st.slider(
                "Overbuying Index", 0.0, 1.0, 0.40
            )

            awareness_campaign = st.selectbox(
                "Awareness Campaign",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            penalty_regime = st.selectbox(
                "Penalty Regime",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

            composting_avail = st.selectbox(
                "Composting Available",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No"
            )

        submit = st.form_submit_button(
            "🔍 Predict Food Waste",
            use_container_width=True
        )

    if submit:

        input_data = {
            "sector_enc": sector_encoding[sector],
            "month": month,
            "is_ramadan": is_ramadan,
            "is_eid": is_eid,
            "household_size": household_size,
            "income_tier": income_tier,
            "expat_pct": expat_pct,
            "buffet_style": buffet_style,
            "portion_control": portion_control,
            "cold_chain_quality": cold_chain_quality,
            "food_literacy": food_literacy,
            "temperature_c": temperature_c,
            "storage_quality": storage_quality,
            "purchase_planning": purchase_planning,
            "delivery_spoilage": delivery_spoilage,
            "menu_diversity": menu_diversity,
            "overbuying_idx": overbuying_idx,
            "awareness_campaign": awareness_campaign,
            "penalty_regime": penalty_regime,
            "composting_avail": composting_avail
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )

        try:

            prediction = float(model.predict(input_df)[0])

            st.session_state["last_prediction"] = prediction
            st.session_state["last_sector"] = sector

            st.divider()

            st.subheader("Prediction Results")

            st.metric(
                "Estimated Food Waste",
                f"{prediction:.3f} kg/person/day"
            )

            st.write("### Input Summary")

            st.dataframe(
                input_df,
                use_container_width=True
            )

            if prediction > 1.5:

                st.warning(
                    "The model estimate is relatively high. "
                    "Review purchasing, storage and portion planning."
                )

            elif prediction >= 0.8:

                st.info(
                    "Review food management practices and "
                    "monitor actual waste records."
                )

            else:

                st.success(
                    "The model estimate is below 0.8 kg/person/day. "
                    "Continue monitoring actual food waste."
                )

            st.caption(
                "This is an estimate from a model trained on "
                "synthetic data, not a verified real-world measurement."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =====================================================
# SMART DONATION RECOMMENDATION
# =====================================================

elif page == "Smart Donation":

    st.title("🤝 Smart Food Donation Center")

    st.write(
        "Find a suitable potential donation channel "
        "for surplus food."
    )

    st.divider()

    st.subheader("📦 Surplus Food Details")

    with st.form("donation_form"):

        col1, col2 = st.columns(2)

        with col1:

            food_type = st.selectbox(
                "Food Category",
                [
                    "Fresh Produce",
                    "Packaged Dry Food",
                    "Cooked Meals",
                    "Dairy Products",
                    "Bakery Products"
                ]
            )

            quantity = st.number_input(
                "Surplus Quantity (kg)",
                min_value=0.0,
                max_value=10000.0,
                value=10.0
            )

            expiry_hours = st.number_input(
                "Hours Remaining Until Expiry",
                min_value=0.0,
                max_value=720.0,
                value=24.0
            )

        with col2:

            location = st.text_input(
                "Location / Area",
                placeholder="Example: Chennai"
            )

            storage_safe = st.selectbox(
                "Storage Condition",
                [
                    "Select condition",
                    "Properly stored",
                    "Not sure",
                    "Improperly stored"
                ]
            )

            food_acceptable = st.checkbox(
                "Food is unexpired and has no visible signs of spoilage"
            )

        submit_donation = st.form_submit_button(
            "✨ Generate Donation Recommendation",
            use_container_width=True
        )

    if submit_donation:

        st.divider()

        st.subheader("🎯 Recommendation Results")

        if quantity <= 0:

            st.error("Enter a surplus quantity greater than zero.")

        elif expiry_hours <= 0:

            st.error("The entered expiry time has passed.")

        elif storage_safe != "Properly stored" or not food_acceptable:

            st.error(
                "Donation is not recommended based on the information "
                "provided. Do not distribute potentially unsafe food."
            )

        else:

            # Food type recommendation

            if food_type == "Fresh Produce":

                channel = (
                    "Community Kitchens / Food Redistribution NGOs"
                )

                reason = (
                    "Potentially suitable for organizations that "
                    "accept fresh produce."
                )

            elif food_type == "Packaged Dry Food":

                channel = (
                    "Registered Food Banks / Community NGOs"
                )

                reason = (
                    "Potentially suitable for accepted, "
                    "unexpired packaged food."
                )

            elif food_type == "Cooked Meals":

                channel = (
                    "Authorized Community Kitchens"
                )

                reason = (
                    "Requires prompt recipient acceptance "
                    "and safe handling."
                )

            elif food_type == "Dairy Products":

                channel = (
                    "Registered Food Banks with Suitable Cold Storage"
                )

                reason = (
                    "Recipient must be able to maintain "
                    "required storage conditions."
                )

            else:

                channel = (
                    "Community Kitchens / Registered Food Banks"
                )

                reason = (
                    "Check packaging, freshness and "
                    "recipient acceptance."
                )

            # Quantity recommendation

            if quantity < 5:

                scale = "Small Surplus"

                action = (
                    "Contact a nearby eligible organization."
                )

            elif quantity <= 20:

                scale = "Medium Surplus"

                action = (
                    "Coordinate with a registered food bank "
                    "or community kitchen."
                )

            else:

                scale = "Large Surplus"

                action = (
                    "Coordinate collection with suitable "
                    "donation organizations."
                )

            # Expiry priority

            if expiry_hours <= 6:

                urgency = "Urgent Coordination"

            elif expiry_hours <= 24:

                urgency = "High Priority"

            else:

                urgency = "Planned Coordination"

            st.success(
                "Potential donation channel identified!"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Surplus Quantity",
                f"{quantity:.1f} kg"
            )

            b.metric(
                "Surplus Level",
                scale
            )

            c.metric(
                "Expiry Remaining",
                f"{expiry_hours:g} hours"
            )

            st.divider()

            st.subheader("🏢 Suggested Donation Channel")

            st.info(channel)

            st.subheader("📍 Donation Information")

            st.write("**Food Category:**", food_type)

            st.write(
                "**Location:**",
                location if location else "Not provided"
            )

            st.write("**Reason:**", reason)

            st.write("**Suggested Action:**", action)

            st.write("**Priority:**", urgency)

            st.divider()

            st.subheader("🌱 Potential Social Benefit")

            st.success(
                "This surplus may help eligible community "
                "organizations if the food meets their "
                "safety and acceptance requirements."
            )

            st.caption(
                "This is a rule-based recommendation, not an ML "
                "prediction or verified partner match. Confirm "
                "local food safety rules, storage, transport and "
                "recipient acceptance before donation."
            )


# =====================================================
# ANALYTICS AND INSIGHTS
# =====================================================

elif page == "Analytics & Insights":

    st.title("📈 Analytics & Insights")

    st.write(
        "Understand the latest food waste prediction "
        "and model information."
    )

    if "last_prediction" not in st.session_state:

        st.info(
            "No prediction available. Visit Food Waste Prediction "
            "to generate your first result."
        )

    else:

        prediction = st.session_state["last_prediction"]

        sector = st.session_state["last_sector"]

        st.subheader("Prediction Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Waste",
            f"{prediction:.3f} kg/person/day"
        )

        c2.metric(
            "Selected Sector",
            sector
        )

        c3.metric(
            "Prediction Method",
            "Regression"
        )

        st.divider()

        st.subheader("Food Waste Estimate")

        chart_data = pd.DataFrame(
            {
                "Metric": ["Predicted Food Waste"],
                "Waste": [prediction]
            }
        ).set_index("Metric")

        st.bar_chart(
            chart_data,
            color="#13866a"
        )

        st.subheader("Prediction Interpretation")

        if prediction > 1.5:

            st.warning(
                "The model estimate is above 1.5 kg/person/day. "
                "Review purchasing, storage and portion planning."
            )

        elif prediction >= 0.8:

            st.info(
                "The model estimate is between 0.8 and 1.5 "
                "kg/person/day. Review food management practices."
            )

        else:

            st.success(
                "The model estimate is below 0.8 kg/person/day. "
                "Continue monitoring actual food waste."
            )

        st.divider()

        st.subheader("Model Information")

        info1, info2 = st.columns(2)

        with info1:

            st.markdown("""
            **Model Type**

            Machine Learning Regression

            **Input Features**

            20 features

            **Prediction Output**

            Estimated kg/person/day
            """)

        with info2:

            st.markdown("""
            **Model Framework**

            Scikit-learn

            **Model Format**

            Pickle / Joblib

            **Training Dataset**

            Synthetic dataset
            """)

        st.caption(
            "This page displays the latest prediction only. "
            "It does not represent historical analytics or "
            "verified real-world trends."
        )


# =====================================================
# FOOD WASTE REDUCTION TIPS
# =====================================================

elif page == "Waste Reduction Tips":

    st.title("🌱 Food Waste Reduction Tips")

    st.write(
        "Explore sector-specific suggestions for "
        "reducing avoidable food waste."
    )

    tips = {

        "Household": [
            "Prepare a weekly shopping list.",
            "Store food according to product requirements.",
            "Use suitable older items before newer purchases.",
            "Plan portions before cooking."
        ],

        "Restaurant": [
            "Monitor customer plate waste.",
            "Review portion sizes.",
            "Improve demand forecasting.",
            "Coordinate safe surplus donation."
        ],

        "Hotel": [
            "Monitor buffet consumption.",
            "Prepare food in suitable batches.",
            "Track unused ingredients.",
            "Coordinate with approved donation partners."
        ],

        "Supermarket": [
            "Monitor stock expiry dates.",
            "Improve inventory rotation.",
            "Plan markdowns where legally appropriate.",
            "Coordinate with registered food banks."
        ],

        "Catering": [
            "Estimate event attendance carefully.",
            "Plan serving quantities.",
            "Track leftover food.",
            "Coordinate with authorized recipients."
        ],

        "School": [
            "Plan meals based on attendance.",
            "Monitor plate waste.",
            "Teach food awareness.",
            "Review meal portion planning."
        ],

        "Hospital": [
            "Plan meals according to approved dietary requirements.",
            "Monitor untouched meals.",
            "Follow strict food safety procedures.",
            "Use approved waste reduction processes."
        ]
    }

    selected_sector = st.selectbox(
        "Select Sector",
        list(tips.keys())
    )

    st.subheader(
        f"Suggestions for {selected_sector}"
    )

    for index, tip in enumerate(
        tips[selected_sector],
        start=1
    ):

        st.write(f"**{index}.** {tip}")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Based Food Waste Prediction and Smart Donation "
    "Recommendation System | Python | Scikit-learn | Streamlit"
)
