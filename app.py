
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD EXISTING MODEL
# ==========================================

MODEL_PATH = Path(__file__).parent / "model_food_waste.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = None
model_error = None

try:
    if MODEL_PATH.exists():
        model = load_model(str(MODEL_PATH))
    else:
        model_error = "model_food_waste.pkl was not found."
except Exception as e:
    model_error = str(e)

# ==========================================
# 3. MODEL FEATURES
# ==========================================

SECTOR_MAPPING = {
    "Catering": 0,
    "Hospital": 1,
    "Household": 2,
    "Hotel": 3,
    "Restaurant": 4,
    "School": 5,
    "Supermarket": 6
}

FEATURES = [
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

# ==========================================
# 4. SESSION STATE
# ==========================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_sector" not in st.session_state:
    st.session_state.prediction_sector = None

if "prediction_inputs" not in st.session_state:
    st.session_state.prediction_inputs = None

# ==========================================
# 5. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1020;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #151B35;
    border-right: 1px solid #34466D;
}

[data-testid="stSidebar"] * {
    color: #F1F5F9;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(120deg, #202B58, #5635A8, #087E9B);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid #5264A0;
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #E0E7FF;
    font-size: 16px;
}

.info-card {
    background: #17213B;
    border: 1px solid #34466D;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #A5B4FC !important;
}

.info-card p {
    color: #CBD5E1;
}

div[data-testid="stMetric"] {
    background: #182342;
    border: 1px solid #34466D;
    padding: 18px;
    border-radius: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #A5B4FC;
}

div[data-testid="stMetricValue"] {
    color: white;
}

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #7047D7, #347FCE);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #5631B5;
    color: white;
}

h1, h2, h3 {
    color: #C4B5FD;
}

hr {
    border-color: #34466D;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 6. REUSABLE FUNCTIONS
# ==========================================

def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_card(title, description):
    st.markdown(
        f"""
        <div class="info-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# 7. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.title("🌍 FOODWISE AI")
    st.caption("SMART FOOD SECURITY PLATFORM")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Home Dashboard",
            "Food Waste Prediction",
            "Smart Donation",
            "Analytics & Insights",
            "Waste Reduction Tips",
            "ML Model Information"
        ]
    )

    st.divider()

    if model is not None:
        st.success("Model Connected")
    else:
        st.error("Model Not Connected")

    st.caption("Version 2.1")

# ==========================================
# 8. HOME DASHBOARD
# ==========================================

if page == "Home Dashboard":

    show_hero(
        "🌍 FOODWISE AI",
        "AI-Based Food Waste Prediction and Smart Donation Recommendation System"
    )

    st.subheader("Welcome to FoodWise AI")

    st.write(
        "An educational machine learning application that estimates "
        "food waste and provides rule-based donation suggestions."
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric("ML Algorithm", "Ridge Regression")
    c2.metric("Input Features", "20")
    c3.metric("Model Status", "Online" if model is not None else "Offline")

    st.subheader("Platform Modules")

    col1, col2 = st.columns(2)

    with col1:
        show_card(
            "🤖 Food Waste Prediction",
            "Estimate food waste using your trained regression model."
        )
        show_card(
            "📊 Analytics & Insights",
            "View the latest prediction and input details."
        )
        show_card(
            "🧠 ML Model Information",
            "Explore Ridge Regression and its model configuration."
        )

    with col2:
        show_card(
            "🎁 Smart Donation",
            "Receive possible donation-channel suggestions using predefined rules."
        )
        show_card(
            "🌱 Waste Reduction Tips",
            "Explore practical food waste reduction strategies."
        )

    if model is None:
        st.error("Model loading failed: " + str(model_error))

# ==========================================
# 9. FOOD WASTE PREDICTION
# ==========================================

elif page == "Food Waste Prediction":

    show_hero(
        "🤖 FOOD WASTE PREDICTION",
        "Enter the required information to generate a prediction."
    )

    if model is None:
        st.error("Model could not be loaded: " + str(model_error))
        st.stop()

    with st.form("food_prediction_form"):

        st.subheader("Sector Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            sector = st.selectbox(
                "Sector",
                list(SECTOR_MAPPING.keys())
            )

        with c2:
            month = st.selectbox(
                "Month",
                list(range(1, 13)),
                index=datetime.now().month - 1
            )

        with c3:
            household_size = st.number_input(
                "Household Size",
                min_value=1,
                max_value=100,
                value=4
            )

        st.subheader("Seasonal Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            ramadan = st.selectbox("Ramadan", ["No", "Yes"])

        with c2:
            eid = st.selectbox("Eid", ["No", "Yes"])

        with c3:
            income_tier = st.slider("Income Tier", 1, 5, 3)

        st.subheader("Food Management Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            expat_pct = st.slider("Expat Population (%)", 0, 100, 30)

        with c2:
            buffet = st.selectbox("Buffet Style", ["No", "Yes"])

        with c3:
            portion_control = st.slider(
                "Portion Control", 0.0, 1.0, 0.5
            )

        st.subheader("Storage and Quality")

        c1, c2, c3 = st.columns(3)

        with c1:
            cold_chain_quality = st.slider(
                "Cold Chain Quality", 0.0, 1.0, 0.7
            )

        with c2:
            food_literacy = st.slider(
                "Food Literacy", 0.0, 1.0, 0.6
            )

        with c3:
            temperature_c = st.slider(
                "Temperature (°C)", 0, 50, 30
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            storage_quality = st.slider(
                "Storage Quality", 0.0, 1.0, 0.7
            )

        with c2:
            purchase_planning = st.slider(
                "Purchase Planning", 0.0, 1.0, 0.6
            )

        with c3:
            delivery_spoilage = st.slider(
                "Delivery Spoilage", 0.0, 1.0, 0.2
            )

        st.subheader("Additional Factors")

        c1, c2, c3 = st.columns(3)

        with c1:
            menu_diversity = st.slider(
                "Menu Diversity", 0.0, 1.0, 0.5
            )

        with c2:
            overbuying_idx = st.slider(
                "Overbuying Index", 0.0, 1.0, 0.3
            )

        with c3:
            awareness_campaign = st.slider(
                "Awareness Campaign", 0.0, 1.0, 0.5
            )

        c1, c2 = st.columns(2)

        with c1:
            penalty_regime = st.slider(
                "Penalty Regime", 0.0, 1.0, 0.5
            )

        with c2:
            composting_avail = st.slider(
                "Composting Availability", 0.0, 1.0, 0.5
            )

        st.divider()

        submit = st.form_submit_button(
            "🚀 PREDICT FOOD WASTE",
            use_container_width=True
        )

    if submit:

        input_data = {
            "sector_enc": SECTOR_MAPPING[sector],
            "month": month,
            "is_ramadan": int(ramadan == "Yes"),
            "is_eid": int(eid == "Yes"),
            "household_size": household_size,
            "income_tier": income_tier,
            "expat_pct": expat_pct,
            "buffet_style": int(buffet == "Yes"),
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

        input_df = pd.DataFrame([input_data], columns=FEATURES)

        try:
            prediction = float(model.predict(input_df)[0])

            if not pd.notna(prediction):
                raise ValueError("The model returned an invalid prediction.")

            st.session_state.prediction = prediction
            st.session_state.prediction_sector = sector
            st.session_state.prediction_inputs = input_data

            st.success("Prediction completed successfully!")

            st.subheader("Prediction Result")

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted Food Waste",
                f"{prediction:.4f} kg/person/day"
            )

            c2.metric("Selected Sector", sector)

            st.info(
                "This is an educational model estimate, not a measurement of actual food waste."
            )

            with st.expander("View all 20 input features"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error("Prediction error: " + str(e))

# ==========================================
# 10. SMART DONATION
# ==========================================

elif page == "Smart Donation":

    show_hero(
        "🎁 SMART DONATION",
        "Rule-based food donation suggestions"
    )

    st.write(
        "Enter surplus food details to receive possible donation-channel suggestions."
    )

    with st.form("donation_form"):

        food_type = st.selectbox(
            "Food Type",
            [
                "Cooked Food",
                "Vegetables",
                "Fruits",
                "Bakery Items",
                "Packaged Food",
                "Dairy Products"
            ]
        )

        quantity = st.number_input(
            "Quantity (kg)",
            min_value=0.1,
            value=10.0
        )

        expiry_hours = st.number_input(
            "Estimated Hours Until Expiry",
            min_value=0.0,
            value=12.0
        )

        location = st.text_input("Location")

        storage_safe = st.checkbox(
            "Food is stored safely and is suitable for food safety assessment"
        )

        submit_donation = st.form_submit_button(
            "GET DONATION SUGGESTIONS",
            use_container_width=True
        )

    if submit_donation:

        if not storage_safe:
            st.warning(
                "Food safety must be assessed before donation. Do not donate food unless it is confirmed safe."
            )

        elif expiry_hours <= 0:
            st.error(
                "The entered expiry time has passed. Do not recommend this food for donation."
            )

        else:

            if quantity < 5:
                surplus = "Small Surplus"
            elif quantity < 25:
                surplus = "Medium Surplus"
            else:
                surplus = "Large Surplus"

            if expiry_hours <= 4:
                priority = "High Priority"
            elif expiry_hours <= 12:
                priority = "Medium Priority"
            else:
                priority = "Normal Priority"

            channels = {
                "Cooked Food": "Registered food redistribution organizations",
                "Vegetables": "Community food banks",
                "Fruits": "Community food distribution programs",
                "Bakery Items": "Registered charitable organizations",
                "Packaged Food": "Food banks and donation organizations",
                "Dairy Products": "Approved organizations with suitable cold storage"
            }

            st.success("Suggestions generated!")

            c1, c2, c3 = st.columns(3)

            c1.metric("Quantity", f"{quantity:.1f} kg")
            c2.metric("Surplus", surplus)
            c3.metric("Priority", priority)

            show_card(
                "Suggested Donation Channel",
                channels[food_type]
            )

            st.write("Location:", location if location else "Not entered")

            st.info(
                "These are rule-based suggestions. The application does not "
                "verify food safety or locate actual available donation partners."
            )

# ==========================================
# 11. ANALYTICS
# ==========================================

elif page == "Analytics & Insights":

    show_hero(
        "📊 ANALYTICS & INSIGHTS",
        "Analysis of your latest model prediction"
    )

    if st.session_state.prediction is None:

        st.info(
            "Generate a prediction first using the Food Waste Prediction page."
        )

    else:

        prediction = st.session_state.prediction
        sector = st.session_state.prediction_sector

        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Food Waste",
            f"{prediction:.4f} kg/person/day"
        )

        c2.metric("Selected Sector", sector)

        st.subheader("Prediction Chart")

        chart_data = pd.DataFrame({
            "Prediction": [prediction]
        })

        st.bar_chart(chart_data)

        st.subheader("Input Feature Analysis")

        inputs = st.session_state.prediction_inputs

        if inputs is not None:
            st.dataframe(
                pd.DataFrame(
                    list(inputs.items()),
                    columns=["Feature", "Value"]
                ),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "This represents the latest prediction only, not historical trends."
        )

# ==========================================
# 12. WASTE REDUCTION TIPS
# ==========================================

elif page == "Waste Reduction Tips":

    show_hero(
        "🌱 WASTE REDUCTION TIPS",
        "Practical food management strategies"
    )

    tips = {
        "Household": [
            "Prepare a shopping list.",
            "Plan meals before purchasing.",
            "Store food correctly.",
            "Check expiry dates.",
            "Avoid unnecessary purchases."
        ],
        "Restaurant": [
            "Monitor kitchen waste.",
            "Plan portion sizes.",
            "Improve inventory management.",
            "Estimate customer demand.",
            "Monitor surplus food."
        ],
        "Hotel": [
            "Monitor buffet consumption.",
            "Prepare food according to demand.",
            "Improve cold storage.",
            "Measure food waste.",
            "Coordinate approved donation."
        ],
        "Supermarket": [
            "Use stock rotation.",
            "Monitor product dates.",
            "Improve inventory planning.",
            "Maintain suitable storage.",
            "Coordinate food redistribution."
        ],
        "Hospital": [
            "Plan meals according to dietary requirements.",
            "Monitor kitchen preparation.",
            "Maintain food safety.",
            "Measure surplus food.",
            "Improve inventory control."
        ],
        "School": [
            "Plan meals based on attendance.",
            "Encourage responsible portions.",
            "Educate students.",
            "Monitor cafeteria surplus.",
            "Improve meal planning."
        ],
        "Catering": [
            "Estimate guest numbers.",
            "Plan food quantities.",
            "Monitor leftovers.",
            "Improve food storage.",
            "Coordinate approved donation."
        ]
    }

    sector_choice = st.selectbox(
        "Select Sector",
        list(tips.keys())
    )

    st.subheader("Recommended Actions")

    for index, tip in enumerate(tips[sector_choice], start=1):
        show_card(f"Action {index}", tip)

# ==========================================
# 13. ML MODEL INFORMATION
# ==========================================

elif page == "ML Model Information":

    show_hero(
        "🧠 ML MODEL INFORMATION",
        "Technical information about your deployed model"
    )

    if model is None:
        st.error("Model is unavailable: " + str(model_error))
        st.stop()

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Ridge Regression")
    c2.metric("Regularization", "L2")
    c3.metric("Alpha", str(model.get_params().get("alpha", "N/A")))

    st.subheader("Model Details")

    show_card(
        "Ridge Regression",
        "A supervised regression algorithm that uses L2 regularization "
        "to penalize large model coefficients."
    )

    st.write("**Model file:** model_food_waste.pkl")
    st.write("**Target:** waste_kg_person_day")
    st.write("**Input feature count:** 20")

    st.subheader("Feature Names")

    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = FEATURES

    st.dataframe(
        pd.DataFrame({
            "No.": range(1, len(feature_names) + 1),
            "Feature": feature_names
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Model Parameters")

    parameters = model.get_params()

    st.dataframe(
        pd.DataFrame(
            list(parameters.items()),
            columns=["Parameter", "Value"]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The supplied model was trained using synthetic data. "
        "Its predictions are educational estimates and have not been "
        "validated for real-world food waste decisions."
    )

# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94A3B8; padding:15px;">
        <h3>🌍 FOODWISE AI</h3>
        AI-Based Food Waste Prediction and Smart Donation Recommendation System
        <br><br>
        Python | Pandas | Scikit-learn | Joblib | Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
