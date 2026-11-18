
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Food Waste Prediction",
    page_icon="🍽️",
    layout="wide"
)

# -------------------------------
# MODEL LOADING
# -------------------------------
MODEL_PATH = Path(__file__).parent / "model_food_waste.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🍽️ AI-Based Food Waste Prediction")
st.write(
    "Predict food waste using a pretrained "
    "Machine Learning regression model."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Check model_food_waste.pkl")
    st.stop()

try:
    model = load_model()
    st.success("Food Waste ML Model Loaded Successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# -------------------------------
# MODEL FEATURES
# -------------------------------
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

# Verify the saved model's expected feature order
expected_features = getattr(model, "feature_names_in_", None)

if expected_features is not None:
    if list(expected_features) != features:
        st.error(
            "Model feature names or order do not match "
            "the application. Please verify the downloaded model."
        )
        st.stop()

# -------------------------------
# INPUT FORM
# -------------------------------
st.header("📋 Enter Food Waste Information")

with st.form("food_waste_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sector Information")

        sector = st.selectbox(
            "Select Sector",
            [
                "Catering",
                "Hospital",
                "Household",
                "Hotel",
                "Restaurant",
                "School",
                "Supermarket"
            ]
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
            "Household Size",
            min_value=1,
            max_value=11,
            value=4
        )

        income_tier = st.slider(
            "Income Tier",
            1, 5, 3
        )

        expat_pct = st.slider(
            "Expat Percentage",
            0.30, 0.75, 0.50
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
            "Cold Chain Quality",
            1, 5, 3
        )

    with col2:
        st.subheader("Food Management Information")

        food_literacy = st.slider(
            "Food Literacy",
            1, 5, 3
        )

        temperature_c = st.selectbox(
            "Temperature (°C)",
            [22, 28, 35, 40, 44, 46]
        )

        storage_quality = st.slider(
            "Storage Quality",
            1, 5, 3
        )

        purchase_planning = st.slider(
            "Purchase Planning",
            1, 5, 3
        )

        delivery_spoilage = st.slider(
            "Delivery Spoilage",
            0.0, 1.0, 0.15
        )

        menu_diversity = st.slider(
            "Menu Diversity",
            5, 49, 20
        )

        overbuying_idx = st.slider(
            "Overbuying Index",
            0.0, 1.0, 0.40
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

# -------------------------------
# PREDICTION
# -------------------------------
if submit:

    sector_encoding = {
        "Catering": 0,
        "Hospital": 1,
        "Household": 2,
        "Hotel": 3,
        "Restaurant": 4,
        "School": 5,
        "Supermarket": 6
    }

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
        prediction = model.predict(input_df)[0]

        st.divider()
        st.header("📊 Prediction Results")

        st.metric(
            label="Predicted Food Waste",
            value=f"{prediction:.3f} kg/person/day"
        )

        st.write("### Input Summary")
        st.dataframe(input_df, use_container_width=True)

        st.info(
            "Prediction generated using the downloaded "
            "pretrained Food Waste Regression model."
        )

        st.caption(
            "Educational prototype: the original model "
            "was trained on synthetic data."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.caption(
    "AI-Based Food Waste Prediction System | "
    "Hugging Face + Scikit-learn + Streamlit"
)
