import streamlit as st
import pandas as pd
import pickle
from io import StringIO

# Page configuration: must be the first Streamlit command
st.set_page_config(
    page_title="🔧 Sensor Fault Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles for a more professional look
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #0052cc;
        color: white;
        border-radius: 5px;
        height: 2.5em;
        width: 100%;
    }
    .stDownloadButton>button {
        background-color: #0073e6;
        color: white;
        border-radius: 5px;
        height: 2.5em;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs and instructions
with st.sidebar:
    st.header("Upload & Instructions")
    st.markdown(
        "1. Prepare a CSV file with 50 sensor columns (sensor_00 to sensor_49)."
        "\n2. Upload the file below."
        "\n3. Click **Predict** to generate fault detection results."
    )
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"], help="File must contain exactly 50 features."
    )

# Load model artifacts with caching
@st.cache_resource
def load_artifacts():
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("artifacts/pca.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, pca, model

scaler, pca, model = load_artifacts()

# Main area
st.title("Sensor Fault Detection Dashboard")
st.markdown(
    "Use the sidebar to upload sensor data and obtain predictive insights after preprocessing."
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 50:
            st.error("Error: CSV must have exactly 50 columns (sensor_00 to sensor_49).")
        else:
            st.success("File uploaded successfully.")
            st.dataframe(df.head(), use_container_width=True)

            # Predict button
            if st.button("🛰️ Predict Faults"):
                with st.spinner("Processing data and generating predictions..."):
                    # Scale and transform
                    X_scaled = scaler.transform(df)
                    X_pca = pca.transform(X_scaled)
                    # Predict
                    preds = model.predict(X_pca)
                    df_results = df.copy()
                    df_results['Fault Prediction'] = preds

                st.success("Prediction complete.")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Prediction Results Preview")
                    st.dataframe(df_results[['Fault Prediction']].head(), use_container_width=True)
                with col2:
                    st.subheader("Download")
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results",
                        data=csv,
                        file_name="sensor_fault_predictions.csv",
                        mime="text/csv"
                    )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
else:
    st.info("Awaiting CSV file upload in the sidebar.")

# Footer
st.markdown("---")
st.caption("© 2025 Sensor Analytics Inc. Built with Streamlit.")
