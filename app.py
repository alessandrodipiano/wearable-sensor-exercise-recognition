import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Physical Therapy Exercise Evaluation",
    layout="wide"
)

st.title("Physical Therapy Exercise Evaluation System")

# --------------------------------------------------
# Load data
# --------------------------------------------------

DATA_PATH = "data/processed_data.csv"

df = pd.read_csv(DATA_PATH)

st.sidebar.header("Session Selection")

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

subject = st.sidebar.selectbox(
    "Select subject",
    sorted(df["subject"].unique())
)

exercise = st.sidebar.selectbox(
    "Select exercise",
    sorted(df["exercise"].unique())
)

filtered_df = df[
    (df["subject"] == subject) &
    (df["exercise"] == exercise)
]

# --------------------------------------------------
# Main dashboard
# --------------------------------------------------

st.subheader("Selected Session")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Subject", subject)

with col2:
    st.metric("Exercise", exercise)

with col3:
    st.metric("Number of samples", len(filtered_df))

st.divider()

# --------------------------------------------------
# Signal visualization
# --------------------------------------------------

st.subheader("Sensor Magnitude Visualization")

magnitude_cols = [
    col for col in df.columns
    if "acc_mag" in col or "gyr_mag" in col or "mag_mag" in col
]

selected_signal = st.selectbox(
    "Select signal to visualize",
    magnitude_cols
)

fig = px.line(
    filtered_df,
    y=selected_signal,
    title=f"{selected_signal} over time"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Label summary
# --------------------------------------------------

st.subheader("Execution Quality Summary")

if "label" in df.columns:
    label_counts = filtered_df["label"].value_counts().reset_index()
    label_counts.columns = ["Execution Type", "Count"]

    fig_labels = px.bar(
        label_counts,
        x="Execution Type",
        y="Count",
        title="Distribution of Exercise Execution Types"
    )

    st.plotly_chart(fig_labels, use_container_width=True)

    majority_label = filtered_df["label"].mode()[0]

    st.subheader("System Feedback")

    if majority_label == "correct":
        st.success("Good execution.")
    elif majority_label == "fast":
        st.warning("Movement appears too fast. The patient should slow down.")
    elif majority_label == "low_amplitude":
        st.warning("Movement amplitude appears too low. The patient should increase range of motion.")
    else:
        st.info(f"Detected execution type: {majority_label}")

else:
    st.info("No label column found in the dataset.")

# --------------------------------------------------
# Raw data preview
# --------------------------------------------------

with st.expander("Show raw filtered data"):
    st.dataframe(filtered_df.head(100))