import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Multi-Domain Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ BANNER ------------------
st.image("assets/banner.png", width=700)
st.markdown("<h1 style='text-align:center; color: #4B8BBE;'>Multi-Domain Churn Prediction & Retention</h1>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Predict", "Retention Strategy", "Alerts", "Visualizations"])

# ------------------ UTILS ------------------
def color_risk(risk):
    if str(risk).lower() in ["high", "critical"]:
        return "🔴 ⚠️ High"
    elif str(risk).lower() == "medium":
        return "🟠 ⚠️ Medium"
    else:
        return "🟢 ✅ Low"

def download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

domains = ["Telecom", "Banking", "E-commerce"]

# ------------------ HOME ------------------
if page == "Home":
    st.markdown("### Welcome! 👋")
    st.markdown("""
    This dashboard allows you to:
    - Predict customer churn for **Telecom, Banking, E-commerce**.
    - Generate **Retention Strategies**.
    - Create **Alerts** with risk levels.
    - Visualize **risk distributions** and alert counts per domain.
    """)

# ------------------ UPLOAD & PREDICT ------------------
elif page == "Upload & Predict":
    st.markdown("### Upload CSVs for Prediction")
    uploaded_files = {}
    for domain in domains:
        uploaded_file = st.file_uploader(f"{domain} CSV", type="csv", key=f"{domain}_file")
        if uploaded_file:
            uploaded_files[domain] = pd.read_csv(uploaded_file)
            st.success(f"{domain} file loaded! Shape: {uploaded_files[domain].shape}")
            st.dataframe(uploaded_files[domain].head())

    if st.button("Run Prediction for All Domains"):
        st.info("Running predictions...")
        predicted_files = {}
        for domain, df in uploaded_files.items():
            df['Churn_Probability'] = 0.5  # Placeholder for model prediction
            df['Churn_Prediction'] = df['Churn_Probability'].apply(lambda x: 1 if x > 0.45 else 0)
            st.success(f"{domain} prediction completed!")
            st.dataframe(df.head())
            st.markdown(download_link(df, f"predicted_{domain}.csv"), unsafe_allow_html=True)
            predicted_files[domain] = df

# ------------------ RETENTION STRATEGY ------------------
elif page == "Retention Strategy":
    st.markdown("### Generate Retention Strategy")
    retention_files = {}
    for domain in domains:
        uploaded_file = st.file_uploader(f"Upload predicted {domain} CSV", type="csv", key=f"{domain}_pred_file")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['risk_level'] = df['Churn_Probability'].apply(lambda x: "High" if x>0.7 else ("Medium" if x>0.45 else "Low"))
            df['retention_action'] = df['risk_level'].apply(lambda x: "Call Customer" if x=="High" else ("Email Offer" if x=="Medium" else "No Action"))
            df['risk_level_display'] = df['risk_level'].apply(color_risk)
            st.success(f"{domain} retention strategy generated!")
            st.dataframe(df[['risk_level_display','retention_action']])
            st.markdown(download_link(df, f"retention_{domain}.csv"), unsafe_allow_html=True)
            retention_files[domain] = df

# ------------------ ALERTS ------------------
elif page == "Alerts":
    st.markdown("### Generate Alerts")
    alert_files = {}
    for domain in domains:
        uploaded_file = st.file_uploader(f"Upload retention {domain} CSV", type="csv", key=f"{domain}_ret_file")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['Alert_Message'] = df.apply(
                lambda row: f"⚠️ ALERT: Customer at {row['risk_level']} risk. Action: {row['retention_action']}"
                if row['risk_level'] == "High" else (
                    f"⚠️ CAUTION: Customer at {row['risk_level']} risk. Action: {row['retention_action']}"
                    if row['risk_level']=="Medium" else
                    f"✅ Customer at {row['risk_level']} risk. No immediate action needed."
                ),
                axis=1
            )
            st.success(f"{domain} alerts generated!")
            st.dataframe(df[['risk_level_display','retention_action','Alert_Message']])
            st.markdown(download_link(df, f"alerts_{domain}.csv"), unsafe_allow_html=True)
            alert_files[domain] = df

# ------------------ VISUALIZATIONS ------------------
elif page == "Visualizations":
    st.markdown("### Risk Distribution & Alerts Visualization")
    for domain in domains:
        uploaded_file = st.file_uploader(f"Upload alerts CSV for {domain}", type="csv", key=f"{domain}_alerts_file")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            st.markdown(f"#### {domain} - Risk Distribution")
            risk_counts = df['risk_level'].value_counts()
            fig1, ax1 = plt.subplots()
            colors = ['#ff4c4c', '#ffa500', '#4CAF50']
            ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
            ax1.axis('equal')
            st.pyplot(fig1)

            st.markdown(f"#### {domain} - Alerts vs Total Customers")
            fig2, ax2 = plt.subplots()
            sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=colors, ax=ax2)
            ax2.set_ylabel("Number of Customers")
            st.pyplot(fig2)
