import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go

#  PAGE CONFIG 
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD DATA & MODEL
@st.cache_data
def load_data():
    df = pd.read_csv('processed_customers.csv')
    return df

@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    with open('model_features.json', 'r') as f:
        features = json.load(f)
    return model, features

df = load_data()
model, feature_cols = load_model()

# SIDEBAR
st.sidebar.image("https://img.icons8.com/fluency/96/signal.png", width=60)
st.sidebar.title("Telecom Churn Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Executive Dashboard",
        "📈 Interactive Visualizations",
        "👥 Segment Explorer",
        "🔮 Churn Predictor",
        "💡 Retention Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit")


# PAGE 1: EXECUTIVE DASHBOARD 
if page == "📊 Executive Dashboard":

    st.title("📊 Executive Dashboard")
    st.markdown("High-level overview of customer churn and revenue metrics.")
    st.markdown("---")

    # KPI Metrics
    total_customers = len(df)
    churn_rate = df['Churn'].mean() * 100
    avg_revenue = df['MonthlyCharges'].mean()
    avg_clv = df['CLV'].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        st.metric("Avg Monthly Revenue", f"${avg_revenue:.2f}")
    with col4:
        st.metric("Avg Customer CLV", f"${avg_clv:.0f}")

    st.markdown("---")

    # Segment Distribution
    st.subheader("Customer Segment Distribution")
    seg_counts = df['SegmentName'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']

    fig = px.pie(
        seg_counts, names='Segment', values='Count',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hole=0.4
    )
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)



# PAGE 2: INTERACTIVE VISUALIZATIONS
elif page == "📈 Interactive Visualizations":

    st.title("📈 Interactive Visualizations")
    st.markdown("---")

    # Churn by Contract Type
    st.subheader("Churn by Contract Type")
    contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
    contract_churn['ChurnRate'] = contract_churn['Churn'] * 100

    fig1 = px.bar(
        contract_churn, x='Contract', y='ChurnRate',
        color='Contract', text_auto='.1f',
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'ChurnRate': 'Churn Rate (%)'}
    )
    fig1.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Monthly Charges vs Churn
    st.subheader("Monthly Charges Distribution by Churn")
    fig2 = px.histogram(
        df, x='MonthlyCharges', color='Churn',
        barmode='overlay', nbins=40,
        color_discrete_map={0: '#00c853', 1: '#ff4b4b'},
        labels={'Churn': 'Churned'}
    )
    fig2.update_layout(template='plotly_dark')
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Tenure vs Churn
    st.subheader("Tenure Distribution by Churn")
    fig3 = px.histogram(
        df, x='tenure', color='Churn',
        barmode='overlay', nbins=40,
        color_discrete_map={0: '#00c853', 1: '#ff4b4b'},
        labels={'Churn': 'Churned'}
    )
    fig3.update_layout(template='plotly_dark')
    st.plotly_chart(fig3, use_container_width=True)



# PAGE 3: SEGMENT EXPLORER
elif page == "👥 Segment Explorer":

    st.title("👥 Customer Segment Explorer")
    st.markdown("Select a segment to explore its profile.")
    st.markdown("---")

    segments = df['SegmentName'].unique().tolist()
    selected_segment = st.selectbox("Choose a Segment", segments)

    seg_df = df[df['SegmentName'] == selected_segment]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Customers", f"{len(seg_df):,}")
    with col2:
        st.metric("Churn Rate", f"{seg_df['Churn'].mean()*100:.1f}%")
    with col3:
        st.metric("Avg Monthly Charges", f"${seg_df['MonthlyCharges'].mean():.2f}")
    with col4:
        st.metric("Avg Tenure", f"{seg_df['tenure'].mean():.0f} months")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Contract Type Breakdown")
        contract_counts = seg_df['Contract'].value_counts().reset_index()
        contract_counts.columns = ['Contract', 'Count']
        fig4 = px.pie(
            contract_counts, names='Contract', values='Count',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig4.update_layout(template='plotly_dark')
        st.plotly_chart(fig4, use_container_width=True)

    with col6:
        st.subheader("Payment Method Breakdown")
        payment_counts = seg_df['PaymentMethod'].value_counts().reset_index()
        payment_counts.columns = ['PaymentMethod', 'Count']
        fig5 = px.bar(
            payment_counts, x='PaymentMethod', y='Count',
            color='PaymentMethod',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig5.update_layout(template='plotly_dark', showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)



# PAGE 4: CHURN PREDICTOR 
elif page == "🔮 Churn Predictor":

    st.title("🔮 Churn Predictor")
    st.markdown("Enter customer details to predict churn probability.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    st.markdown("---")
    st.subheader("Services Subscribed")

    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        online_security = st.checkbox("Online Security")
        online_backup = st.checkbox("Online Backup")
    with scol2:
        device_protection = st.checkbox("Device Protection")
        tech_support = st.checkbox("Tech Support")
    with scol3:
        streaming_tv = st.checkbox("Streaming TV")
        streaming_movies = st.checkbox("Streaming Movies")
    with scol4:
        multiple_lines = st.checkbox("Multiple Lines")

    st.markdown("---")

    if st.button("🔮 Predict Churn", use_container_width=True):

        # Build input dict
        input_dict = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 0, 'Dependents': 0,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'OnlineSecurity': 1 if online_security else 0,
            'OnlineBackup': 1 if online_backup else 0,
            'DeviceProtection': 1 if device_protection else 0,
            'TechSupport': 1 if tech_support else 0,
            'StreamingTV': 1 if streaming_tv else 0,
            'StreamingMovies': 1 if streaming_movies else 0,
            'MultipleLines': 1 if multiple_lines else 0,
            'CLV': tenure * monthly_charges,
            'ServiceCount': sum([phone_service=="Yes", multiple_lines, online_security,
                                 online_backup, device_protection, tech_support,
                                 streaming_tv, streaming_movies]),
            'EngagementScore': 0.5,
            'ContractRisk': {"Month-to-month": 2, "One year": 1, "Two year": 0}[contract],
            'SupportDependency': sum([tech_support, online_backup, device_protection]),
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        }

        # Align with model features
        input_df = pd.DataFrame([input_dict])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        prob = model.predict_proba(input_df)[0][1]
        risk = "🔴 High Risk" if prob >= 0.6 else "🟡 Medium Risk" if prob >= 0.35 else "🟢 Low Risk"

        st.subheader("Prediction Result")
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.markdown(f"### {risk}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff4b4b" if prob >= 0.6 else "#ffa500" if prob >= 0.35 else "#00c853"},
                'steps': [
                    {'range': [0, 35], 'color': '#1a2a1a'},
                    {'range': [35, 60], 'color': '#2a2a1a'},
                    {'range': [60, 100], 'color': '#2a1a1a'}
                ]
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        


# PAGE 5: RETENTION RECOMMENDATIONS
elif page == "💡 Retention Recommendations":

    st.title("💡 Retention Recommendations")
    st.markdown("Business actions based on customer risk profiles.")
    st.markdown("---")

    recommendations = {
        "At Risk Newcomers": {
            "risk": "🔴 High Risk",
            "finding": "50% churn rate among new customers",
            "actions": [
                "Improve onboarding experience",
                "Offer early loyalty discount after 3 months",
                "Assign dedicated support contact"
            ]
        },
        "Month-to-Month Users": {
            "risk": "🔴 High Risk",
            "finding": "#1 churn driver — no long term commitment",
            "actions": [
                "Incentivize switch to annual contract",
                "Offer 1 month free for upgrading plan",
                "Send personalized retention offers"
            ]
        },
        "Fiber Optic Users": {
            "risk": "🟡 Medium Risk",
            "finding": "High churn despite premium plan",
            "actions": [
                "Investigate service quality issues",
                "Offer free tech support for 3 months",
                "Send satisfaction surveys"
            ]
        },
        "Electronic Check Users": {
            "risk": "🟡 Medium Risk",
            "finding": "Higher churn than auto-pay users",
            "actions": [
                "Nudge toward auto-pay with small discount",
                "Simplify payment switching process",
                "Send reminders before billing date"
            ]
        },
        "Loyal High Value": {
            "risk": "🟢 Low Risk",
            "finding": "Only 14% churn, highest CLV",
            "actions": [
                "Reward with exclusive loyalty program",
                "Offer early access to new features",
                "Protect at all costs — priority support"
            ]
        }
    }

    for customer_type, info in recommendations.items():
        with st.expander(f"{info['risk']} — {customer_type}"):
            st.markdown(f"**Key Finding:** {info['finding']}")
            st.markdown("**Recommended Actions:**")
            for action in info['actions']:
                st.markdown(f"- ✅ {action}")

    st.markdown("---")
    st.subheader("Churn Rate by Segment")

    seg_churn = df.groupby('SegmentName')['Churn'].mean().reset_index()
    seg_churn['ChurnRate'] = seg_churn['Churn'] * 100

    fig = px.bar(
        seg_churn, x='SegmentName', y='ChurnRate',
        color='SegmentName', text_auto='.1f',
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={'ChurnRate': 'Churn Rate (%)', 'SegmentName': 'Segment'}
    )
    fig.update_layout(template='plotly_dark', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)