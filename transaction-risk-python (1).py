import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="Transaction Risk Scoring App",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-medium { color: #fd7e14; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    
    .narrative-box {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TransactionRiskScorer:
    def __init__(self):
        self.high_risk_countries = ['AF', 'IR', 'KP', 'SY', 'MM', 'YE', 'SO', 'LY']
        self.high_risk_categories = ['gambling', 'crypto', 'adult', 'pharmaceuticals', 'weapons']
        
    def calculate_risk_score(self, transaction):
        """Calculate risk score based on business rules"""
        score = 10  # Start score at 10
        
        # Sanctioned party flag check - immediate max score
        if transaction.get('sanctioned_party_flag', 0) == 1:
            return 100
        
        # Amount-based risk
        amount = float(transaction.get('amount_usd', 0))
        if amount > 100000:
            score += 30
        elif amount > 50000:
            score += 20
        elif amount > 10000:
            score += 10
        elif amount > 1000:
            score += 5
        
        # Country corridor risk
        sender_country = transaction.get('sender_country', '').upper()
        receiver_country = transaction.get('receiver_country', '').upper()
        
        if sender_country in self.high_risk_countries or receiver_country in self.high_risk_countries:
            score += 25
        
        # Cross-border transactions add slight risk
        if sender_country != receiver_country and sender_country and receiver_country:
            score += 5
        
        # KYC tier risk
        kyc_tier = str(transaction.get('kyc_tier', 'unknown')).lower()
        if 'tier_1' in kyc_tier:
            score += 5
        elif 'tier_2' in kyc_tier:
            score += 15
        elif 'tier_3' in kyc_tier or kyc_tier == 'unknown':
            score += 25
        
        # Velocity risk
        velocity_1h = float(transaction.get('velocity_1h', 0))
        velocity_24h = float(transaction.get('velocity_24h', 0))
        
        if velocity_1h > 50000:
            score += 20
        elif velocity_1h > 10000:
            score += 10
        
        if velocity_24h > 100000:
            score += 15
        elif velocity_24h > 50000:
            score += 8
        
        # Merchant category risk
        merchant_category = str(transaction.get('merchant_category', '')).lower()
        if any(risk_cat in merchant_category for risk_cat in self.high_risk_categories):
            score += 15
        
        # Device change flag
        if transaction.get('device_change_flag', 0) == 1:
            score += 10
        
        # Account age risk
        account_age = int(transaction.get('customer_age_days', 365))
        if account_age < 30:
            score += 20
        elif account_age < 90:
            score += 10
        elif account_age < 365:
            score += 5
        
        # Prior transaction history in 24h
        prior_txn_24h = int(transaction.get('prior_txn_24h', 0))
        if prior_txn_24h > 20:
            score += 15
        elif prior_txn_24h > 10:
            score += 10
        elif prior_txn_24h > 5:
            score += 5
        
        # Cap at 100
        return min(score, 100)
    
    def categorize_risk(self, score):
        """Categorize risk based on score"""
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Low'
    
    def process_transactions(self, df):
        """Process all transactions and add risk scores"""
        # Calculate risk scores
        df['risk_score'] = df.apply(self.calculate_risk_score, axis=1)
        df['risk_category'] = df['risk_score'].apply(self.categorize_risk)
        
        # Sort by risk score descending
        df = df.sort_values('risk_score', ascending=False).reset_index(drop=True)
        
        return df

def generate_sample_data():
    """Generate sample transaction data for testing"""
    np.random.seed(42)
    n_transactions = 1000
    
    sample_data = {
        'txn_id': [f'TXN_{i:06d}' for i in range(1, n_transactions + 1)],
        'timestamp': pd.date_range('2024-01-01', periods=n_transactions, freq='1H'),
        'sender_country': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'IR', 'AF', 'KP'], n_transactions),
        'receiver_country': np.random.choice(['US', 'UK', 'DE', 'FR', 'CA', 'AU', 'MM', 'SY'], n_transactions),
        'amount_usd': np.random.lognormal(8, 2, n_transactions),
        'channel': np.random.choice(['online', 'mobile', 'branch', 'atm'], n_transactions),
        'customer_age_days': np.random.randint(1, 2000, n_transactions),
        'prior_txn_24h': np.random.poisson(3, n_transactions),
        'sanctioned_party_flag': np.random.choice([0, 1], n_transactions, p=[0.98, 0.02]),
        'kyc_tier': np.random.choice(['tier_1', 'tier_2', 'tier_3'], n_transactions),
        'merchant_category': np.random.choice(['retail', 'gambling', 'crypto', 'food', 'travel', 'adult'], n_transactions),
        'velocity_1h': np.random.lognormal(6, 1.5, n_transactions),
        'velocity_24h': np.random.lognormal(8, 1.5, n_transactions),
        'device_change_flag': np.random.choice([0, 1], n_transactions, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(sample_data)

def create_risk_distribution_chart(df):
    """Create risk distribution chart"""
    risk_counts = df['risk_category'].value_counts()
    colors = {'High': '#dc3545', 'Medium': '#fd7e14', 'Low': '#28a745'}
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=[colors[label] for label in risk_counts.index]
    )])
    
    fig.update_layout(
        title="Risk Distribution",
        font_size=14,
        height=400
    )
    
    return fig

def create_risk_score_histogram(df):
    """Create risk score histogram"""
    fig = px.histogram(
        df, 
        x='risk_score', 
        nbins=20,
        title='Risk Score Distribution',
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(
        xaxis_title="Risk Score",
        yaxis_title="Number of Transactions",
        height=400
    )
    return fig

def create_amount_vs_risk_scatter(df):
    """Create scatter plot of amount vs risk score"""
    fig = px.scatter(
        df.head(1000),  # Limit to 1000 points for performance
        x='amount_usd',
        y='risk_score',
        color='risk_category',
        title='Transaction Amount vs Risk Score',
        color_discrete_map={'High': '#dc3545', 'Medium': '#fd7e14', 'Low': '#28a745'},
        hover_data=['txn_id', 'sender_country', 'receiver_country']
    )
    fig.update_layout(
        xaxis_title="Amount (USD)",
        yaxis_title="Risk Score",
        height=400
    )
    return fig

def generate_narrative_summary(df):
    """Generate AI-like narrative summary"""
    total = len(df)
    high_risk = len(df[df['risk_category'] == 'High'])
    medium_risk = len(df[df['risk_category'] == 'Medium'])
    low_risk = len(df[df['risk_category'] == 'Low'])
    
    high_percent = (high_risk / total) * 100
    avg_score = df['risk_score'].mean()
    
    sanctioned_count = (df['sanctioned_party_flag'] == 1).sum()
    
    # Top risk factors analysis
    high_risk_df = df[df['risk_category'] == 'High']
    
    top_risk_countries = []
    if not high_risk_df.empty:
        country_counts = pd.concat([
            high_risk_df['sender_country'].value_counts(),
            high_risk_df['receiver_country'].value_counts()
        ]).groupby(level=0).sum().head(3)
        top_risk_countries = country_counts.index.tolist()
    
    top_channels = high_risk_df['channel'].value_counts().head(2).index.tolist() if not high_risk_df.empty else []
    
    avg_amount_high_risk = high_risk_df['amount_usd'].mean() if not high_risk_df.empty else 0
    
    narrative = f"""
    **Executive Summary:** Analysis of {total:,} transactions reveals a {high_percent:.1f}% high-risk rate with an average risk score of {avg_score:.1f}.
    
    **Key Findings:**
    • {sanctioned_count} transactions flagged for sanctioned parties (automatic high risk)
    • High-risk transactions average ${avg_amount_high_risk:,.2f} per transaction
    • Geographic risk concentrated in: {', '.join(top_risk_countries[:3]) if top_risk_countries else 'various regions'}
    • Channels showing elevated risk: {', '.join(top_channels) if top_channels else 'multiple channels'}
    • {(df['customer_age_days'] < 90).sum()} transactions from accounts less than 90 days old
    
    **Risk Factors Distribution:**
    • High velocity transactions (>$50K/1h): {(df['velocity_1h'] > 50000).sum()}
    • Cross-border transactions: {(df['sender_country'] != df['receiver_country']).sum()}
    • Device changes detected: {(df['device_change_flag'] == 1).sum()}
    
    **Recommendations:** 
    Prioritize manual review of the top {min(20, high_risk)} highest-scoring transactions. 
    Enhanced monitoring recommended for accounts with multiple risk indicators, particularly 
    new accounts with high transaction velocity or connections to high-risk jurisdictions.
    """
    
    return narrative

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Transaction Risk Scoring App</h1>', unsafe_allow_html=True)
    
    # Add deployment info
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <p>AI Engineering Challenge - Transaction Risk Scoring System</p>
        <p><em>Deploy to Streamlit Cloud for public URL sharing</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Controls")
    
    # Initialize the risk scorer
    risk_scorer = TransactionRiskScorer()
    
    # File upload or sample data
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data"]
    )
    
    df = None
    
    if data_option == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(df)} transactions")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    else:
        if st.sidebar.button("Generate Sample Data"):
            df = generate_sample_data()
            st.sidebar.success(f"✅ Generated {len(df)} sample transactions")
    
    if df is not None:
        # Process transactions
        with st.spinner("Processing transactions..."):
            df = risk_scorer.process_transactions(df)
        
        # Display results
        st.success(f"✅ Processed {len(df)} transactions successfully!")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_transactions = len(df)
        high_risk_count = len(df[df['risk_category'] == 'High'])
        medium_risk_count = len(df[df['risk_category'] == 'Medium'])
        low_risk_count = len(df[df['risk_category'] == 'Low'])
        
        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")
        with col2:
            st.metric("High Risk", f"{high_risk_count:,}", delta=f"{(high_risk_count/total_transactions)*100:.1f}%")
        with col3:
            st.metric("Medium Risk", f"{medium_risk_count:,}", delta=f"{(medium_risk_count/total_transactions)*100:.1f}%")
        with col4:
            st.metric("Low Risk", f"{low_risk_count:,}", delta=f"{(low_risk_count/total_transactions)*100:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = create_risk_distribution_chart(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_hist = create_risk_score_histogram(df)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Amount vs Risk scatter plot
        fig_scatter = create_amount_vs_risk_scatter(df)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top 20 High-Risk Transactions Table
        st.subheader("🚨 Top 20 High-Risk Transactions")
        
        top_20_high_risk = df.head(20)
        
        # Create display columns based on available data
        available_columns = []
        display_names = []
        
        # Transaction ID
        for col in ['txn_id', 'transaction_id', 'id', 'trans_id']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Transaction ID')
                break
        
        # Risk score and category (always available after processing)
        available_columns.extend(['risk_score', 'risk_category'])
        display_names.extend(['Risk Score', 'Risk Category'])
        
        # Amount
        for col in ['amount_usd', 'amount', 'transaction_amount', 'amt_usd']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Amount (USD)')
                break
        
        # Countries
        for col in ['sender_country', 'from_country', 'origin_country']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Sender Country')
                break
                
        for col in ['receiver_country', 'to_country', 'destination_country']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Receiver Country')
                break
        
        # Channel
        for col in ['channel', 'transaction_channel', 'method']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Channel')
                break
        
        # Merchant category
        for col in ['merchant_category', 'merchant_type', 'category']:
            if col in df.columns:
                available_columns.append(col)
                display_names.append('Merchant Category')
                break
        
        # Format the display dataframe
        display_df = top_20_high_risk[available_columns].copy()
        
        # Format amount columns
        for col in available_columns:
            if 'amount' in col.lower() and col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${float(x):,.2f}" if pd.notna(x) else "N/A")
        
        # Rename columns for better display
        column_mapping = dict(zip(available_columns, display_names))
        display_df = display_df.rename(columns=column_mapping)
        
        # Style the dataframe
        def style_risk_category(val):
            if val == 'High':
                return 'color: #dc3545; font-weight: bold'
            elif val == 'Medium':
                return 'color: #fd7e14; font-weight: bold'
            else:
                return 'color: #28a745; font-weight: bold'
        
        styled_df = display_df.style.applymap(style_risk_category, subset=['risk_category'])
        st.dataframe(styled_df, use_container_width=True)
        
        # AI Narrative Summary
        st.subheader("🤖 AI Risk Assessment Summary")
        narrative = generate_narrative_summary(df)
        st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)
        
        # Download processed data
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name=f"risk_scored_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            high_risk_csv = df[df['risk_category'] == 'High'].to_csv(index=False)
            st.download_button(
                label="Download High-Risk Only (CSV)",
                data=high_risk_csv,
                file_name=f"high_risk_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Risk scoring details
        with st.expander("📋 Risk Scoring Details"):
            st.markdown("""
            **Risk Scoring Algorithm:**
            - Base score: 10 points
            - Sanctioned party flag: Automatic 100 points
            - Amount-based risk: Up to 30 points
            - Country risk: Up to 25 points  
            - KYC tier: Up to 25 points
            - Velocity risk: Up to 35 points
            - Merchant category: Up to 15 points
            - Device changes: Up to 10 points
            - Account age: Up to 20 points
            - Transaction frequency: Up to 15 points
            - Maximum score: 100 points
            
            **Risk Categories:**
            - **High Risk:** 70-100 points
            - **Medium Risk:** 40-69 points  
            - **Low Risk:** 0-39 points
            """)
    
    else:
        st.info("👆 Please upload a CSV file or generate sample data to begin analysis.")
        
        # Show expected CSV format
        with st.expander("📄 Expected CSV Format"):
            st.markdown("""
            Your CSV should contain the following columns:
            - `txn_id`: Transaction ID
            - `timestamp`: Transaction timestamp
            - `sender_country`: Sender country code
            - `receiver_country`: Receiver country code  
            - `amount_usd`: Amount in USD
            - `channel`: Transaction channel
            - `customer_age_days`: Customer account age in days
            - `prior_txn_24h`: Prior transactions in 24 hours
            - `sanctioned_party_flag`: 1 if sanctioned party involved, 0 otherwise
            - `kyc_tier`: KYC verification tier
            - `merchant_category`: Merchant category
            - `velocity_1h`: Transaction velocity in 1 hour
            - `velocity_24h`: Transaction velocity in 24 hours
            - `device_change_flag`: 1 if device changed, 0 otherwise
            """)

if __name__ == "__main__":
    main()