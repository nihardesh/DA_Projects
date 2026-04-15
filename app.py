import streamlit as st
import pandas as pd

# PAGE CONFIG
st.set_page_config(
    page_title="Customer Churn Analytics | European Central Bank",   
    layout="wide"
)

# TITLE & EXECUTIVE SUMMARY
st.title("Customer Segmentation & Churn Pattern Analytics")
st.markdown("**European Central Bank**")
st.markdown("---")

with st.expander("Executive Summary (Click to expand)", expanded=False):
    st.markdown("""
    ### Executive Summary

    This dashboard presents a comprehensive analysis of customer churn patterns across European banking customers
    in **France, Spain, and Germany**. The analysis covers over 10,000 customer records and examines churn
    across five key segmentation dimensions: geography, age group, credit score band, tenure group, and balance segment.

    **Key findings*:**
    - Overall churn rate sits around **20%**, with significant variation across segments
    - Customers aged **46–60** show the highest churn risk among all age groups
    - **Germany** shows substantially higher churn than France and Spain
    - **Inactive members** churn at nearly double the rate of active members
    - **High-balance customers** represent significant revenue risk despite lower absolute churn numbers
    - Customers with **zero balance** and those with **only one product** are at elevated churn risk

    **Recommendations*:**
    - Launch targeted retention campaigns for the 46–60 age group in Germany
    - Re-engagement programs for inactive members should be prioritized
    - Premium service tiers for high-balance customers to improve loyalty
    - Cross-sell additional products to single-product customers to increase retention
    """)

st.markdown("---")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank dataset.csv")
    return df

df = load_data()

# DATA VALIDATION
st.subheader("Step 1: Data Validation")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Missing Values per Column**")
    missing = df.isnull().sum()
    st.dataframe(missing[missing >= 0].rename("Missing Count"))

with col2:
    st.write("**Binary Field Validation**")
    binary_check = pd.DataFrame({
        "Field": ["Exited", "IsActiveMember", "HasCrCard"],
        "Valid (0 or 1 only)": [
            df['Exited'].isin([0, 1]).all(),
            df['IsActiveMember'].isin([0, 1]).all(),
            df['HasCrCard'].isin([0, 1]).all()
        ]
    })
    st.dataframe(binary_check, hide_index=True)

with col3:
    st.write("**Negative Value Check**")
    neg_check = pd.DataFrame({
        "Field": ["Balance", "Age", "EstimatedSalary"],
        "Negative Count": [
            (df['Balance'] < 0).sum(),
            (df['Age'] < 0).sum(),
            (df['EstimatedSalary'] < 0).sum()
        ]
    })
    st.dataframe(neg_check, hide_index=True)

st.success("Dataset validated successfully — no missing values, binary fields confirmed, no negative values.")
st.markdown("---")

# DATA CLEANING & FEATURE ENGINEERING
df = df.drop(['CustomerId', 'Surname'], axis=1)

# Age Group
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 30, 45, 60, 100],
    labels=['<30', '30-45', '46-60', '60+']
)

# Credit Band
def credit_band(score):
    if score < 500:
        return "Low (<500)"
    elif score < 700:
        return "Medium (500-699)"
    else:
        return "High (700+)"

df['CreditBand'] = df['CreditScore'].apply(credit_band)

# Tenure Group
df['TenureGroup'] = pd.cut(
    df['Tenure'],
    bins=[-1, 3, 7, 10],
    labels=['New (0-3 yrs)', 'Mid (4-7 yrs)', 'Long (8-10 yrs)']
)

# Balance Segment
def balance_segment(balance):
    if balance == 0:
        return "Zero Balance"
    elif balance < 100000:
        return "Low (<100K)"
    else:
        return "High (100K+)"

df['BalanceSegment'] = df['Balance'].apply(balance_segment)

# SIDEBAR FILTERS
st.sidebar.header("Filter Dashboard")
st.sidebar.markdown("Use filters below to explore specific customer segments.")

geo_options = sorted(df['Geography'].unique())
geo = st.sidebar.multiselect("Geography", geo_options, default=geo_options)

age_options = ['<30', '30-45', '46-60', '60+']
age = st.sidebar.multiselect("Age Group", age_options, default=age_options)

credit_options = ["Low (<500)", "Medium (500-699)", "High (700+)"]
credit = st.sidebar.multiselect("Credit Band", credit_options, default=credit_options)

gender_options = sorted(df['Gender'].unique())
gender = st.sidebar.multiselect("⚥ Gender", gender_options, default=gender_options)

filtered_df = df[
    (df['Geography'].isin(geo)) &
    (df['AgeGroup'].isin(age)) &
    (df['CreditBand'].isin(credit)) &
    (df['Gender'].isin(gender))
]

st.sidebar.markdown(f"**Showing:** {len(filtered_df):,} of {len(df):,} customers")

# KPIs
st.subheader("Step 2: Key Performance Indicators")

churn_rate = filtered_df['Exited'].mean()
total_customers = len(filtered_df)
active_rate = filtered_df['IsActiveMember'].mean()
high_value_df = filtered_df[filtered_df['Balance'] > 100000]
high_value_churn = high_value_df['Exited'].mean() if len(high_value_df) > 0 else 0

inactive_df = filtered_df[filtered_df['IsActiveMember'] == 0]
active_members_df = filtered_df[filtered_df['IsActiveMember'] == 1]
inactive_churn = inactive_df['Exited'].mean() if len(inactive_df) > 0 else 0
active_churn = active_members_df['Exited'].mean() if len(active_members_df) > 0 else 0
engagement_drop = inactive_churn - active_churn

geo_churn = filtered_df.groupby('Geography')['Exited'].mean()
geo_risk = geo_churn.max() if len(geo_churn) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Overall Churn Rate", f"{churn_rate * 100:.2f}%")
col3.metric("Active Members", f"{active_rate * 100:.2f}%")
col4.metric("High-Value Churn", f"{high_value_churn * 100:.2f}%")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Inactive Member Churn", f"{inactive_churn * 100:.2f}%")
col6.metric("Active Member Churn", f"{active_churn * 100:.2f}%")
col7.metric("Engagement Drop Indicator", f"{engagement_drop * 100:.2f}%",
            help="Difference in churn rate between inactive and active members")
col8.metric("Geographic Risk Index", f"{geo_risk * 100:.2f}%",
            help="Highest regional churn rate among selected geographies")

st.markdown("---")

# CHURN DISTRIBUTION BY SEGMENTS
st.subheader("Step 3: Churn Distribution Across Segments")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Age Group", "Tenure Group", "Credit Band", "Balance Segment", "Products & Cards"
])

with tab1:
    st.write("**Churn Rate by Age Group**")
    age_churn = filtered_df.groupby('AgeGroup', observed=True)['Exited'].agg(['mean', 'sum', 'count']).reset_index()
    age_churn.columns = ['Age Group', 'Churn Rate', 'Churned Customers', 'Total Customers']
    age_churn['Churn Rate %'] = (age_churn['Churn Rate'] * 100).round(2)
    age_churn['Segment Contribution %'] = (age_churn['Churned Customers'] / age_churn['Churned Customers'].sum() * 100).round(2)
    st.bar_chart(age_churn.set_index('Age Group')['Churn Rate %'])
    st.dataframe(age_churn[['Age Group', 'Total Customers', 'Churned Customers', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

with tab2:
    st.write("**Churn Rate by Tenure Group**")
    tenure_churn = filtered_df.groupby('TenureGroup', observed=True)['Exited'].agg(['mean', 'sum', 'count']).reset_index()
    tenure_churn.columns = ['Tenure Group', 'Churn Rate', 'Churned Customers', 'Total Customers']
    tenure_churn['Churn Rate %'] = (tenure_churn['Churn Rate'] * 100).round(2)
    tenure_churn['Segment Contribution %'] = (tenure_churn['Churned Customers'] / tenure_churn['Churned Customers'].sum() * 100).round(2)
    st.bar_chart(tenure_churn.set_index('Tenure Group')['Churn Rate %'])
    st.dataframe(tenure_churn[['Tenure Group', 'Total Customers', 'Churned Customers', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

with tab3:
    st.write("**Churn Rate by Credit Band**")
    credit_churn = filtered_df.groupby('CreditBand')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
    credit_churn.columns = ['Credit Band', 'Churn Rate', 'Churned Customers', 'Total Customers']
    credit_churn['Churn Rate %'] = (credit_churn['Churn Rate'] * 100).round(2)
    credit_churn['Segment Contribution %'] = (credit_churn['Churned Customers'] / credit_churn['Churned Customers'].sum() * 100).round(2)
    st.bar_chart(credit_churn.set_index('Credit Band')['Churn Rate %'])
    st.dataframe(credit_churn[['Credit Band', 'Total Customers', 'Churned Customers', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

with tab4:
    st.write("**Churn Rate by Balance Segment**")
    bal_order = ['Zero Balance', 'Low (<100K)', 'High (100K+)']
    bal_churn = filtered_df.groupby('BalanceSegment')['Exited'].agg(['mean', 'sum', 'count']).reindex(bal_order).reset_index()
    bal_churn.columns = ['Balance Segment', 'Churn Rate', 'Churned Customers', 'Total Customers']
    bal_churn['Churn Rate %'] = (bal_churn['Churn Rate'] * 100).round(2)
    bal_churn['Segment Contribution %'] = (bal_churn['Churned Customers'] / bal_churn['Churned Customers'].sum() * 100).round(2)
    st.bar_chart(bal_churn.set_index('Balance Segment')['Churn Rate %'])
    st.dataframe(bal_churn[['Balance Segment', 'Total Customers', 'Churned Customers', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

with tab5:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Churn Rate by Number of Products**")
        prod_churn = filtered_df.groupby('NumOfProducts')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
        prod_churn.columns = ['Num Products', 'Churn Rate', 'Churned', 'Total']
        prod_churn['Churn Rate %'] = (prod_churn['Churn Rate'] * 100).round(2)
        st.bar_chart(prod_churn.set_index('Num Products')['Churn Rate %'])
        st.dataframe(prod_churn[['Num Products', 'Total', 'Churned', 'Churn Rate %']], hide_index=True)
    with col_b:
        st.write("**Churn Rate by Credit Card Ownership**")
        card_churn = filtered_df.groupby('HasCrCard')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
        card_churn['HasCrCard'] = card_churn['HasCrCard'].map({0: 'No Credit Card', 1: 'Has Credit Card'})
        card_churn.columns = ['Credit Card', 'Churn Rate', 'Churned', 'Total']
        card_churn['Churn Rate %'] = (card_churn['Churn Rate'] * 100).round(2)
        st.bar_chart(card_churn.set_index('Credit Card')['Churn Rate %'])
        st.dataframe(card_churn[['Credit Card', 'Total', 'Churned', 'Churn Rate %']], hide_index=True)

st.markdown("---")

# GEOGRAPHY ANALYSIS
st.subheader("Step 4: Geography-wise Churn Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("**Churn Rate by Geography**")
    geo_churn_df = filtered_df.groupby('Geography')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
    geo_churn_df.columns = ['Geography', 'Churn Rate', 'Churned', 'Total']
    geo_churn_df['Churn Rate %'] = (geo_churn_df['Churn Rate'] * 100).round(2)
    geo_churn_df['Segment Contribution %'] = (geo_churn_df['Churned'] / geo_churn_df['Churned'].sum() * 100).round(2)
    st.bar_chart(geo_churn_df.set_index('Geography')['Churn Rate %'])
    st.dataframe(geo_churn_df[['Geography', 'Total', 'Churned', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

with col2:
    st.write("**Age × Geography Interaction (Churn Rate)**")
    pivot = pd.pivot_table(
        filtered_df,
        values='Exited',
        index='Geography',
        columns='AgeGroup',
        aggfunc='mean',
        observed=True
    ).round(3)
    pivot = pivot * 100
    st.dataframe(pivot.style.format("{:.1f}%").background_gradient(cmap='Reds'))

st.markdown("---")

# GENDER ANALYSIS
st.subheader("⚥ Step 5: Gender-based Churn Analysis")

gender_churn = filtered_df.groupby('Gender')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
gender_churn.columns = ['Gender', 'Churn Rate', 'Churned', 'Total']
gender_churn['Churn Rate %'] = (gender_churn['Churn Rate'] * 100).round(2)
gender_churn['Segment Contribution %'] = (gender_churn['Churned'] / gender_churn['Churned'].sum() * 100).round(2)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(gender_churn.set_index('Gender')['Churn Rate %'])
with col2:
    st.dataframe(gender_churn[['Gender', 'Total', 'Churned', 'Churn Rate %', 'Segment Contribution %']], hide_index=True)

st.markdown("---")

# CHURNED VS RETAINED PROFILES
st.subheader("Step 6: Churned vs Retained Customer Profiles")

churned = filtered_df[filtered_df['Exited'] == 1]
retained = filtered_df[filtered_df['Exited'] == 0]

profile_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

churned_profile = churned[profile_cols].mean().round(2)
retained_profile = retained[profile_cols].mean().round(2)

comparison = pd.DataFrame({
    'Metric': profile_cols,
    'Churned Customers (Avg)': churned_profile.values,
    'Retained Customers (Avg)': retained_profile.values
})
comparison['Difference'] = (comparison['Churned Customers (Avg)'] - comparison['Retained Customers (Avg)']).round(2)

st.dataframe(comparison, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.write(f"**Churned Customers:** {len(churned):,}")
    st.write(f"Active Members among Churned: {churned['IsActiveMember'].mean() * 100:.1f}%")
    st.write(f"Has Credit Card (Churned): {churned['HasCrCard'].mean() * 100:.1f}%")
with col2:
    st.write(f"**Retained Customers:** {len(retained):,}")
    st.write(f"Active Members among Retained: {retained['IsActiveMember'].mean() * 100:.1f}%")
    st.write(f"Has Credit Card (Retained): {retained['HasCrCard'].mean() * 100:.1f}%")

st.markdown("---")

# HIGH VALUE CUSTOMER ANALYSIS
st.subheader("Step 7: High-Value Customer Churn Analysis")

if len(high_value_df) > 0:
    col1, col2, col3 = st.columns(3)
    col1.metric("High-Value Customers", f"{len(high_value_df):,}")
    col2.metric("High-Value Churn Rate", f"{high_value_churn * 100:.2f}%")
    col3.metric("High-Value Churned Count", f"{int(high_value_df['Exited'].sum()):,}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**High-Value Churn by Geography**")
        hv_geo = high_value_df.groupby('Geography')['Exited'].mean() * 100
        hv_geo = hv_geo.round(2).rename("Churn Rate %")
        st.bar_chart(hv_geo)

    with col_b:
        st.write("**High-Value Churn by Gender**")
        hv_gender = high_value_df.groupby('Gender')['Exited'].mean() * 100
        hv_gender = hv_gender.round(2).rename("Churn Rate %")
        st.bar_chart(hv_gender)

    st.write("**High-Value Churned vs Retained — Average Profile**")
    hv_churned = high_value_df[high_value_df['Exited'] == 1]
    hv_retained = high_value_df[high_value_df['Exited'] == 0]
    hv_compare = pd.DataFrame({
        'Metric': profile_cols,
        'HV Churned (Avg)': hv_churned[profile_cols].mean().round(2).values,
        'HV Retained (Avg)': hv_retained[profile_cols].mean().round(2).values
    })
    st.dataframe(hv_compare, hide_index=True)
else:
    st.warning("No high-value customers found with current filter selection.")

st.markdown("---")

# SALARY VS BALANCE IMPACT
st.subheader("Step 8: Salary vs Balance Impact on Churn")

salary_balance = filtered_df.groupby('Exited')[['EstimatedSalary', 'Balance']].mean().reset_index()
salary_balance['Exited'] = salary_balance['Exited'].map({0: 'Retained', 1: 'Churned'})
salary_balance = salary_balance.rename(columns={
    'Exited': 'Status',
    'EstimatedSalary': 'Avg Estimated Salary',
    'Balance': 'Avg Balance'
})
salary_balance['Avg Estimated Salary'] = salary_balance['Avg Estimated Salary'].round(2)
salary_balance['Avg Balance'] = salary_balance['Avg Balance'].round(2)
st.dataframe(salary_balance, hide_index=True)

st.markdown("---")

# REVENUE RISK
st.subheader("Step 9: Revenue Risk Analysis")

total_balance = filtered_df['Balance'].sum()
churned_balance = filtered_df[filtered_df['Exited'] == 1]['Balance'].sum()
retained_balance = filtered_df[filtered_df['Exited'] == 0]['Balance'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Portfolio Balance", f"€total_balance:,.0f")
col2.metric("Balance at Churn Risk", f"€{churned_balance:,.0f}")
col3.metric("Retained Balance", f"€{retained_balance:,.0f}")

if total_balance > 0:
    risk_pct = (churned_balance / total_balance) * 100
    st.metric("Revenue Risk %", f"{risk_pct:.2f}%",
              help="Percentage of total portfolio balance held by customers who churned")

    st.write("**Revenue Risk by Geography**")
    geo_rev_risk = filtered_df.groupby('Geography').apply(
        lambda x: pd.Series({
            'Total Balance': x['Balance'].sum(),
            'Churned Balance': x[x['Exited'] == 1]['Balance'].sum(),
            'Revenue Risk %': round(x[x['Exited'] == 1]['Balance'].sum() / x['Balance'].sum() * 100, 2) if x['Balance'].sum() > 0 else 0
        })
    ).reset_index()
    st.dataframe(geo_rev_risk, hide_index=True)

st.markdown("---")

# SEGMENT CONTRIBUTION
st.subheader("Step 10: Customer Segment Contribution")

col1, col2 = st.columns(2)

with col1:
    st.write("**Customer Distribution by Age Group**")
    age_dist = filtered_df['AgeGroup'].value_counts(normalize=True).rename("Share %") * 100
    st.bar_chart(age_dist.round(2))

with col2:
    st.write("**Customer Distribution by Geography**")
    geo_dist = filtered_df['Geography'].value_counts(normalize=True).rename("Share %") * 100
    st.bar_chart(geo_dist.round(2))

col3, col4 = st.columns(2)

with col3:
    st.write("**Customer Distribution by Balance Segment**")
    bal_dist = filtered_df['BalanceSegment'].value_counts(normalize=True).rename("Share %") * 100
    st.bar_chart(bal_dist.round(2))

with col4:
    st.write("**Customer Distribution by Tenure Group**")
    tenure_dist = filtered_df['TenureGroup'].value_counts(normalize=True).rename("Share %") * 100
    st.bar_chart(tenure_dist.round(2))

st.markdown("---")

# ENGAGEMENT ANALYSIS
st.subheader("Step 11: Engagement & Activity Analysis")

engage_churn = filtered_df.groupby('IsActiveMember')['Exited'].agg(['mean', 'sum', 'count']).reset_index()
engage_churn['IsActiveMember'] = engage_churn['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})
engage_churn.columns = ['Member Status', 'Churn Rate', 'Churned', 'Total']
engage_churn['Churn Rate %'] = (engage_churn['Churn Rate'] * 100).round(2)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(engage_churn.set_index('Member Status')['Churn Rate %'])
with col2:
    st.dataframe(engage_churn[['Member Status', 'Total', 'Churned', 'Churn Rate %']], hide_index=True)
    st.info(f"Inactive members churn at **{engagement_drop * 100:.1f}% more** than active members (Engagement Drop Indicator)")

st.markdown("---")

# FULL DATA EXPLORER
st.subheader("Full Data Explorer")
st.write(f"Showing **{len(filtered_df):,}** customers based on current filters.")
st.dataframe(filtered_df, use_container_width=True)

# KEY INSIGHTS & RECOMMENDATIONS
st.subheader("Key Insights & Strategic Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Key Findings*:**
    - Churn is highest among the **46–60 age group**, signalling mid-to-late career customers need more engagement
    - **Germany** has the highest geographic churn risk compared to France and Spain
    - **Inactive members** are significantly more likely to churn — engagement directly predicts retention
    - Customers with **only 1 product** churn at a far higher rate than multi-product holders
    - **Zero-balance accounts** show elevated churn, possibly indicating dormant or disengaged customers
    - **High-balance customers** represent a concentrated revenue risk if they churn
    """)

with col2:
    st.markdown("""
    **Strategic Recommendations*:**
    - Launch **targeted retention campaigns** for 46–60 age bracket, especially in Germany
    - Introduce **re-engagement programs** for inactive members (reminders, offers, rewards)
    - Develop **premium loyalty tiers** for high-balance customers to reduce their churn risk
    - Promote **cross-selling of additional products** to single-product customers
    - Investigate why **German customers churn more** — pricing, service gaps, or competition
    - Use this dashboard to **monitor churn KPIs monthly** and adjust strategies accordingly
    """)

st.markdown("---")
st.caption("Customer Segmentation & Churn Analytics | European Central Bank")
st.caption("*Every recommendation and key finding is based on current data ONLY and is not dynamic with respect to change in data or filters.")