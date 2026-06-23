import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# 1. PAGE CONFIGURATION & LIVE ASSET PIPELINE
# =====================================================================
st.set_page_config(
    page_title="Retail Analytics Dashboard", 
    page_icon="🛍️", 
    layout="wide"
)

@st.cache_resource
def load_and_compute_assets():
    """Loads segmentation models and builds the matrix live from transactions"""
    try:
        # 1. Load small segmentation files
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_model.pkl')
        
        # 2. Load your clean transactional CSV file directly
        # ⚠️ Make sure the filename below matches your CSV file exactly!
        df = pd.read_csv('mini_retail_data.csv')
        
        # 3. Build the User-Item matrix live in the background
        user_item = df.pivot_table(
            index='CustomerID', 
            columns='StockCode', 
            values='Quantity', 
            aggfunc='sum'
        ).fillna(0)
        
        # 4. Compute Cosine Similarity instantly
        item_sim_array = cosine_similarity(user_item.T)
        similarity_matrix = pd.DataFrame(item_sim_array, index=user_item.columns, columns=user_item.columns)
        
        return scaler, kmeans, similarity_matrix
    except FileNotFoundError as e:
        st.error(f"❌ Missing required file: {e.filename}. Please make sure your models and clean CSV are in this folder!")
        return None, None, None

scaler, kmeans, similarity_matrix = load_and_compute_assets()

# =====================================================================
# 2. APP HEADER UI
# =====================================================================
st.title("🛍️ Shopper Spectrum Analytics Dashboard")
st.markdown("""
Welcome to the production workspace. Use the operational tabs below to switch 
between **Customer Segmentation Sliders** and the **Cross-Sell Recommendation Matrix**.
""")
st.write("---")

# Setup Navigation Tabs
tab1, tab2 = st.tabs(["👥 Customer Segmentation", "📦 Product Recommendations"])

# =====================================================================
# 3. TAB 1: CUSTOMER SEGMENTATION (SLIDERS + K-MEANS)
# =====================================================================
with tab1:
    st.header("Real-Time Customer Cohort Predictor")
    st.write("Adjust the behavioral attributes to map a customer profile instantly into a business segment:")
    st.write("")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.slider("Recency (Days since last checkout)", min_value=1, max_value=365, value=30)
    with col2:
        frequency = st.slider("Frequency (Total distinct orders)", min_value=1, max_value=100, value=5)
    with col3:
        monetary = st.number_input("Monetary Value (Total money spent in $)", min_value=1.0, max_value=100000.0, value=500.0, step=50.0)
        
    st.write("")
    if st.button("Predict Segment Group", type="primary"):
        if scaler is not None and kmeans is not None:
            user_features = np.array([[recency, frequency, monetary]])
            scaled_features = scaler.transform(user_features)
            cluster_id = kmeans.predict(scaled_features)[0]
            
            cluster_mapping = {
                0: ('Regular Customer', '🔵 Active profile. Maintain standard catalog communications.'),
                1: ('At-Risk Account', '🔴 Churn hazard! Trigger automated win-back discount emails immediately.'),
                2: ('High-Value Champion', '🟢 VIP buyer! Provide premium customer lines and early access perks.'),
                3: ('Occasional Buyer', '🟡 Low frequency profile. Offer transactional product bundles to grow basket size.')
            }
            
            segment, strategy = cluster_mapping.get(cluster_id, ("Unknown", "Inspect model boundaries."))
            
            st.write("---")
            st.metric(label="Assigned Operational Segment", value=segment)
            
            if cluster_id == 2:
                st.success(f"**Strategic Next Step:** {strategy}")
            elif cluster_id == 1:
                st.error(f"**Strategic Next Step:** {strategy}")
            elif cluster_id == 3:
                st.warning(f"**Strategic Next Step:** {strategy}")
            else:
                st.info(f"**Strategic Next Step:** {strategy}")

# =====================================================================
# 4. TAB 2: PRODUCT RECOMMENDATIONS (DYNAMICS)
# =====================================================================
with tab2:
    st.header("Product Recommendation Look-up Matrix")
    st.write("Select a target item to pull the top 5 complementary products based on co-occurrence behaviors:")
    st.write("")
    
    if similarity_matrix is not None:
        all_products = sorted(similarity_matrix.index.tolist())
        selected_product = st.selectbox("Search/Select Product StockCode ID:", all_products)
        
        st.write("")
        if st.button("Generate Cross-Sell Items", type="primary"):
            product_scores = similarity_matrix[selected_product]
            top_5 = product_scores.sort_values(ascending=False).iloc[1:6]
            
            rec_df = pd.DataFrame({
                'Recommended StockCode ID': top_5.index,
                'Match Confidence Score (Cosine)': top_5.values
            }).reset_index(drop=True)
            
            st.write("---")
            st.markdown(f"### 🎯 Top 5 Items to Display Alongside Product `{selected_product}`:")
            st.dataframe(rec_df.style.format({'Match Confidence Score (Cosine)': '{:.4f}'}), use_container_width=True)
