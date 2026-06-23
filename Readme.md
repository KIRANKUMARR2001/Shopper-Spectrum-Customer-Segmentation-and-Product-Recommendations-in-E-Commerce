# 🛍️ Shopper Spectrum: Enterprise Retail Analytics Platform

Shopper Spectrum is a dual-engine machine learning platform designed to optimize e-commerce business operations. By linking customer transaction histories directly to predictive intelligence, the platform addresses two critical retail workflows: **Customer Lifecycle Segmentation** and **Automated Cross-Sell Recommendations**. 

The entire system is deployed via an interactive, production-ready **Streamlit Web Application** hosted in the cloud.

---

## 🚀 Live Application Architecture

The deployed solution scales complex backend data pipelines and pre-trained statistical matrices into a clean, two-tab dashboard layout:
              ┌───────────────────────────────┐
              │ Raw E-Commerce Transactions   │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Data Pre-processing Pipeline │
              │  (Removes Duplicates & Noise) │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
 ┌─────────────────────────┐     ┌─────────────────────────┐
 │  Customer Segmentation  │     │ Product Recommendations │
 │        (RFM)            │     │   (User-Item Matrix)    │
 └────────────┬────────────┘     └────────────┬────────────┘
              │                               │
              ▼                               ▼
 ┌─────────────────────────┐     ┌─────────────────────────┐
 │   K-Means Clustering    │     │ Item-Based Collaborative│
 │         (K=4)           │     │   Filtering Pipeline    │
 └────────────┬────────────┘     └────────────┬────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐ 
              │     Interactive Streamlit     │
              │         Cloud Portal        │
              └───────────────────────────────┘

              ---

## 🧠 Core Machine Learning Engines

### 👥 Engine 1: Customer Segmentation (K-Means Clustering)
* **Objective:** Automatically identify distinct customer behavior profiles to support hyper-targeted marketing and minimize churn.
* **Methodology:** Transaction records are aggregated into a **Recency, Frequency, and Monetary (RFM)** vector space. Data scaling is handled via a persistent `StandardScaler` pipeline.
* **Model Configuration:** A **K-Means** model initialized with `k-means++` partitions customers into 4 structurally distinct operational boundaries validated explicitly by both the Elbow Method and Silhouette Coefficient local maximums:
  * **High-Value Champions (VIP):** Core revenue drivers. Target with early product access and zero-fee loyalty structures.
  * **Regular Customers:** Consistent shoppers showing standard lifecycle traits. Maintain monthly catalog engagement.
  * **Occasional Buyers:** High-latency shoppers. Target with multi-item bundle deals to naturally scale average basket sizes.
  * **At-Risk Accounts:** Severe churn hazards. Automatically trigger high-incentive win-back discount structures.

### 📦 Engine 2: Product Recommendation (Item-Based Collaborative Filtering)
* **Objective:** Boost Average Order Value (AOV) by dynamically suggesting complementary products on checkout screens.
* **Methodology:** Shuns generic text-keyword matching in favor of global transaction co-occurrence tracking. 
* **Live Pipeline Bypass:** To circumvent large-file cloud data transfer limits, a stripped down 3-column interaction matrix (`mini_retail_data.csv`) is processed live into a dense user-item vector layout on application launch.
* **Mathematics:** The algorithm calculates the multi-dimensional geometric overlap of product vectors using **Cosine Similarity** to instantly extract the optimal **Top-5 Neighbor Boundary ($N=5$)** of corresponding items.

---

## 📂 Repository File Structure

Ensure your production workspace contains these files mapped flat to the root directory:

```text
├── app.py                  # Streamlit production frontend application code
├── scaler.pkl              # Saved scikit-learn standard normalizer binary 
├── kmeans_model.pkl        # Pre-trained K-Means model pipeline binary
├── mini_retail_data.csv    # Lightweight optimized interaction dataset (< 5 MB)
└── requirements.txt        # Server package installation manifest
