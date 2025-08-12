import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

try: 
    # Load model ready data
    model_data_path = os.path.normpath(os.path.join(script_dir, "../datasets/processed/model_ready_features.csv"))
    model_df = pd.read_csv(model_data_path)
    print("Model data loaded successfully")

    # Load original data
    orginal_data_path = os.path.normpath(os.path.join(script_dir, "../datasets/processed/customer_features.csv"))
    orginal_df = pd.read_csv(orginal_data_path)
    print("data loaded ")

except FileNotFoundError as e:
    print(f"not found: {e}")
    exit(1)
except Exception as e:
    print(f"error: {e}")
    exit(1)

# 
model_df.set_index("household_id", inplace=True)
print(f"Model data shape: {model_df.shape}")

# optimal 
print("Searching for optimal K value...")
k_list = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=37)
    kmeans.fit(model_df)
    k_list.append(kmeans.inertia_)
    print(f"K={k}: Inertia={kmeans.inertia_}")
"""
# Elbow Method plot
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_list, marker='o', linestyle='--', linewidth=2, markersize=8)
plt.title('Elbow Method for K-Means Clustering', fontsize=14, fontweight="bold")
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
# finded optimal k = 5


optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(model_df)

orginal_df["cluster"] = clusters

cluster_profile = orginal_df.groupby("cluster").agg({
    "total_sales": ["mean", "count"],
    "basket_count": ["mean"],
    "recency": ["mean"],
    'total_coupons_redeemed': ['mean'],
    'discount_ratio': ['mean'],
    'income_desc': lambda x: x.mode()[0], 
    'age_desc': lambda x: x.mode()[0]      
}).round(2)



plt.figure(figsize=(12, 8))
sns.scatterplot(data=orginal_df, 
                x='recency', 
                y='total_sales', 
                hue='cluster',
                palette='viridis',
                s=100,
                alpha=0.8)
plt.title("Recency vs. Monetary")
plt.xlabel("Recency")
plt.ylabel("Total Sales")
plt.legend(title="Segment id")
plt.show()



if __name__ == "__main__":
    
    output_path = "../datasets/processed/customer_segments.csv"
    if script_dir:
        output_path = os.path.normpath(os.path.join(script_dir, output_path))
    orginal_df.to_csv(output_path, index=False)
    print("data saved")
