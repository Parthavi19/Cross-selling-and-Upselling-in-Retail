import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Instacart dataset files
orders = pd.read_csv('orders.csv')
order_products = pd.read_csv('order_products__train.csv')
products = pd.read_csv('products.csv')
aisles = pd.read_csv('aisles.csv')

# Sample 10,000 orders to reduce computational load
sample_orders = orders.sample(n=10000, random_state=42)
order_products = order_products[order_products['order_id'].isin(sample_orders['order_id'])]

# Merge datasets to get user_id, product, and aisle information
order_products = order_products.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
order_products = order_products.merge(products, on='product_id', how='left')
order_products = order_products.merge(aisles, on='aisle_id', how='left')

# -------------------------------
# Apriori Algorithm for Cross-Selling
# -------------------------------
# Prepare transactional data: one-hot encoded products per order
basket = (order_products.groupby(['order_id', 'product_name'])['order_id']
          .count().unstack().reset_index().fillna(0)
          .set_index('order_id'))

# Convert to boolean for mlxtend (avoid deprecation warning)
basket = basket > 0

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter high-confidence rules
rules = rules[(rules['confidence'] >= 0.5) & (rules['lift'] >= 1.2)]
rules = rules.sort_values('lift', ascending=False)

# Save rules
rules.to_csv('association_rules.csv', index=False)
print("Top 5 Association Rules for Cross-Selling:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# -------------------------------
# K-Means Clustering for Customer Segmentation
# -------------------------------
# Prepare customer-level features: purchase frequency by aisle
customer_features = (order_products.groupby(['user_id', 'aisle'])['product_id']
                     .count().unstack().reset_index().fillna(0)
                     .set_index('user_id'))

# Normalize features
scaler = StandardScaler()
customer_features_scaled = scaler.fit_transform(customer_features)

# Apply K-Means clustering (k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
customer_features['cluster'] = kmeans.fit_predict(customer_features_scaled)

# Analyze cluster characteristics
cluster_summary = customer_features.groupby('cluster').mean()
cluster_summary.to_csv('cluster_summary.csv')

# Plot cluster sizes
plt.figure(figsize=(8, 6))
customer_features['cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('Customer Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.savefig('cluster_sizes.png')
plt.close()

# Print cluster summary
print("\nCluster Summary (Average Purchase Frequency by Aisle):")
print(cluster_summary.head())

# -------------------------------
# Example Recommendations
# -------------------------------
print("\nExample Recommendations:")
print("Cross-Selling (from Apriori):")
for idx, row in rules.head(2).iterrows():
    antecedents = ', '.join(list(row['antecedents']))
    consequents = ', '.join(list(row['consequents']))
    print(f"If customer buys {antecedents}, recommend {consequents} (Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f})")

print("\nUpselling (from K-Means):")
for cluster in range(5):
    top_aisles = cluster_summary.loc[cluster].sort_values(ascending=False).head(3)
    print(f"Cluster {cluster}: Target customers buying {', '.join(top_aisles.index)} with premium products from these categories.")
