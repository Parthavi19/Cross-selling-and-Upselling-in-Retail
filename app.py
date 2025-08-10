import gradio as gr
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import warnings
import re
import io
import base64

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid GUI issues
plt.switch_backend('Agg')

def clean_column_names(df, column_name):
    """Clean column names to ensure they are valid Python identifiers"""
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.strip()
        df[column_name] = df[column_name].str.replace(r'[^\w\s]', '', regex=True)
        df[column_name] = df[column_name].str.replace(r'\s+', '_', regex=True)
        df[column_name] = df[column_name].str.lower()
        df[column_name] = df[column_name].apply(lambda x: f"item_{x}" if x and not x[0].isalpha() and x[0] != '_' else x)
        df[column_name] = df[column_name].replace('', 'unknown')
    return df

def validate_file_structure(file_path, expected_columns, file_name):
    """Validate CSV file structure and return error message if invalid"""
    try:
        df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for validation
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            return f"❌ {file_name} missing columns: {', '.join(missing_cols)}"
        return None
    except Exception as e:
        return f"❌ Error reading {file_name}: {str(e)}"

def create_visualization(clusters, cluster_summary):
    """Create visualization and return as base64 encoded image"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F0E68C']
        bars = ax1.bar([f'Cluster {i}' for i in cluster_counts.index], 
                      cluster_counts.values, 
                      color=colors[:len(cluster_counts)])
        ax1.set_title('Customer Cluster Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Number of Customers')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Top aisles heatmap
        top_aisles_overall = cluster_summary.mean(axis=0).nlargest(min(10, cluster_summary.shape[1]))
        heatmap_data = cluster_summary[top_aisles_overall.index]
        
        if not heatmap_data.empty:
            display_columns = [col.replace('_', ' ').title() for col in heatmap_data.columns]
            heatmap_data_display = heatmap_data.copy()
            heatmap_data_display.columns = display_columns
            
            sns.heatmap(heatmap_data_display.T, annot=True, cmap='YlOrRd', 
                       ax=ax2, cbar_kws={'label': 'Average Purchases'}, fmt='.1f')
            ax2.set_title('Top Aisles by Cluster', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Aisle')
        else:
            ax2.text(0.5, 0.5, 'No data for heatmap', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top Aisles by Cluster (No Data)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return f"data:image/png;base64,{image_base64}"
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

def process_data(orders_file, order_products_file, products_file, aisles_file):
    """Process Instacart dataset to perform market basket analysis and customer clustering"""
    
    try:
        # Validate file uploads
        files = [orders_file, order_products_file, products_file, aisles_file]
        file_names = ["orders.csv", "order_products.csv", "products.csv", "aisles.csv"]
        
        for i, (file, expected_name) in enumerate(zip(files, file_names)):
            if file is None:
                return f"❌ Please upload {expected_name}", "", "", "", ""
        
        # Validate file structures
        required_columns = {
            'orders.csv': ['order_id', 'user_id'],
            'order_products.csv': ['order_id', 'product_id'],
            'products.csv': ['product_id', 'product_name', 'aisle_id'],
            'aisles.csv': ['aisle_id', 'aisle']
        }
        
        file_paths = [file.name for file in files]
        
        for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
            error = validate_file_structure(file_path, required_columns[file_name], file_name)
            if error:
                return error, "", "", "", ""
        
        # Load datasets
        print("Loading datasets...")
        orders = pd.read_csv(file_paths[0])
        order_products = pd.read_csv(file_paths[1])
        products = pd.read_csv(file_paths[2])
        aisles = pd.read_csv(file_paths[3])
        
        # Clean product and aisle names
        products = clean_column_names(products, 'product_name')
        aisles = clean_column_names(aisles, 'aisle')
        
        # Sample data for performance (adjust based on dataset size)
        max_orders = 15000
        if len(orders) > max_orders:
            sample_orders = orders.sample(n=max_orders, random_state=42)
            order_products = order_products[order_products['order_id'].isin(sample_orders['order_id'])]
        
        # Merge datasets
        merged_data = order_products.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
        merged_data = merged_data.merge(products, on='product_id', how='left')
        merged_data = merged_data.merge(aisles, on='aisle_id', how='left')
        
        # Clean data
        merged_data = merged_data.dropna(subset=['user_id', 'aisle', 'product_name'])
        
        if merged_data.empty:
            return "❌ No valid data after merging datasets.", "", "", "", ""
        
        print(f"Processing {len(merged_data)} records from {len(merged_data['user_id'].unique())} customers...")
        
        # --- MARKET BASKET ANALYSIS ---
        apriori_result = ""
        try:
            # Create basket matrix
            basket_data = merged_data.groupby(['order_id', 'product_name']).size().unstack(fill_value=0)
            
            if basket_data.empty or basket_data.shape[0] < 20:
                apriori_result = "❌ Insufficient data for market basket analysis (minimum 20 orders required)."
            else:
                # Convert to boolean matrix
                basket_sets = basket_data > 0
                
                # Apply Apriori algorithm
                frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True, max_len=3)
                
                if frequent_itemsets.empty:
                    # Try with lower support
                    frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True, max_len=2)
                
                if frequent_itemsets.empty:
                    apriori_result = "❌ No frequent itemsets found. Dataset may be too sparse."
                else:
                    # Generate association rules
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    
                    if rules.empty:
                        apriori_result = "❌ No association rules found with current thresholds."
                    else:
                        # Filter and sort rules
                        rules_filtered = rules[(rules['confidence'] >= 0.4) & (rules['lift'] >= 1.2)]
                        
                        if rules_filtered.empty:
                            # Try with lower thresholds
                            rules_filtered = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.1)]
                        
                        if rules_filtered.empty:
                            rules_filtered = rules.sort_values('lift', ascending=False).head(10)
                        else:
                            rules_filtered = rules_filtered.sort_values('lift', ascending=False).head(15)
                        
                        if not rules_filtered.empty:
                            rules_text = []
                            rules_text.append(f"📊 **MARKET BASKET ANALYSIS RESULTS**\n")
                            rules_text.append(f"Found {len(rules_filtered)} strong association rules:\n\n")
                            
                            for i, (_, row) in enumerate(rules_filtered.iterrows(), 1):
                                antecedents = ', '.join(list(row['antecedents']))
                                consequents = ', '.join(list(row['consequents']))
                                rules_text.append(
                                    f"{i}. **{antecedents}** ➜ **{consequents}**\n"
                                    f"   • Support: {row['support']:.3f} ({row['support']*100:.1f}% of transactions)\n"
                                    f"   • Confidence: {row['confidence']:.3f} ({row['confidence']*100:.1f}% success rate)\n"
                                    f"   • Lift: {row['lift']:.3f} ({row['lift']:.1f}x more likely)\n\n"
                                )
                            apriori_result = ''.join(rules_text)
                        else:
                            apriori_result = "❌ No strong association patterns found in the dataset."
        except Exception as e:
            apriori_result = f"❌ Error in market basket analysis: {str(e)}"
        
        # --- CUSTOMER CLUSTERING ---
        cluster_result = ""
        viz_html = ""
        
        try:
            # Create customer-aisle purchase matrix
            customer_aisle_data = merged_data.groupby(['user_id', 'aisle']).size().unstack(fill_value=0)
            
            if customer_aisle_data.empty or customer_aisle_data.shape[0] < 10:
                cluster_result = f"❌ Not enough customers for clustering (found {customer_aisle_data.shape[0]}, minimum 10 required)."
            else:
                print(f"Customer-aisle matrix shape: {customer_aisle_data.shape}")
                
                # Remove columns with all zeros
                customer_aisle_data = customer_aisle_data.loc[:, (customer_aisle_data != 0).any(axis=0)]
                
                if customer_aisle_data.shape[1] < 2:
                    cluster_result = "❌ Not enough diverse aisles for meaningful clustering."
                else:
                    # Scale features
                    scaler = StandardScaler()
                    customer_features_scaled = scaler.fit_transform(customer_aisle_data)
                    
                    # Handle non-finite values
                    customer_features_scaled = np.nan_to_num(customer_features_scaled, 0)
                    
                    # Determine optimal number of clusters
                    n_clusters = min(6, max(3, customer_aisle_data.shape[0] // 20))
                    
                    print(f"Using {n_clusters} clusters for {customer_aisle_data.shape[0]} customers")
                    
                    # Apply K-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
                    clusters = kmeans.fit_predict(customer_features_scaled)
                    
                    # Create cluster summary
                    customer_aisle_with_clusters = customer_aisle_data.copy()
                    customer_aisle_with_clusters['cluster'] = clusters
                    
                    cluster_summary = customer_aisle_with_clusters.groupby('cluster').mean()
                    cluster_summary = cluster_summary.drop('cluster', axis=1, errors='ignore')
                    
                    # Generate cluster descriptions
                    cluster_descriptions = []
                    cluster_descriptions.append("👥 **CUSTOMER SEGMENTATION RESULTS**\n\n")
                    
                    for cluster_id in range(n_clusters):
                        cluster_customers = sum(clusters == cluster_id)
                        if cluster_customers > 0:
                            top_aisles = cluster_summary.iloc[cluster_id].nlargest(5)
                            top_aisles = top_aisles[top_aisles > 0.5]  # Filter meaningful purchases
                            
                            if len(top_aisles) > 0:
                                aisle_desc = ', '.join([f'{aisle.replace("_", " ").title()} ({score:.1f})' 
                                                      for aisle, score in top_aisles.items()])
                                cluster_descriptions.append(
                                    f"**🏷️ Cluster {cluster_id}** - {cluster_customers} customers ({cluster_customers/len(clusters)*100:.1f}%)\n"
                                    f"Primary shopping categories: {aisle_desc}\n"
                                    f"Behavior: {'Heavy shoppers' if top_aisles.iloc[0] > 5 else 'Moderate shoppers' if top_aisles.iloc[0] > 2 else 'Light shoppers'}\n\n"
                                )
                    
                    cluster_result = ''.join(cluster_descriptions) if cluster_descriptions else "❌ Unable to generate cluster descriptions."
                    
                    # Create visualization
                    viz_data = create_visualization(clusters, cluster_summary)
                    if viz_data:
                        viz_html = f'<img src="{viz_data}" style="max-width:100%; height:auto;">'
        
        except Exception as e:
            cluster_result = f"❌ Error in customer clustering: {str(e)}"
        
        # --- GENERATE RECOMMENDATIONS ---
        cross_sell_recs = []
        upsell_recs = []
        
        # Cross-selling recommendations
        if 'rules_filtered' in locals() and not rules_filtered.empty:
            cross_sell_recs.append("🎯 **CROSS-SELLING OPPORTUNITIES**\n\n")
            cross_sell_recs.append("*Recommend these product combinations to increase basket size:*\n\n")
            
            for i, (_, rule) in enumerate(rules_filtered.head(5).iterrows(), 1):
                antecedents = ', '.join(list(rule['antecedents'])).replace('_', ' ').title()
                consequents = ', '.join(list(rule['consequents'])).replace('_', ' ').title()
                cross_sell_recs.append(
                    f"**{i}. Bundle Opportunity:** {antecedents} + {consequents}\n"
                    f"   • Success Rate: {rule['confidence']:.1%}\n"
                    f"   • Impact: {rule['lift']:.2f}x more likely to buy together\n"
                    f"   • Action: Create bundle offers, place items nearby\n\n"
                )
        else:
            cross_sell_recs.append("❌ No strong cross-selling patterns identified. Consider collecting more transaction data.")
        
        # Upselling recommendations
        if 'cluster_summary' in locals() and not cluster_summary.empty:
            upsell_recs.append("📈 **UPSELLING STRATEGIES**\n\n")
            upsell_recs.append("*Target premium products to these customer segments:*\n\n")
            
            for cluster_id in range(n_clusters):
                if cluster_id < len(cluster_summary):
                    top_aisles = cluster_summary.iloc[cluster_id].nlargest(3)
                    top_aisles = top_aisles[top_aisles > 1]  # Meaningful purchase frequency
                    customer_count = sum(clusters == cluster_id)
                    
                    if len(top_aisles) > 0 and customer_count > 0:
                        aisle_names = ', '.join([aisle.replace('_', ' ').title() for aisle in top_aisles.index[:2]])
                        avg_purchases = top_aisles.iloc[0]
                        
                        strategy = "Premium product promotions" if avg_purchases > 5 else "Value-added bundles" if avg_purchases > 2 else "Entry-level upgrades"
                        
                        upsell_recs.append(
                            f"**Cluster {cluster_id}** ({customer_count} customers):\n"
                            f"   • Primary interests: {aisle_names}\n"
                            f"   • Strategy: {strategy}\n"
                            f"   • Approach: Target with higher-margin alternatives\n\n"
                        )
        else:
            upsell_recs.append("❌ Customer segmentation needed for targeted upselling strategies.")
        
        cross_sell_text = ''.join(cross_sell_recs)
        upsell_text = ''.join(upsell_recs)
        
        return apriori_result, cluster_result, viz_html, cross_sell_text, upsell_text
        
    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}\n\nPlease ensure all files are valid CSV files with correct structure."
        print(f"Full error: {e}")
        return error_msg, "", "", "", ""

# Create Gradio interface with updated components
def create_interface():
    with gr.Blocks(title="🛒 Market Basket Analysis Dashboard", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🛒 Market Basket Analysis Dashboard
        
        Upload your retail transaction data to discover purchasing patterns and generate actionable insights for cross-selling and upselling.
        
        ### 📋 Required Files:
        1. **orders.csv** - Order information (order_id, user_id)
        2. **order_products.csv** - Products in each order (order_id, product_id)  
        3. **products.csv** - Product details (product_id, product_name, aisle_id)
        4. **aisles.csv** - Aisle information (aisle_id, aisle)
        
        ### 🚀 What you'll get:
        - **Association Rules**: Which products are frequently bought together
        - **Customer Segments**: Different types of shoppers in your data
        - **Cross-selling Ideas**: Product bundle recommendations  
        - **Upselling Strategies**: How to increase average order value
        """)
        
        with gr.Row():
            with gr.Column():
                orders_file = gr.File(label="📊 orders.csv", file_types=[".csv"])
                order_products_file = gr.File(label="🛍️ order_products.csv", file_types=[".csv"])
            with gr.Column():
                products_file = gr.File(label="📦 products.csv", file_types=[".csv"])
                aisles_file = gr.File(label="🏪 aisles.csv", file_types=[".csv"])
        
        analyze_btn = gr.Button("🔍 Analyze Data", variant="primary", size="lg")
        
        with gr.Tabs():
            with gr.TabItem("📊 Association Rules"):
                apriori_output = gr.Markdown()
            
            with gr.TabItem("👥 Customer Segments"):
                with gr.Row():
                    with gr.Column():
                        cluster_output = gr.Markdown()
                    with gr.Column():
                        viz_output = gr.HTML()
            
            with gr.TabItem("💡 Recommendations"):
                with gr.Row():
                    with gr.Column():
                        cross_sell_output = gr.Markdown()
                    with gr.Column():
                        upsell_output = gr.Markdown()
        
        analyze_btn.click(
            fn=process_data,
            inputs=[orders_file, order_products_file, products_file, aisles_file],
            outputs=[apriori_output, cluster_output, viz_output, cross_sell_output, upsell_output]
        )
    
    return interface

if __name__ == "__main__":
    print("🚀 Starting Market Basket Analysis Dashboard...")
    
    try:
        app = create_interface()
        app.launch(
            server_name="0.0.0.0",  # Changed for deployment compatibility
            server_port=int(os.environ.get("PORT", 7860)),  # Use environment PORT or default
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        print("Please check your Gradio installation and try again.")

