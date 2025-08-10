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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set matplotlib backend to avoid GUI issues
plt.switch_backend('Agg')

# Print Gradio version for debugging, with fallback
try:
    print(f"Gradio version: {gr.__version__}")
    if gr.__version__ != '4.44.1' and gr.__version__ < '4.44.1':
        print(f"⚠️ Warning: Current Gradio version {gr.__version__} may cause compatibility issues. It is recommended to upgrade to 4.44.1 by running 'pip install gradio==4.44.1'.")
except AttributeError:
    print("⚠️ Could not retrieve Gradio version. Ensure Gradio is correctly installed.")

def clean_column_names(df, column_name):
    """
    Clean column names to ensure they are valid Python identifiers
    """
    if column_name in df.columns:
        # More robust cleaning: only keep alphanumeric and underscores, start with letter if needed
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.strip()  # Remove leading/trailing spaces
        df[column_name] = df[column_name].str.replace(r'[^\w\s]', '', regex=True)  # Remove special chars except spaces
        df[column_name] = df[column_name].str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscores
        df[column_name] = df[column_name].str.lower()  # Convert to lowercase
        # Ensure it starts with a letter or underscore
        df[column_name] = df[column_name].apply(lambda x: f"item_{x}" if x and not x[0].isalpha() and x[0] != '_' else x)
        # Remove empty strings
        df[column_name] = df[column_name].replace('', 'unknown')
    return df

def process_data(orders_file, order_products_file, products_file, aisles_file):
    """
    Process Instacart dataset to perform market basket analysis and customer clustering.
    
    Args:
        orders_file: Path to orders.csv
        order_products_file: Path to order_products__train.csv
        products_file: Path to products.csv
        aisles_file: Path to aisles.csv
    
    Returns:
        Tuple containing:
        - Association rules (str)
        - Customer clusters (str)
        - Path to visualization (str)
        - Cross-selling recommendations (str)
        - Upselling strategies (str)
    """
    # Define maximum allowed file size (200 MB)
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB in bytes
    
    try:
        # Check if all files are uploaded and validate their size
        files = [orders_file, order_products_file, products_file, aisles_file]
        file_names = ["orders.csv", "order_products__train.csv", "products.csv", "aisles.csv"]
        
        for i, (file, expected_name) in enumerate(zip(files, file_names)):
            if file is None:
                return f"❌ Please upload {expected_name}", "", None, "", ""
            
            # Validate file size
            if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
                return f"❌ {expected_name} exceeds the maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024)} MB. Current size: {file.size / (1024 * 1024):.2f} MB", "", None, "", ""
            
            # Additional file validation
            if hasattr(file, 'size') and file.size == 0:
                return f"❌ {expected_name} appears to be empty", "", None, "", ""
        
        # Handle file paths properly with better error handling
        try:
            orders_path = orders_file.name if hasattr(orders_file, 'name') else str(orders_file)
            order_products_path = order_products_file.name if hasattr(order_products_file, 'name') else str(order_products_file)
            products_path = products_file.name if hasattr(products_file, 'name') else str(products_file)
            aisles_path = aisles_file.name if hasattr(aisles_file, 'name') else str(aisles_file)
        except Exception as path_error:
            return f"❌ Error accessing uploaded files: {str(path_error)}", "", None, "", ""
        
        # Log file paths for debugging
        print(f"File paths: orders={orders_path}, order_products={order_products_path}, "
              f"products={products_path}, aisles={aisles_path}")
        
        # Load datasets and validate structure with better error handling
        print("Loading datasets...")
        try:
            orders = pd.read_csv(orders_path)
        except Exception as e:
            return f"❌ Error reading orders.csv: {str(e)}. Please check the file format.", "", None, "", ""
        
        try:
            order_products = pd.read_csv(order_products_path)
        except Exception as e:
            return f"❌ Error reading order_products.csv: {str(e)}. Please check the file format.", "", None, "", ""
        
        try:
            products = pd.read_csv(products_path)
        except Exception as e:
            return f"❌ Error reading products.csv: {str(e)}. Please check the file format.", "", None, "", ""
        
        try:
            aisles = pd.read_csv(aisles_path)
        except Exception as e:
            return f"❌ Error reading aisles.csv: {str(e)}. Please check the file format.", "", None, "", ""
        
        # Validate required columns
        required_columns = {
            'orders': ['order_id', 'user_id'],
            'order_products': ['order_id', 'product_id'],
            'products': ['product_id', 'product_name', 'aisle_id'],
            'aisles': ['aisle_id', 'aisle']
        }
        for df, name in [(orders, 'orders'), (order_products, 'order_products'), 
                         (products, 'products'), (aisles, 'aisles')]:
            missing_cols = [col for col in required_columns[name] if col not in df.columns]
            if missing_cols:
                return f"❌ Missing columns in {name}.csv: {', '.join(missing_cols)}", "", None, "", ""
        
        # Clean product and aisle names
        products = clean_column_names(products, 'product_name')
        aisles = clean_column_names(aisles, 'aisle')
        
        print(f"Loaded: {len(orders)} orders, {len(order_products)} order_products, "
              f"{len(products)} products, {len(aisles)} aisles")
        
        # Sample data for performance
        sample_size = min(10000, len(orders))
        sample_orders = orders.sample(n=sample_size, random_state=42)
        order_products_filtered = order_products[order_products['order_id'].isin(sample_orders['order_id'])]
        
        # Merge datasets with user_id
        merged_data = order_products_filtered.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
        merged_data = merged_data.merge(products, on='product_id', how='left')
        merged_data = merged_data.merge(aisles, on='aisle_id', how='left')
        
        print(f"Processing {len(merged_data)} records...")
        
        # Drop rows with missing aisle or user_id
        merged_data = merged_data.dropna(subset=['user_id', 'aisle'])
        
        # Additional validation
        if merged_data.empty:
            return "❌ No valid data after merging datasets.", "", None, "", ""
        
        # Check for sufficient unique aisles
        unique_aisles = merged_data['aisle'].unique()
        if len(unique_aisles) < 2:
            return "❌ Too few unique aisles for clustering.", "", None, "", ""
        
        print(f"Valid aisles found: {len(unique_aisles)}")
        
        # --- MARKET BASKET ANALYSIS ---
        apriori_result = ""
        try:
            # Create basket matrix with better error handling
            basket_data = merged_data.groupby(['order_id', 'product_name']).size().unstack(fill_value=0)
            
            if basket_data.empty or basket_data.shape[0] < 10:
                apriori_result = "❌ Insufficient data for market basket analysis (minimum 10 orders required)."
            else:
                # Clean column names for basket
                basket_data.columns = [f"product_{i}" if not str(col).replace('_', '').replace(' ', '').isalnum() 
                                     else str(col) for i, col in enumerate(basket_data.columns)]
                
                # Convert to boolean matrix
                basket_sets = basket_data > 0
                
                # Apply Apriori algorithm with lower threshold
                frequent_itemsets = apriori(basket_sets, min_support=0.005, use_colnames=True, max_len=3)
                
                if frequent_itemsets.empty:
                    apriori_result = "❌ No frequent itemsets found. Dataset may be too sparse."
                else:
                    # Generate association rules
                    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                    if rules.empty:
                        apriori_result = "❌ No association rules found with current thresholds."
                    else:
                        # Filter and sort rules
                        rules_filtered = rules[(rules['confidence'] >= 0.3) & (rules['lift'] >= 1.1)]
                        rules_filtered = rules_filtered.sort_values('lift', ascending=False).head(15)
                        
                        if rules_filtered.empty:
                            apriori_result = "❌ No rules meet the filtering criteria. Lowering thresholds...\n"
                            # Try with lower thresholds
                            rules_filtered = rules[(rules['confidence'] >= 0.2) & (rules['lift'] >= 1.05)]
                            rules_filtered = rules_filtered.sort_values('lift', ascending=False).head(10)
                        
                        if not rules_filtered.empty:
                            rules_text = []
                            for i, (_, row) in enumerate(rules_filtered.iterrows(), 1):
                                antecedents = ', '.join(list(row['antecedents']))
                                consequents = ', '.join(list(row['consequents']))
                                rules_text.append(
                                    f"{i}. {antecedents} ➜ {consequents}\n"
                                    f"   Support: {row['support']:.3f} | Confidence: {row['confidence']:.3f} | Lift: {row['lift']:.3f}\n"
                                )
                            apriori_result = '\n'.join(rules_text)
                        else:
                            apriori_result = "❌ No strong association patterns found in the dataset."
        except Exception as e:
            apriori_result = f"❌ Error in market basket analysis: {str(e)}"
            print(f"Market basket error: {e}")
        
        # --- CUSTOMER CLUSTERING ---
        cluster_result = ""
        plot_path = None
        
        try:
            # Create customer-aisle purchase matrix with better validation
            customer_aisle_data = merged_data.groupby(['user_id', 'aisle']).size().unstack(fill_value=0)
            
            if customer_aisle_data.empty:
                cluster_result = "❌ No data available for customer clustering."
            elif customer_aisle_data.shape[0] < 10:
                cluster_result = f"❌ Not enough customers for meaningful clustering (found {customer_aisle_data.shape[0]}, minimum 10 required)."
            elif customer_aisle_data.shape[1] < 2:
                cluster_result = f"❌ Not enough aisles for meaningful clustering (found {customer_aisle_data.shape[1]}, minimum 2 required)."
            else:
                print(f"Customer-aisle matrix shape: {customer_aisle_data.shape}")
                
                # Remove any columns with all zeros
                customer_aisle_data = customer_aisle_data.loc[:, (customer_aisle_data != 0).any(axis=0)]
                
                if customer_aisle_data.shape[1] < 2:
                    cluster_result = "❌ All aisle columns contain only zeros."
                else:
                    # Scale features with better error handling
                    scaler = StandardScaler()
                    try:
                        customer_features_scaled = scaler.fit_transform(customer_aisle_data)
                        
                        # Check for non-finite values
                        if not np.all(np.isfinite(customer_features_scaled)):
                            print("Warning: Non-finite values found after scaling, replacing with zeros")
                            customer_features_scaled = np.nan_to_num(customer_features_scaled, 0)
                        
                        # Apply K-means clustering
                        n_clusters = min(5, customer_aisle_data.shape[0] // 2, 8)  # More conservative cluster count
                        n_clusters = max(2, n_clusters)  # Ensure at least 2 clusters
                        
                        print(f"Using {n_clusters} clusters for {customer_aisle_data.shape[0]} customers")
                        
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
                        clusters = kmeans.fit_predict(customer_features_scaled)
                        
                        # Create cluster summary
                        customer_aisle_with_clusters = customer_aisle_data.copy()
                        customer_aisle_with_clusters['cluster'] = clusters
                        
                        cluster_summary = customer_aisle_with_clusters.groupby('cluster').mean()
                        cluster_summary = cluster_summary.drop('cluster', axis=1, errors='ignore')
                        
                        # Generate cluster descriptions
                        cluster_descriptions = []
                        for cluster_id in range(n_clusters):
                            cluster_customers = sum(clusters == cluster_id)
                            if cluster_customers > 0:
                                top_aisles = cluster_summary.iloc[cluster_id].nlargest(5)
                                # Filter out zero values
                                top_aisles = top_aisles[top_aisles > 0]
                                
                                if len(top_aisles) > 0:
                                    aisle_desc = ', '.join([f'{aisle.replace("_", " ").title()} ({score:.1f})' 
                                                          for aisle, score in top_aisles.items()])
                                    cluster_descriptions.append(
                                        f"🏷️ Cluster {cluster_id} ({cluster_customers} customers):\n"
                                        f"   Top aisles: {aisle_desc}\n"
                                    )
                        
                        if cluster_descriptions:
                            cluster_result = '\n'.join(cluster_descriptions)
                        else:
                            cluster_result = "❌ Unable to generate meaningful cluster descriptions."
                        
                        # Create visualization
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
                            
                            # Top aisles heatmap - select top aisles across all clusters
                            top_aisles_overall = cluster_summary.mean(axis=0).nlargest(min(10, cluster_summary.shape[1]))
                            heatmap_data = cluster_summary[top_aisles_overall.index]
                            
                            if not heatmap_data.empty:
                                # Clean column names for display
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
                            
                            # Save plot
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=tempfile.gettempdir())
                            plt.savefig(temp_file.name, dpi=300, bbox_inches='tight', facecolor='white')
                            plt.close()
                            plot_path = temp_file.name
                            print(f"Plot saved to: {plot_path}")
                            
                        except Exception as viz_error:
                            print(f"Visualization error: {viz_error}")
                            plot_path = None
                        
                    except Exception as scaling_error:
                        cluster_result = f"❌ Error in feature scaling: {str(scaling_error)}"
                        print(f"Scaling error: {scaling_error}")
        
        except Exception as e:
            cluster_result = f"❌ Error in customer clustering: {str(e)}"
            print(f"Clustering error details: {str(e)}")
        
        # --- GENERATE RECOMMENDATIONS ---
        cross_sell_recs = []
        upsell_recs = []
        
        # Cross-selling recommendations from association rules
        if 'rules_filtered' in locals() and not rules_filtered.empty:
            cross_sell_recs.append("🎯 **TOP CROSS-SELLING OPPORTUNITIES:**\n")
            for i, (_, rule) in enumerate(rules_filtered.head(3).iterrows(), 1):
                antecedents = ', '.join(list(rule['antecedents'])).replace('_', ' ').title()
                consequents = ', '.join(list(rule['consequents'])).replace('_', ' ').title()
                cross_sell_recs.append(
                    f"{i}. If customer buys {antecedents}, recommend {consequents}\n"
                    f"   → Success rate: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}x\n"
                )
        else:
            cross_sell_recs.append("❌ No strong cross-selling patterns found in current dataset.")
        
        # Upselling recommendations from clustering
        if 'cluster_summary' in locals() and not cluster_summary.empty:
            upsell_recs.append("📈 **UPSELLING STRATEGIES BY CUSTOMER SEGMENT:**\n")
            for cluster_id in range(n_clusters):
                if cluster_id < len(cluster_summary):
                    top_aisles = cluster_summary.iloc[cluster_id].nlargest(3)
                    top_aisles = top_aisles[top_aisles > 0]  # Remove zero values
                    customer_count = sum(clusters == cluster_id)
                    
                    if len(top_aisles) > 0 and customer_count > 0:
                        aisle_names = ', '.join([aisle.replace('_', ' ').title() for aisle in top_aisles.index])
                        upsell_recs.append(
                            f"🏷️ Cluster {cluster_id} ({customer_count} customers):\n"
                            f"   → Target customers buying {aisle_names} with premium products\n"
                        )
        else:
            upsell_recs.append("❌ Clustering analysis needed for upselling recommendations.")
        
        cross_sell_text = ''.join(cross_sell_recs) if cross_sell_recs else "❌ No cross-selling recommendations available."
        upsell_text = ''.join(upsell_recs) if upsell_recs else "❌ No upselling recommendations available."
        
        # Ensure we have some result even if empty
        apriori_result = apriori_result if apriori_result else "❌ Market basket analysis completed but no patterns found."
        cluster_result = cluster_result if cluster_result else "❌ Customer clustering completed but no meaningful segments found."
        
        return apriori_result, cluster_result, plot_path, cross_sell_text, upsell_text
        
    except Exception as e:
        error_msg = f"❌ Error processing data: {str(e)}\n\nPlease check that all uploaded files are valid CSV files with the expected structure."
        print(f"Full error: {e}")
        import traceback
        traceback.print_exc()
        return error_msg, "", None, "", ""

def create_app():
    """
    Create a Gradio interface with proper error handling and file upload stability.
    """
    try:
        interface = gr.Interface(
            fn=process_data,
            inputs=[
                gr.File(label="📊 orders.csv", file_types=[".csv"], file_count="single"),
                gr.File(label="🛍️ order_products__train.csv", file_types=[".csv"], file_count="single"),
                gr.File(label="📦 products.csv", file_types=[".csv"], file_count="single"),
                gr.File(label="🏪 aisles.csv", file_types=[".csv"], file_count="single")
            ],
            outputs=[
                gr.Textbox(label="Association Rules", lines=15, max_lines=25, show_copy_button=True),
                gr.Textbox(label="Customer Clusters", lines=15, max_lines=25, show_copy_button=True),
                gr.File(label="Cluster Visualization", file_types=[".png"]),
                gr.Textbox(label="Cross-Selling Recommendations", lines=10, max_lines=20, show_copy_button=True),
                gr.Textbox(label="Upselling Strategies", lines=10, max_lines=20, show_copy_button=True)
            ],
            title="🛒 Market Basket Analysis Dashboard",
            description="""
            Upload your Instacart dataset files to analyze purchasing patterns and generate insights.

            ### Instructions:
            1. Upload all 4 CSV files from the Instacart dataset (one at a time)
            2. Wait for all files to upload completely before clicking Submit
            3. Click Submit to run the analysis
            4. Review the results for marketing insights
            5. Download the visualization file to see the cluster plot

            **Note**: Analysis uses a sample for performance optimization. Please be patient during file uploads.
            Maximum file size allowed is 200 MB.
            """,
            examples=None,
            cache_examples=False,
            allow_flagging="never",
            live=False,
            batch=False
        )
        return interface
    except Exception as e:
        print(f"❌ Error creating Gradio interface: {e}")
        return gr.Interface(
            fn=lambda *args: ("Error creating interface. Please check your Gradio installation.", "", None, "", ""),
            inputs=gr.File(label="Upload any CSV file"),
            outputs=gr.Textbox(label="Error Message"),
            title="🚨 Error - Please Fix Installation"
        )

if __name__ == "__main__":
    print("🚀 Creating Gradio application...")
    
    try:
        app = create_app()
        print("✅ App created successfully")
        
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            share=False,
            show_error=True,
            quiet=True,
            max_file_size=200 * 1024 * 1024
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Gradio: {e}")
        print("\n🔧 IMMEDIATE FIX - Run these commands in order:")
        print("1. pip uninstall gradio gradio-client pydantic pydantic-core -y")
        print("2. pip install pydantic==2.8.2 pydantic-core==2.20.1")
        print("3. pip install gradio==4.44.1")
        print("4. python app.py")
        print("\nAlternatively, try the latest stable version:")
        print("pip install gradio==4.44.1")
