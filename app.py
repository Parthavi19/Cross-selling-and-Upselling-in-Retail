import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

print("🚀 Starting Market Basket Analysis Dashboard...")
print(f"Python version: {sys.version}")
print(f"Starting on port: {os.environ.get('PORT', 8080)}")

import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import io
import base64

# Import ML libraries with error handling
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ All libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MarketBasketAnalyzer:
    """Streamlined market basket analyzer with optimized processing"""
    
    def __init__(self):
        self.max_orders = 5000  # Reduced for large file handling
        self.chunk_size = 10000  # Process in chunks
        
    def clean_names(self, df, column):
        """Fast column cleaning"""
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
        return df
    
    def validate_files(self, files, expected_columns):
        """Quick file validation with size check"""
        for i, (file, cols) in enumerate(zip(files, expected_columns)):
            if file is None:
                return f"❌ Please upload file {i+1}"
            
            # Check file size (limit to 200MB per file)
            try:
                file_size = os.path.getsize(file.name) / (1024 * 1024)  # Size in MB
                if file_size > 200:
                    return f"❌ File {i+1} too large ({file_size:.1f}MB). Please use a smaller sample."
            except:
                pass  # Continue if can't check size
                
            try:
                df = pd.read_csv(file.name, nrows=1)
                missing = [col for col in cols if col not in df.columns]
                if missing:
                    return f"❌ File {i+1} missing columns: {missing}"
            except Exception as e:
                return f"❌ File {i+1} error: {str(e)}"
        return None
    
    def load_large_file(self, file_path, columns=None, sample_frac=0.1):
        """Load large files efficiently with sampling"""
        try:
            # First, get file size and row count estimate
            with open(file_path, 'r') as f:
                first_line = f.readline()
                
            # For very large files, sample during loading
            if os.path.getsize(file_path) > 50 * 1024 * 1024:  # > 50MB
                print(f"📊 Large file detected, sampling {sample_frac*100:.0f}% of data...")
                
                # Read in chunks and sample
                chunks = []
                chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size, usecols=columns)
                
                for i, chunk in enumerate(chunk_iter):
                    # Sample from each chunk
                    sampled_chunk = chunk.sample(frac=sample_frac, random_state=42)
                    chunks.append(sampled_chunk)
                    
                    # Limit total chunks to control memory
                    if i >= 20:  # Max 20 chunks
                        break
                
                return pd.concat(chunks, ignore_index=True)
            else:
                # Load normally for smaller files
                return pd.read_csv(file_path, usecols=columns)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.read_csv(file_path, usecols=columns, nrows=10000)  # Fallback
    
    def run_analysis(self, orders_file, order_products_file, products_file, aisles_file):
        """Main analysis function with optimized processing"""
        try:
            # Validate inputs
            files = [orders_file, order_products_file, products_file, aisles_file]
            required_cols = [
                ['order_id', 'user_id'],
                ['order_id', 'product_id'], 
                ['product_id', 'product_name', 'aisle_id'],
                ['aisle_id', 'aisle']
            ]
            
            error = self.validate_files(files, required_cols)
            if error:
                return error, "", "", "", ""
            
            print("📊 Loading data efficiently...")
            
            # Load with memory optimization and sampling for large files
            orders = self.load_large_file(orders_file.name, 
                                        columns=['order_id', 'user_id'], 
                                        sample_frac=0.3)
            
            # Sample orders early to reduce downstream processing
            if len(orders) > self.max_orders:
                orders = orders.sample(n=self.max_orders, random_state=42)
                print(f"📉 Sampled to {len(orders)} orders for analysis")
            
            # Load order_products with filtering
            print("📦 Loading order products...")
            order_products = self.load_large_file(order_products_file.name, 
                                                columns=['order_id', 'product_id'],
                                                sample_frac=0.2)
            
            # Filter to sampled orders immediately to save memory
            order_products = order_products[order_products['order_id'].isin(orders['order_id'])]
            print(f"🔍 Filtered to {len(order_products)} order items")
            
            # Load smaller reference files normally
            products = pd.read_csv(products_file.name)
            aisles = pd.read_csv(aisles_file.name)
            
            # Clean names
            products = self.clean_names(products, 'product_name')
            aisles = self.clean_names(aisles, 'aisle')
            
            print(f"📈 Processing {len(order_products)} order items...")
            
            # Merge data efficiently
            data = (order_products
                   .merge(orders, on='order_id')
                   .merge(products, on='product_id')
                   .merge(aisles, on='aisle_id')
                   .dropna())
            
            if data.empty:
                return "❌ No valid data after merging", "", "", "", ""
            
            # Market Basket Analysis
            market_analysis = self.analyze_market_basket(data)
            
            # Customer Clustering  
            cluster_analysis, visualization = self.analyze_customers(data)
            
            # Generate recommendations
            cross_sell = self.generate_cross_sell_recommendations(market_analysis)
            upsell = self.generate_upsell_recommendations(cluster_analysis)
            
            return market_analysis, cluster_analysis, visualization, cross_sell, upsell
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return f"❌ Analysis failed: {str(e)}", "", "", "", ""
    
    def analyze_market_basket(self, data):
        """Optimized market basket analysis"""
        try:
            print("🛒 Running market basket analysis...")
            
            # Create transaction matrix
            basket = (data.groupby(['order_id', 'product_name'])
                     .size().unstack(fill_value=0) > 0)
            
            if basket.shape[0] < 20:
                return "❌ Need at least 20 orders for analysis"
            
            # Run Apriori with lower thresholds for speed
            frequent_items = apriori(basket, min_support=0.02, max_len=2)
            
            if frequent_items.empty:
                return "❌ No frequent patterns found"
            
            # Generate rules
            rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)
            
            if rules.empty:
                return "❌ No association rules found"
            
            # Format results
            rules = rules.sort_values('lift', ascending=False).head(10)
            
            result = ["📊 **MARKET BASKET ANALYSIS**\n"]
            result.append(f"Found {len(rules)} association rules:\n")
            
            for i, (_, rule) in enumerate(rules.iterrows(), 1):
                ant = ', '.join(rule['antecedents'])
                con = ', '.join(rule['consequents'])
                result.append(f"{i}. {ant} → {con}")
                result.append(f"   Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}\n")
            
            return '\n'.join(result)
            
        except Exception as e:
            return f"❌ Market basket error: {str(e)}"
    
    def analyze_customers(self, data):
        """Optimized customer clustering"""
        try:
            print("👥 Analyzing customer segments...")
            
            # Customer-aisle matrix
            customer_matrix = (data.groupby(['user_id', 'aisle'])
                             .size().unstack(fill_value=0))
            
            if customer_matrix.shape[0] < 10:
                return "❌ Need more customers for clustering", ""
            
            # Scale and cluster
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(customer_matrix)
            
            n_clusters = min(5, customer_matrix.shape[0] // 10)
            n_clusters = max(2, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters
            customer_matrix['cluster'] = clusters
            cluster_summary = customer_matrix.groupby('cluster').mean().drop('cluster', axis=1)
            
            # Generate description
            result = ["👥 **CUSTOMER SEGMENTS**\n"]
            
            for i in range(n_clusters):
                count = sum(clusters == i)
                top_aisles = cluster_summary.iloc[i].nlargest(3)
                top_aisles = top_aisles[top_aisles > 0]
                
                if len(top_aisles) > 0:
                    aisles_str = ', '.join(top_aisles.index)
                    result.append(f"**Segment {i+1}** ({count} customers):")
                    result.append(f"Primary interests: {aisles_str}\n")
            
            # Create simple visualization
            viz = self.create_visualization(clusters, cluster_summary)
            
            return '\n'.join(result), viz
            
        except Exception as e:
            return f"❌ Clustering error: {str(e)}", ""
    
    def create_visualization(self, clusters, cluster_summary):
        """Create lightweight visualization"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Simple bar chart of cluster sizes
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'][:len(cluster_counts)])
            
            ax.set_title('Customer Segments Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Segment')
            ax.set_ylabel('Number of Customers')
            ax.set_xticks(range(len(cluster_counts)))
            ax.set_xticklabels([f'Segment {i+1}' for i in cluster_counts.index])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;">'
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return ""
    
    def generate_cross_sell_recommendations(self, analysis):
        """Generate cross-selling recommendations"""
        if "❌" in analysis:
            return "❌ Cross-selling analysis unavailable"
        
        return """🎯 **CROSS-SELLING RECOMMENDATIONS**

**Strategy 1: Product Bundles**
- Create bundles from frequently bought-together items
- Offer discounts for bundle purchases
- Place complementary products nearby

**Strategy 2: Recommendation Engine**
- Show "customers who bought X also bought Y" 
- Implement in-store displays and online recommendations
- Train staff on complementary product suggestions

**Strategy 3: Targeted Promotions**
- Send personalized offers based on purchase history
- Create seasonal bundles for high-lift combinations
- Use email marketing for cross-sell opportunities"""
    
    def generate_upsell_recommendations(self, analysis):
        """Generate upselling recommendations"""
        if "❌" in analysis:
            return "❌ Upselling analysis unavailable"
        
        return """📈 **UPSELLING STRATEGIES**

**Strategy 1: Customer Segmentation**
- Target high-value segments with premium products
- Create loyalty programs for frequent shoppers  
- Offer exclusive products to top customers

**Strategy 2: Category Expansion**
- Introduce premium alternatives in popular categories
- Create "good-better-best" product tiers
- Highlight value-added features

**Strategy 3: Behavioral Targeting**
- Track purchase patterns to identify upgrade opportunities
- Time promotions with natural buying cycles
- Use data to personalize upgrade offers"""

def create_streamlined_interface():
    """Create fast-loading interface with progress tracking"""
    analyzer = MarketBasketAnalyzer()
    
    with gr.Blocks(title="Market Basket Analysis", theme=gr.themes.Soft()) as app:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🛒 Market Basket Analysis Dashboard</h1>
            <p>Upload your transaction data to discover purchasing patterns and generate insights</p>
            <p><strong>📝 Note:</strong> Large files will be automatically sampled for faster processing</p>
        </div>
        """)
        
        with gr.Accordion("📋 File Requirements", open=False):
            gr.Markdown("""
            **Required CSV files:**
            1. **orders.csv** - Order information (columns: order_id, user_id)
            2. **order_products.csv** - Products in each order (columns: order_id, product_id)  
            3. **products.csv** - Product details (columns: product_id, product_name, aisle_id)
            4. **aisles.csv** - Aisle information (columns: aisle_id, aisle)
            
            **💡 Tips for large files:**
            - Files over 50MB will be automatically sampled for performance
            - Maximum file size: 200MB per file
            - Processing time: 30-60 seconds for large datasets
            """)
        
        with gr.Row():
            orders_file = gr.File(label="📊 Orders CSV", file_types=[".csv"])
            order_products_file = gr.File(label="🛍️ Order Products CSV", file_types=[".csv"])
            products_file = gr.File(label="📦 Products CSV", file_types=[".csv"])
            aisles_file = gr.File(label="🏪 Aisles CSV", file_types=[".csv"])
        
        analyze_btn = gr.Button("🔍 Analyze Data", variant="primary", size="lg")
        
        # Add a status indicator
        status_text = gr.Textbox(label="📊 Analysis Status", value="Ready to analyze...", interactive=False)
        
        with gr.Row():
            with gr.Column():
                market_output = gr.Textbox(label="Market Basket Results", lines=10)
                cross_sell_output = gr.Textbox(label="Cross-Selling Strategy", lines=8)
            with gr.Column():
                cluster_output = gr.Textbox(label="Customer Segments", lines=10)
                upsell_output = gr.Textbox(label="Upselling Strategy", lines=8)
        
        viz_output = gr.HTML(label="Visualization")
        
        def update_status(status):
            return status
        
        def run_with_progress(*args):
            """Run analysis with progress updates"""
            try:
                # Update status
                yield "🔄 Starting analysis...", "", "", "", "", ""
                
                # Run the actual analysis
                results = analyzer.run_analysis(*args)
                
                # Final results
                yield "✅ Analysis complete!", *results
                
            except Exception as e:
                yield f"❌ Error: {str(e)}", "", "", "", "", ""
        
        analyze_btn.click(
            fn=run_with_progress,
            inputs=[orders_file, order_products_file, products_file, aisles_file],
            outputs=[status_text, market_output, cluster_output, viz_output, cross_sell_output, upsell_output]
        )
    
    return app

if __name__ == "__main__":
    try:
        print("✅ Creating interface...")
        app = create_streamlined_interface()
        
        print("✅ Interface created, starting server...")
        
        # Get port from environment
        port = int(os.environ.get("PORT", 8080))
        
        # Launch with optimized settings for large file uploads
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            quiet=True,
            favicon_path=None,
            show_tips=False,
            enable_queue=True,  # Enable queue for large files
            max_threads=5,
            max_file_size="500mb"  # Increase file size limit
        )
        
        print(f"✅ Server started on port {port}")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
