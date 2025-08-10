import os
import sys
import warnings
import logging
import tempfile
import io
import base64
import gc
from typing import Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterWarnings("ignore", category=FutureWarning)
warnings.filterWarnings("ignore", category=UserWarning)

# Set environment variables to reduce warnings
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"
os.environ["GRADIO_SHARE"] = "false"

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

logging.info("🚀 Starting Market Basket Analysis Dashboard...")
logging.info(f"Python version: {sys.version}")
logging.info(f"Port: {os.environ.get('PORT', 8080)}")

try:
    logging.info("✅ All libraries imported successfully")
except ImportError as e:
    logging.error(f"❌ Import error: {e}", exc_info=True)
    print(f"❌ Import error: {e}")
    sys.exit(1)

class OptimizedMarketBasketAnalyzer:
    """Optimized analyzer for production deployment with large file handling"""
    
    def __init__(self):
        self.max_orders = 50000  # Increased to handle larger datasets
        self.chunk_size = 5000
        self.sample_rates = {
            'orders': 0.2,  # Reduced for large datasets
            'order_products': 0.3,
            'large_threshold': 50 * 1024 * 1024  # 50MB
        }
        
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0
    
    def clean_product_names(self, df, column: str):
        """Clean product/aisle names efficiently"""
        if column in df.columns:
            df[column] = (df[column]
                         .astype(str)
                         .str.lower()
                         .str.replace(r'[^\w\s]', '', regex=True)
                         .str.replace(r'\s+', '_', regex=True)
                         .fillna('unknown'))
        return df
    
    def validate_csv_files(self, files: list, required_columns: list) -> Optional[str]:
        """Validate uploaded CSV files structure and size"""
        file_names = ["orders.csv", "order_products.csv", "products.csv", "aisles.csv"]
        
        for i, (file, cols, name) in enumerate(zip(files, required_columns, file_names)):
            if file is None:
                return f"❌ Please upload {name}"
            
            # Check file size
            file_size = self.get_file_size_mb(file.name)
            if file_size > 800:
                return f"❌ {name} too large ({file_size:.1f}MB). Max 800MB per file."
            
            try:
                # Quick structure validation
                test_df = pd.read_csv(file.name, nrows=5)
                missing_cols = [col for col in cols if col not in test_df.columns]
                if missing_cols:
                    return f"❌ {name} missing columns: {', '.join(missing_cols)}"
                    
                # Check for empty files
                if len(test_df) == 0:
                    return f"❌ {name} appears to be empty"
                    
            except Exception as e:
                return f"❌ Error reading {name}: {str(e)}"
        
        return None
    
    def load_file_smart(self, file_path: str, columns: list = None, sample_rate: float = 1.0):
        """Smart file loading with automatic sampling for large files"""
        file_size = self.get_file_size_mb(file_path)
        
        try:
            logging.info(f"Loading file {file_path} ({file_size:.1f}MB)")
            if file_size > 200:  # Adjusted threshold for aggressive sampling
                logging.info(f"Large file detected ({file_size:.1f}MB), using chunked loading with sample rate {sample_rate}")
                
                chunks = []
                total_rows = 0
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size, usecols=columns):
                    if sample_rate < 1.0:
                        chunk = chunk.sample(frac=sample_rate, random_state=42)
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    # Limit memory usage
                    if total_rows >= self.max_orders * 2:  # Allow some buffer
                        logging.info(f"Stopping at {total_rows:,} rows to prevent memory issues")
                        break
                
                if not chunks:
                    logging.error("No data loaded from chunks")
                    return pd.DataFrame(columns=columns)
                
                result = pd.concat(chunks, ignore_index=True)
                logging.info(f"Loaded {len(result):,} rows from {len(chunks)} chunks")
                return result
                
            else:
                # Normal loading for smaller files
                df = pd.read_csv(file_path, usecols=columns)
                logging.info(f"Loaded {len(df):,} rows from {file_path}")
                if sample_rate < 1.0:
                    df = df.sample(frac=sample_rate, random_state=42)
                    logging.info(f"Sampled to {len(df):,} rows")
                return df
                
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}", exc_info=True)
            # Fallback: load limited rows
            return pd.read_csv(file_path, usecols=columns, nrows=10000)
    
    def analyze_market_basket(self, data) -> str:
        """Perform market basket analysis with error handling"""
        try:
            logging.info("Analyzing market basket patterns")
            
            # Create transaction matrix
            basket_df = (data.groupby(['order_id', 'product_name'])
                        .size().unstack(fill_value=0))
            
            # Convert to boolean for Apriori
            basket_bool = basket_df > 0
            
            logging.info(f"Analyzing {basket_bool.shape[0]} orders with {basket_bool.shape[1]} unique products")
            
            if basket_bool.shape[0] < 20:
                return "❌ Need at least 20 orders for meaningful analysis"
            
            # Run Apriori with adaptive thresholds
            min_support = max(0.01, 10 / basket_bool.shape[0])  # Adaptive threshold
            frequent_itemsets = apriori(basket_bool, min_support=min_support, max_len=2, use_colnames=True)
            
            if frequent_itemsets.empty:
                return "❌ No frequent item patterns found. Dataset may be too sparse."
            
            # Generate association rules
            try:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
            except Exception:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            
            if rules.empty:
                return "❌ No significant association rules found"
            
            # Filter and sort rules
            good_rules = rules[
                (rules['confidence'] >= 0.3) & 
                (rules['lift'] >= 1.2) & 
                (rules['support'] >= 0.01)
            ].sort_values('lift', ascending=False).head(10)
            
            if good_rules.empty:
                # Fallback with lower thresholds
                good_rules = rules.sort_values('lift', ascending=False).head(8)
            
            # Format results
            result_lines = [
                "📊 **MARKET BASKET ANALYSIS RESULTS**\n",
                f"Analyzed {basket_bool.shape[0]:,} orders with {basket_bool.shape[1]:,} unique products\n",
                f"Found {len(good_rules)} significant association rules:\n"
            ]
            
            for i, (_, rule) in enumerate(good_rules.iterrows(), 1):
                antecedent = ', '.join(list(rule['antecedents']))
                consequent = ', '.join(list(rule['consequents']))
                
                result_lines.append(
                    f"**{i}. {antecedent}** → **{consequent}**\n"
                    f"   • Support: {rule['support']:.3f} ({rule['support']*100:.1f}% of orders)\n"
                    f"   • Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}% success rate)\n"
                    f"   • Lift: {rule['lift']:.2f}x (strength of association)\n\n"
                )
            
            return ''.join(result_lines)
            
        except Exception as e:
            logging.error(f"Market basket analysis failed: {str(e)}", exc_info=True)
            return f"❌ Market basket analysis failed: {str(e)}\nTry with a smaller dataset or different file format."
    
    def analyze_customer_segments(self, data) -> Tuple[str, str]:
        """Perform customer segmentation analysis"""
        try:
            logging.info("Analyzing customer segments")
            
            # Create customer-aisle purchase matrix
            customer_aisle = (data.groupby(['user_id', 'aisle'])
                            .size().unstack(fill_value=0))
            
            logging.info(f"Analyzing {customer_aisle.shape[0]} customers across {customer_aisle.shape[1]} aisles")
            
            if customer_aisle.shape[0] < 10:
                return "❌ Need at least 10 customers for segmentation", ""
            
            # Remove zero-variance columns
            customer_aisle = customer_aisle.loc[:, (customer_aisle != 0).any(axis=0)]
            
            if customer_aisle.shape[1] < 2:
                return "❌ Need more diverse shopping patterns for segmentation", ""
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(customer_aisle)
            
            # Handle any remaining NaN/Inf values
            scaled_features = np.nan_to_num(scaled_features, 0)
            
            # Determine optimal clusters
            n_clusters = min(6, max(2, customer_aisle.shape[0] // 15))
            
            logging.info(f"Creating {n_clusters} customer segments")
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Analyze clusters
            customer_aisle_with_clusters = customer_aisle.copy()
            customer_aisle_with_clusters['cluster'] = cluster_labels
            
            cluster_profiles = customer_aisle_with_clusters.groupby('cluster').mean()
            cluster_profiles = cluster_profiles.drop('cluster', axis=1, errors='ignore')
            
            # Generate segment descriptions
            segment_descriptions = ["👥 **CUSTOMER SEGMENTATION RESULTS**\n\n"]
            
            for cluster_id in range(n_clusters):
                cluster_size = sum(cluster_labels == cluster_id)
                cluster_pct = (cluster_size / len(cluster_labels)) * 100
                
                # Get top shopping categories for this segment
                top_categories = cluster_profiles.iloc[cluster_id].nlargest(4)
                top_categories = top_categories[top_categories > 0.5]  # Filter meaningful categories
                
                if len(top_categories) > 0:
                    categories_str = ', '.join([
                        f"{cat.replace('_', ' ').title()} ({score:.1f})" 
                        for cat, score in top_categories.items()
                    ])
                    
                    # Determine segment type
                    avg_intensity = top_categories.iloc[0]
                    if avg_intensity > 8:
                        segment_type = "Heavy Shoppers"
                    elif avg_intensity > 4:
                        segment_type = "Regular Shoppers"  
                    elif avg_intensity > 1:
                        segment_type = "Occasional Shoppers"
                    else:
                        segment_type = "Light Shoppers"
                    
                    segment_descriptions.append(
                        f"**🏷️ Segment {cluster_id + 1}: {segment_type}**\n"
                        f"Size: {cluster_size:,} customers ({cluster_pct:.1f}% of total)\n"
                        f"Primary categories: {categories_str}\n"
                        f"Shopping behavior: {self._get_behavior_insights(avg_intensity, top_categories)}\n\n"
                    )
            
            # Create visualization
            viz_html = self._create_segment_visualization(cluster_labels, cluster_profiles)
            
            return ''.join(segment_descriptions), viz_html
            
        except Exception as e:
            logging.error(f"Customer segmentation failed: {str(e)}", exc_info=True)
            return f"❌ Customer segmentation failed: {str(e)}", ""
    
    def _get_behavior_insights(self, intensity: float, categories) -> str:
        """Generate behavioral insights for customer segments"""
        if intensity > 8:
            return f"High-frequency shoppers, strong preference for {categories.index[0].replace('_', ' ')}"
        elif intensity > 4:
            return f"Moderate shoppers with focus on {len(categories)} main categories"
        elif intensity > 1:
            return "Selective shoppers, likely price-conscious"
        else:
            return "Infrequent shoppers, potential for engagement campaigns"
    
    def _create_segment_visualization(self, clusters, profiles) -> str:
        """Create customer segment visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Segment size distribution
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:len(cluster_counts)]
            
            bars = ax1.bar([f'Segment {i+1}' for i in cluster_counts.index], 
                          cluster_counts.values, color=colors)
            ax1.set_title('Customer Segment Distribution', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Segment')
            ax1.set_ylabel('Number of Customers')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
            
            # Category heatmap for top segments
            top_categories = profiles.mean(axis=0).nlargest(min(8, profiles.shape[1]))
            heatmap_data = profiles[top_categories.index]
            
            if not heatmap_data.empty:
                # Clean category names for display
                display_categories = [cat.replace('_', ' ').title() for cat in heatmap_data.columns]
                heatmap_display = heatmap_data.copy()
                heatmap_display.columns = display_categories
                
                sns.heatmap(heatmap_display.T, annot=True, cmap='YlOrRd', ax=ax2, 
                           cbar_kws={'label': 'Purchase Frequency'}, fmt='.1f')
                ax2.set_title('Shopping Patterns by Segment', fontweight='bold', fontsize=14)
                ax2.set_xlabel('Segment')
                ax2.set_ylabel('Product Category')
            
            plt.tight_layout()
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            img_data = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # Clean up memory
            gc.collect()
            
            return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto;">'
            
        except Exception as e:
            logging.error(f"Visualization error: {e}", exc_info=True)
            return "<p>⚠️ Visualization unavailable</p>"
    
    def generate_recommendations(self, market_result: str, segment_result: str) -> Tuple[str, str]:
        """Generate actionable business recommendations"""
        try:
            # Cross-selling recommendations
            if "❌" not in market_result and "association rules" in market_result.lower():
                cross_sell = """🎯 **CROSS-SELLING STRATEGY**

**Immediate Actions:**
1. **Bundle Creation**: Package frequently bought-together items at slight discount
2. **Store Layout**: Place complementary products in adjacent locations
3. **Recommendation Engine**: Show "customers who bought X also bought Y"
4. **Staff Training**: Educate sales team on high-lift product combinations

**Digital Marketing:**
• Personalized email campaigns featuring complementary products
• Website recommendations based on cart contents  
• Social media ads showcasing popular product combinations
• Mobile app push notifications for bundle offers

**Seasonal Opportunities:**
• Create themed bundles around holidays and events
• Adjust cross-sell recommendations based on seasonal patterns
• Monitor and refresh bundle offerings monthly"""
            else:
                cross_sell = """🎯 **CROSS-SELLING STRATEGY**

**Data-Driven Approach Needed:**
Since specific product associations weren't found in current data, focus on:

• **Collect More Data**: Increase transaction sample size for better patterns
• **Category-Level Analysis**: Look for broader category relationships
• **Customer Surveys**: Ask customers about product preferences
• **A/B Testing**: Experiment with different product placements
• **Seasonal Analysis**: Examine patterns across different time periods"""

            # Upselling recommendations  
            if "❌" not in segment_result and "segment" in segment_result.lower():
                upsell = """📈 **UPSELLING STRATEGY**

**Segment-Specific Approaches:**
1. **Heavy Shoppers**: Target with premium products and exclusive items
2. **Regular Shoppers**: Offer loyalty rewards for higher-value purchases
3. **Occasional Shoppers**: Use limited-time promotions to increase frequency
4. **Light Shoppers**: Focus on value propositions and starter packages

**Tactical Implementation:**
• **Email Segmentation**: Personalized offers based on shopping behavior
• **Dynamic Pricing**: Show premium options to high-value segments
• **Loyalty Programs**: Tier-based rewards encouraging larger purchases
• **Product Recommendations**: AI-driven suggestions for higher-margin items

**Success Metrics:**
• Average order value increase by segment
• Customer lifetime value improvement
• Purchase frequency changes
• Premium product adoption rates"""
            else:
                upsell = """📈 **UPSELLING STRATEGY**

**Foundation Building:**
Current segmentation needs refinement. Recommended steps:

• **Enhanced Data Collection**: Gather more customer behavior data
• **Purchase History Analysis**: Track individual customer journeys
• **Value-Based Segmentation**: Group customers by spending patterns
• **Demographic Integration**: Combine with customer demographic data
• **Behavioral Tracking**: Monitor website/app engagement patterns"""

            return cross_sell, upsell
        except Exception as e:
            logging.error(f"Recommendation generation failed: {str(e)}", exc_info=True)
            return f"❌ Recommendation generation failed: {str(e)}", ""
    
    def run_complete_analysis(self, orders_file, order_products_file, products_file, aisles_file) -> Tuple[str, str, str, str, str]:
        """Main analysis pipeline with comprehensive error handling and chunked merging"""
        try:
            logging.info("Starting complete analysis")
            # Validation phase
            files = [orders_file, order_products_file, products_file, aisles_file]
            required_columns = [
                ['order_id', 'user_id'],
                ['order_id', 'product_id'],
                ['product_id', 'product_name', 'aisle_id'], 
                ['aisle_id', 'aisle']
            ]
            
            validation_error = self.validate_csv_files(files, required_columns)
            if validation_error:
                logging.error(f"Validation failed: {validation_error}")
                return validation_error, "", "", "", ""
            
            logging.info("File validation passed")
            
            # Data loading phase
            logging.info("Loading and processing data")
            
            orders_df = self.load_file_smart(
                orders_file.name, 
                columns=['order_id', 'user_id'], 
                sample_rate=self.sample_rates['orders']
            )
            
            # Limit orders for performance
            if len(orders_df) > self.max_orders:
                orders_df = orders_df.sample(n=self.max_orders, random_state=42)
                logging.info(f"Sampled to {len(orders_df):,} orders for analysis")
            
            # Load order products filtered by sampled orders
            order_products_df = self.load_file_smart(
                order_products_file.name,
                columns=['order_id', 'product_id'],
                sample_rate=self.sample_rates['order_products']
            )
            
            # Filter to our order sample
            order_products_df = order_products_df[
                order_products_df['order_id'].isin(orders_df['order_id'])
            ]
            
            logging.info(f"Processing {len(order_products_df):,} order items")
            
            # Load reference data
            products_df = pd.read_csv(products_file.name)
            aisles_df = pd.read_csv(aisles_file.name)
            
            # Clean reference data
            products_df = self.clean_product_names(products_df, 'product_name')
            aisles_df = self.clean_product_names(aisles_df, 'aisle')
            
            # Chunked merging to reduce memory usage
            logging.info("Merging datasets in chunks")
            merged_data_chunks = []
            chunk_size = 10000  # Smaller chunks for merging
            for start in range(0, len(order_products_df), chunk_size):
                chunk = order_products_df.iloc[start:start + chunk_size]
                merged_chunk = (chunk
                               .merge(orders_df[['order_id', 'user_id']], on='order_id', how='left')
                               .merge(products_df, on='product_id', how='left')
                               .merge(aisles_df, on='aisle_id', how='left')
                               .dropna(subset=['user_id', 'product_name', 'aisle']))
                merged_data_chunks.append(merged_chunk)
                logging.info(f"Processed merge chunk {start//chunk_size + 1} with {len(merged_chunk):,} rows")
            
            if not merged_data_chunks:
                logging.error("No data remains after merging")
                return "❌ No data remains after merging. Check file compatibility.", "", "", "", ""
            
            merged_data = pd.concat(merged_data_chunks, ignore_index=True)
            logging.info(f"Final dataset: {len(merged_data):,} records from {merged_data['user_id'].nunique():,} customers")
            
            # Analysis phase
            logging.info("Running market basket analysis")
            market_analysis = self.analyze_market_basket(merged_data)
            
            logging.info("Running customer segmentation")
            segment_analysis, visualization = self.analyze_customer_segments(merged_data)
            
            logging.info("Generating recommendations")
            cross_sell_recs, upsell_recs = self.generate_recommendations(market_analysis, segment_analysis)
            
            # Memory cleanup
            del merged_data, order_products_df, orders_df, merged_data_chunks
            gc.collect()
            
            logging.info("Analysis complete")
            return market_analysis, segment_analysis, visualization, cross_sell_recs, upsell_recs
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}\n\nPlease check your files and try again."
            logging.error(f"Analysis failed: {str(e)}", exc_info=True)
            return error_msg, "", "", "", ""

def create_production_interface():
    """Create production-ready Gradio interface"""
    
    analyzer = OptimizedMarketBasketAnalyzer()
    
    with gr.Blocks(
        title="Market Basket Analysis Dashboard",
        theme=gr.themes.Base(),
        css="body {font-family: Arial, sans-serif; max-width: 1200px; margin: auto;}"
    ) as gradio_app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
            <h1 style="margin: 0; font-size: 2em;">🛒 Market Basket Analysis Dashboard</h1>
            <p style="margin: 10px 0; font-size: 1em;">Transform your transaction data into actionable business insights</p>
        </div>
        """)
        
        with gr.Accordion("📋 Quick Start Guide", open=False):
            gr.Markdown("""
            **Upload these 4 CSV files from your retail/e-commerce platform:**
            
            | File | Required Columns | Description |
            |------|------------------|-------------|
            | **orders.csv** | `order_id`, `user_id` | Order information linking customers to purchases |
            | **order_products.csv** | `order_id`, `product_id` | Which products were in each order |
            | **products.csv** | `product_id`, `product_name`, `aisle_id` | Product details and categorization |
            | **aisles.csv** | `aisle_id`, `aisle` | Product category/aisle information |
            
            **💡 Performance Tips:**
            - Files up to 800MB supported with automatic optimization
            - Large datasets (>1M rows) are sampled to manage memory and speed
            - Analysis may take 1-5 minutes for very large datasets
            - Ensure sufficient memory (4GB recommended) for large files
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                orders_file = gr.File(
                    label="📊 Orders CSV", 
                    file_types=[".csv"],
                    elem_id="orders_upload"
                )
                products_file = gr.File(
                    label="📦 Products CSV", 
                    file_types=[".csv"],
                    elem_id="products_upload"
                )
            
            with gr.Column(scale=1):
                order_products_file = gr.File(
                    label="🛍️ Order Products CSV", 
                    file_types=[".csv"],
                    elem_id="order_products_upload"
                )
                aisles_file = gr.File(
                    label="🏪 Aisles CSV", 
                    file_types=[".csv"],
                    elem_id="aisles_upload"
                )
        
        analyze_button = gr.Button(
            "🚀 Run Analysis", 
            variant="primary", 
            size="lg",
            elem_id="analyze_btn"
        )
        
        gr.HTML("<hr style='margin: 20px 0;'>")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("📊 Market Basket Analysis"):
                market_results = gr.Markdown(
                    value="Upload your files and click 'Run Analysis' to see market basket results...",
                    elem_id="market_results"
                )
            
            with gr.TabItem("👥 Customer Segments"):
                with gr.Row():
                    with gr.Column(scale=2):
                        segment_results = gr.Markdown(
                            value="Customer segmentation results will appear here...",
                            elem_id="segment_results"
                        )
                    with gr.Column(scale=1):
                        segment_viz = gr.HTML(
                            value="<div style='text-align:center; padding: 50px; color: #666;'>Visualization will appear after analysis</div>",
                            elem_id="segment_viz"
                        )
            
            with gr.TabItem("💡 Business Recommendations"):
                with gr.Row():
                    with gr.Column():
                        cross_sell_recs = gr.Markdown(
                            value="Cross-selling recommendations will be generated based on your analysis...",
                            elem_id="cross_sell"
                        )
                    with gr.Column():
                        upsell_recs = gr.Markdown(
                            value="Upselling strategies will be provided after analysis...",
                            elem_id="upsell"
                        )
        
        # Event handler
        analyze_button.click(
            fn=analyzer.run_complete_analysis,
            inputs=[orders_file, order_products_file, products_file, aisles_file],
            outputs=[market_results, segment_results, segment_viz, cross_sell_recs, upsell_recs],
            show_progress=True
        )
        
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; text-align: center;">
            <p style="margin: 0; color: #666;"><strong>🔧 Built with Gradio</strong> | Optimized for production deployment</p>
        </div>
        """)
    
    return gradio_app

# Create FastAPI app
app = FastAPI()

# Health check endpoint for Cloud Run
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)}"}
    )

# Mount Gradio app using Gradio's integration
gradio_app = create_production_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    try:
        logging.info("✅ Creating optimized interface...")
        print("✅ Market Basket Analysis Dashboard is ready!")
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}", exc_info=True)
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
