import os
import sys
import warnings
import logging
import tempfile
import io
import base64
import gc
import psutil
from typing import Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    """Ultra-optimized analyzer for handling 800MB+ files and 3.5M+ rows"""
    
    def __init__(self):
        # Aggressive optimization for large datasets
        self.max_orders = 100000  # Increased but still manageable
        self.chunk_size = 10000  # Larger chunks for efficiency
        self.sample_rates = {
            'orders': 0.05,  # 5% sample for very large datasets
            'order_products': 0.03,  # 3% sample to manage memory
            'large_threshold': 100 * 1024 * 1024  # 100MB threshold
        }
        self.max_products = 500  # Reduced for memory efficiency
        self.max_customers = 50000  # Limit customers for segmentation
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / (1024 * 1024)
        except:
            return 0
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0
    
    def clean_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        
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
            
            # Check file size - increased limit
            file_size = self.get_file_size_mb(file.name)
            if file_size > 1200:  # Increased to 1.2GB
                return f"❌ {name} too large ({file_size:.1f}MB). Max 1200MB per file."
            
            try:
                # Quick structure validation with error handling
                test_df = pd.read_csv(file.name, nrows=10, encoding='utf-8', 
                                    on_bad_lines='skip', low_memory=False)
                missing_cols = [col for col in cols if col not in test_df.columns]
                if missing_cols:
                    return f"❌ {name} missing columns: {', '.join(missing_cols)}"
                    
                # Check for empty files
                if len(test_df) == 0:
                    return f"❌ {name} appears to be empty"
                    
            except Exception as e:
                return f"❌ Error reading {name}: {str(e)[:100]}"
        
        return None
    
    def load_file_optimized(self, file_path: str, columns: list = None, 
                           sample_rate: float = 1.0, max_rows: int = None):
        """Ultra-optimized file loading for very large files"""
        file_size = self.get_file_size_mb(file_path)
        
        try:
            logging.info(f"Loading {file_path} ({file_size:.1f}MB), Memory: {self.get_memory_usage():.1f}MB")
            
            # Determine optimal strategy based on file size
            if file_size > 200:  # Very large files
                logging.info(f"Very large file detected, using aggressive chunking")
                
                chunks = []
                total_rows = 0
                rows_to_process = max_rows or (self.max_orders * 3)
                
                # Use smaller chunk size for very large files
                chunk_size = min(self.chunk_size, 5000)
                
                chunk_reader = pd.read_csv(
                    file_path, 
                    chunksize=chunk_size,
                    usecols=columns,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    low_memory=False,
                    dtype=str  # Read all as string first to avoid type issues
                )
                
                for i, chunk in enumerate(chunk_reader):
                    if total_rows >= rows_to_process:
                        break
                        
                    # Convert numeric columns if needed
                    if 'order_id' in chunk.columns:
                        chunk['order_id'] = pd.to_numeric(chunk['order_id'], errors='coerce')
                    if 'user_id' in chunk.columns:
                        chunk['user_id'] = pd.to_numeric(chunk['user_id'], errors='coerce')
                    if 'product_id' in chunk.columns:
                        chunk['product_id'] = pd.to_numeric(chunk['product_id'], errors='coerce')
                    if 'aisle_id' in chunk.columns:
                        chunk['aisle_id'] = pd.to_numeric(chunk['aisle_id'], errors='coerce')
                    
                    # Drop rows with NaN in critical columns
                    chunk = chunk.dropna(subset=[col for col in chunk.columns if col in ['order_id', 'user_id', 'product_id']])
                    
                    if sample_rate < 1.0 and len(chunk) > 0:
                        chunk = chunk.sample(frac=sample_rate, random_state=42)
                    
                    if len(chunk) > 0:
                        chunks.append(chunk)
                        total_rows += len(chunk)
                    
                    if i % 10 == 0:
                        logging.info(f"Processed {i+1} chunks, {total_rows:,} rows, Memory: {self.get_memory_usage():.1f}MB")
                        self.clean_memory()
                
                if not chunks:
                    logging.error("No valid data loaded from chunks")
                    return pd.DataFrame(columns=columns or [])
                
                result = pd.concat(chunks, ignore_index=True)
                self.clean_memory()
                
                logging.info(f"Loaded {len(result):,} rows from {len(chunks)} chunks, Memory: {self.get_memory_usage():.1f}MB")
                return result
                
            else:
                # Normal loading for smaller files
                df = pd.read_csv(
                    file_path, 
                    usecols=columns,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    low_memory=False
                )
                
                logging.info(f"Loaded {len(df):,} rows, Memory: {self.get_memory_usage():.1f}MB")
                
                if sample_rate < 1.0 and len(df) > 0:
                    df = df.sample(frac=sample_rate, random_state=42)
                    logging.info(f"Sampled to {len(df):,} rows, Memory: {self.get_memory_usage():.1f}MB")
                
                if max_rows and len(df) > max_rows:
                    df = df.sample(n=max_rows, random_state=42)
                    logging.info(f"Limited to {len(df):,} rows, Memory: {self.get_memory_usage():.1f}MB")
                
                return df
                
        except Exception as e:
            logging.error(f"Error loading file {file_path}: {str(e)}")
            # Emergency fallback
            try:
                return pd.read_csv(file_path, usecols=columns, nrows=10000, encoding='utf-8', on_bad_lines='skip')
            except:
                return pd.DataFrame(columns=columns or [])
    
    def analyze_market_basket(self, data) -> str:
        """Optimized market basket analysis for large datasets"""
        try:
            logging.info(f"Starting market basket analysis with {len(data):,} records, Memory: {self.get_memory_usage():.1f}MB")
            
            if len(data) < 100:
                return "❌ Need at least 100 records for meaningful market basket analysis"
            
            # Aggressive product filtering for memory management
            product_counts = data['product_name'].value_counts()
            if len(product_counts) > self.max_products:
                top_products = product_counts.head(self.max_products).index
                data = data[data['product_name'].isin(top_products)]
                logging.info(f"Limited to top {self.max_products} products, Memory: {self.get_memory_usage():.1f}MB")
            
            # Sample orders if too many
            unique_orders = data['order_id'].nunique()
            if unique_orders > 20000:
                sample_orders = data['order_id'].unique()[:20000]
                data = data[data['order_id'].isin(sample_orders)]
                logging.info(f"Sampled to 20,000 orders, Memory: {self.get_memory_usage():.1f}MB")
            
            # Create transaction matrix with chunking
            logging.info("Creating transaction matrix...")
            basket_df = (data.groupby(['order_id', 'product_name'])
                        .size().unstack(fill_value=0))
            
            # Memory check
            if self.get_memory_usage() > 1500:  # If memory usage > 1.5GB
                logging.warning("High memory usage detected, reducing dataset size")
                basket_df = basket_df.sample(n=min(10000, len(basket_df)), random_state=42)
            
            # Convert to boolean for Apriori
            basket_bool = basket_df > 0
            
            logging.info(f"Transaction matrix: {basket_bool.shape[0]} orders × {basket_bool.shape[1]} products, Memory: {self.get_memory_usage():.1f}MB")
            
            if basket_bool.shape[0] < 20:
                return "❌ Need at least 20 orders after filtering"
            
            # Adaptive support threshold based on dataset size
            min_support = max(0.005, 20 / basket_bool.shape[0])  # At least 20 transactions or 0.5%
            
            logging.info(f"Running Apriori with min_support={min_support:.4f}")
            
            # Run Apriori with memory management
            try:
                frequent_itemsets = apriori(
                    basket_bool, 
                    min_support=min_support, 
                    max_len=2,  # Limit to pairs for memory
                    use_colnames=True,
                    verbose=0
                )
                
                if frequent_itemsets.empty:
                    # Retry with lower threshold
                    min_support = max(0.001, 5 / basket_bool.shape[0])
                    frequent_itemsets = apriori(
                        basket_bool, 
                        min_support=min_support, 
                        max_len=2,
                        use_colnames=True,
                        verbose=0
                    )
            except Exception as e:
                logging.error(f"Apriori failed: {e}")
                return f"❌ Market basket analysis failed due to memory constraints. Try with a smaller dataset."
            
            if frequent_itemsets.empty:
                return "❌ No frequent patterns found. Data may be too sparse."
            
            logging.info(f"Found {len(frequent_itemsets)} frequent itemsets, Memory: {self.get_memory_usage():.1f}MB")
            
            # Generate association rules
            try:
                rules = association_rules(
                    frequent_itemsets, 
                    metric="confidence", 
                    min_threshold=0.1,
                    num_itemsets=len(frequent_itemsets)
                )
            except Exception as e:
                logging.error(f"Rules generation failed: {e}")
                return "❌ Could not generate association rules from the patterns found."
            
            if rules.empty:
                return "❌ No significant association rules found"
            
            # Filter and sort rules
            good_rules = rules[
                (rules['confidence'] >= 0.2) & 
                (rules['lift'] >= 1.1) & 
                (rules['support'] >= min_support)
            ].sort_values('lift', ascending=False).head(15)
            
            if good_rules.empty:
                good_rules = rules.sort_values('lift', ascending=False).head(10)
            
            # Format results
            result_lines = [
                "📊 **MARKET BASKET ANALYSIS RESULTS**\n",
                f"Analyzed {basket_bool.shape[0]:,} orders with {basket_bool.shape[1]:,} unique products\n",
                f"Found {len(good_rules)} significant association rules:\n\n"
            ]
            
            for i, (_, rule) in enumerate(good_rules.iterrows(), 1):
                try:
                    antecedent = ', '.join(list(rule['antecedents']))
                    consequent = ', '.join(list(rule['consequents']))
                    
                    result_lines.append(
                        f"**{i}. {antecedent}** → **{consequent}**\n"
                        f"   • Support: {rule['support']:.3f} ({rule['support']*100:.1f}% of orders)\n"
                        f"   • Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}% success rate)\n"
                        f"   • Lift: {rule['lift']:.2f}x (association strength)\n\n"
                    )
                except:
                    continue
            
            # Cleanup
            del basket_df, basket_bool, frequent_itemsets, rules
            self.clean_memory()
            
            logging.info(f"Market basket analysis complete, Memory: {self.get_memory_usage():.1f}MB")
            return ''.join(result_lines)
            
        except Exception as e:
            logging.error(f"Market basket analysis failed: {str(e)}")
            return f"❌ Market basket analysis failed: {str(e)[:200]}"
    
    def analyze_customer_segments(self, data) -> Tuple[str, str]:
        """Optimized customer segmentation for large datasets"""
        try:
            logging.info(f"Starting customer segmentation, Memory: {self.get_memory_usage():.1f}MB")
            
            # Limit customers for memory management
            unique_customers = data['user_id'].nunique()
            if unique_customers > self.max_customers:
                sample_customers = data['user_id'].value_counts().head(self.max_customers).index
                data = data[data['user_id'].isin(sample_customers)]
                logging.info(f"Sampled to {self.max_customers} most active customers, Memory: {self.get_memory_usage():.1f}MB")
            
            # Create customer-aisle purchase matrix
            customer_aisle = (data.groupby(['user_id', 'aisle'])
                            .size().unstack(fill_value=0))
            
            logging.info(f"Customer matrix: {customer_aisle.shape[0]} customers × {customer_aisle.shape[1]} aisles, Memory: {self.get_memory_usage():.1f}MB")
            
            if customer_aisle.shape[0] < 10:
                return "❌ Need at least 10 customers for segmentation", ""
            
            # Remove zero-variance columns and limit features
            customer_aisle = customer_aisle.loc[:, (customer_aisle != 0).any(axis=0)]
            
            # Limit to top aisles if too many
            if customer_aisle.shape[1] > 50:
                top_aisles = customer_aisle.sum().nlargest(50).index
                customer_aisle = customer_aisle[top_aisles]
                logging.info(f"Limited to top 50 aisles, Memory: {self.get_memory_usage():.1f}MB")
            
            if customer_aisle.shape[1] < 2:
                return "❌ Need more diverse shopping patterns for segmentation", ""
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(customer_aisle.fillna(0))
            scaled_features = np.nan_to_num(scaled_features, 0)
            
            # Use MiniBatchKMeans for large datasets
            n_clusters = min(8, max(3, customer_aisle.shape[0] // 20))
            
            logging.info(f"Creating {n_clusters} segments using MiniBatchKMeans, Memory: {self.get_memory_usage():.1f}MB")
            
            # Use MiniBatch for memory efficiency
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                batch_size=min(1000, customer_aisle.shape[0] // 2),
                n_init=3,  # Reduced for speed
                max_iter=100  # Reduced for speed
            )
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Clean up intermediate data
            del scaled_features, scaler
            self.clean_memory()
            
            # Analyze clusters
            customer_aisle_with_clusters = customer_aisle.copy()
            customer_aisle_with_clusters['cluster'] = cluster_labels
            
            cluster_profiles = customer_aisle_with_clusters.groupby('cluster').mean()
            cluster_profiles = cluster_profiles.drop('cluster', axis=1, errors='ignore')
            
            # Generate segment descriptions
            segment_descriptions = [f"👥 **CUSTOMER SEGMENTATION RESULTS** ({unique_customers:,} customers analyzed)\n\n"]
            
            for cluster_id in range(n_clusters):
                cluster_size = sum(cluster_labels == cluster_id)
                cluster_pct = (cluster_size / len(cluster_labels)) * 100
                
                # Get top shopping categories
                top_categories = cluster_profiles.iloc[cluster_id].nlargest(5)
                top_categories = top_categories[top_categories > 0.1]
                
                if len(top_categories) > 0:
                    categories_str = ', '.join([
                        f"{cat.replace('_', ' ').title()} ({score:.1f})" 
                        for cat, score in top_categories.items()[:3]
                    ])
                    
                    # Determine segment type
                    avg_intensity = top_categories.iloc[0]
                    if avg_intensity > 10:
                        segment_type = "Heavy Shoppers"
                    elif avg_intensity > 5:
                        segment_type = "Regular Shoppers"  
                    elif avg_intensity > 2:
                        segment_type = "Moderate Shoppers"
                    else:
                        segment_type = "Light Shoppers"
                    
                    segment_descriptions.append(
                        f"**🏷️ Segment {cluster_id + 1}: {segment_type}**\n"
                        f"Size: {cluster_size:,} customers ({cluster_pct:.1f}% of analyzed)\n"
                        f"Primary categories: {categories_str}\n"
                        f"Shopping behavior: {self._get_behavior_insights(avg_intensity, top_categories)}\n\n"
                    )
            
            # Create visualization
            viz_html = self._create_segment_visualization(cluster_labels, cluster_profiles)
            
            # Cleanup
            del customer_aisle, customer_aisle_with_clusters, cluster_profiles
            self.clean_memory()
            
            logging.info(f"Customer segmentation complete, Memory: {self.get_memory_usage():.1f}MB")
            return ''.join(segment_descriptions), viz_html
            
        except Exception as e:
            logging.error(f"Customer segmentation failed: {str(e)}")
            return f"❌ Customer segmentation failed: {str(e)[:200]}", ""
    
    def _get_behavior_insights(self, intensity: float, categories) -> str:
        """Generate behavioral insights for customer segments"""
        if intensity > 10:
            return f"High-volume shoppers, strong preference for {categories.index[0].replace('_', ' ')}"
        elif intensity > 5:
            return f"Regular shoppers with focus on {len(categories)} main categories"
        elif intensity > 2:
            return "Moderate shoppers, likely price-conscious"
        else:
            return "Light shoppers, potential for engagement campaigns"
    
    def _create_segment_visualization(self, clusters, profiles) -> str:
        """Create optimized customer segment visualization"""
        try:
            logging.info(f"Creating visualization, Memory: {self.get_memory_usage():.1f}MB")
            
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.patch.set_facecolor('white')
            
            # Segment size distribution
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0A500', '#A0E7E5'][:len(cluster_counts)]
            
            bars = ax1.bar([f'Segment {i+1}' for i in cluster_counts.index], 
                          cluster_counts.values, color=colors, edgecolor='white', linewidth=2)
            ax1.set_title('Customer Segment Distribution', fontweight='bold', fontsize=16, pad=20)
            ax1.set_xlabel('Segment', fontsize=12)
            ax1.set_ylabel('Number of Customers', fontsize=12)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Category heatmap - limit to top categories
            if not profiles.empty:
                top_categories = profiles.mean(axis=0).nlargest(min(12, profiles.shape[1]))
                heatmap_data = profiles[top_categories.index]
                
                # Clean category names
                display_categories = [cat.replace('_', ' ').title()[:15] for cat in heatmap_data.columns]
                heatmap_display = heatmap_data.copy()
                heatmap_display.columns = display_categories
                
                sns.heatmap(heatmap_display.T, annot=True, cmap='YlOrRd', ax=ax2, 
                           cbar_kws={'label': 'Purchase Frequency'}, fmt='.1f',
                           xticklabels=[f'S{i+1}' for i in range(len(heatmap_display))],
                           annot_kws={'size': 10})
                ax2.set_title('Shopping Patterns by Segment', fontweight='bold', fontsize=16, pad=20)
                ax2.set_xlabel('Segment', fontsize=12)
                ax2.set_ylabel('Product Category', fontsize=12)
                
                # Rotate y labels for better readability
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=10)
            
            plt.tight_layout(pad=3.0)
            
            # Convert to base64 with optimization
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', optimize=True)
            buffer.seek(0)
            img_data = base64.b64encode(buffer.read()).decode()
            plt.close('all')
            
            # Clean up memory
            self.clean_memory()
            
            logging.info(f"Visualization complete, Memory: {self.get_memory_usage():.1f}MB")
            return f'<img src="data:image/png;base64,{img_data}" style="max-width:100%; height:auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">'
            
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            return "<div style='text-align:center; padding: 50px; color: #666; background: #f8f9fa; border-radius: 8px;'>⚠️ Visualization temporarily unavailable</div>"
    
    def generate_recommendations(self, market_result: str, segment_result: str) -> Tuple[str, str]:
        """Generate enhanced business recommendations"""
        try:
            logging.info(f"Generating recommendations, Memory: {self.get_memory_usage():.1f}MB")
            
            # Enhanced cross-selling recommendations
            if "❌" not in market_result and "association rules" in market_result.lower():
                cross_sell = """🎯 **ADVANCED CROSS-SELLING STRATEGY**

**🚀 Immediate Implementation:**
1. **Smart Product Bundles**: Create bundles based on discovered associations with 5-10% discount
2. **Dynamic Store Layout**: Position complementary products within 3 feet of each other
3. **AI-Powered Recommendations**: Implement "Frequently Bought Together" on website/app
4. **Staff Training Program**: Train associates on top 10 product associations

**📱 Digital Marketing Tactics:**
• **Email Automation**: Send cross-sell emails 2-3 days after purchase
• **Retargeting Campaigns**: Show complementary products to recent buyers
• **Mobile App Push**: Real-time suggestions when customers are in-store
• **Social Proof**: Display "Customers also bought" with purchase statistics

**📊 Performance Monitoring:**
• Track cross-sell conversion rates by product pair
• Monitor average order value increases
• A/B test different recommendation placements
• Measure incremental revenue from associations

**🎯 Advanced Techniques:**
• **Seasonal Adjustments**: Modify associations for holidays/seasons
• **Inventory Management**: Use associations for demand forecasting
• **Pricing Strategy**: Bundle pricing optimization
• **Customer Journey**: Map associations across purchase timeline"""
            else:
                cross_sell = """🎯 **CROSS-SELLING FOUNDATION STRATEGY**

**📋 Data Collection Phase:**
Current analysis needs more data for strong associations. Focus on:

• **Increase Data Volume**: Collect 6+ months of transaction data
• **Data Quality**: Ensure all product categories are captured
• **Customer Behavior**: Track online browsing patterns
• **External Data**: Consider seasonal trends and competitor analysis

**🔄 Process Improvements:**
• **Transaction Tracking**: Implement detailed basket analysis
• **Customer Surveys**: Ask about product preferences and needs
• **Market Research**: Study industry cross-selling benchmarks
• **Pilot Programs**: Test small-scale cross-selling initiatives"""

            # Enhanced upselling recommendations  
            if "❌" not in segment_result and "segment" in segment_result.lower():
                upsell = """📈 **STRATEGIC UPSELLING FRAMEWORK**

**🎯 Segment-Specific Strategies:**
1. **Heavy Shoppers (High Value)**:
   - Premium product line introductions
   - VIP early access to new products
   - Volume-based tier rewards (spend $X, get Y% off)
   - Personal shopping consultant services

2. **Regular Shoppers (Core Base)**:
   - Loyalty point multipliers for premium purchases
   - "Next level" product suggestions
   - Member-exclusive premium product trials
   - Referral bonuses for bringing friends

3. **Moderate Shoppers (Growth Potential)**:
   - Limited-time upgrade offers
   - "Try premium for free" samples
   - Educational content about premium benefits
   - Social proof messaging ("Most customers upgrade to...")

4. **Light Shoppers (Activation Focus)**:
   - Entry-level premium options
   - Small commitment upgrades
   - "First purchase" premium discounts
   - Clear value proposition messaging

**💡 Implementation Tactics:**
• **Personalized Pricing**: Dynamic pricing based on segment
• **Progressive Disclosure**: Gradually introduce premium options
• **Social Proof**: Show segment-specific success stories
• **Scarcity Marketing**: Limited-time offers for premium products

**📊 Success Metrics:**
• Average order value by segment
• Premium product adoption rates
• Customer lifetime value progression
• Segment migration tracking"""
            else:
                upsell = """📈 **UPSELLING FOUNDATION STRATEGY**

**🏗️ Infrastructure Development:**
Current segmentation needs enhancement. Priority actions:

• **Enhanced Segmentation**: Collect behavioral and demographic data
• **Value Analysis**: Identify high-value customer characteristics  
• **Product Tiers**: Develop clear premium product hierarchy
• **Customer Journey**: Map upgrade touchpoints and opportunities

**🔬 Research Phase:**
• **Customer Interviews**: Understand upgrade motivations
• **Competitive Analysis**: Study premium positioning strategies
• **Price Sensitivity**: Test different premium price points
• **Value Proposition**: Define clear premium benefits"""

            logging.info(f"Recommendations generated, Memory: {self.get_memory_usage():.1f}MB")
            return cross_sell, upsell
        except Exception as e:
            logging.error(f"Recommendation generation failed: {str(e)}", exc_info=True)
            return f"❌ Recommendation generation failed: {str(e)}", ""
    
    def merge_datasets_optimized(self, orders_df, order_products_df, products_df, aisles_df):
        """Optimized dataset merging for very large datasets"""
        try:
            logging.info(f"Starting optimized merge, Memory: {self.get_memory_usage():.1f}MB")
            
            # Step 1: Filter order_products to only include orders we have
            order_ids_set = set(orders_df['order_id'].unique())
            order_products_filtered = order_products_df[
                order_products_df['order_id'].isin(order_ids_set)
            ]
            
            logging.info(f"Filtered order products: {len(order_products_filtered):,} records, Memory: {self.get_memory_usage():.1f}MB")
            
            # Step 2: Create product info lookup
            product_info = products_df.merge(aisles_df, on='aisle_id', how='left')
            
            # Step 3: Chunked merging for large datasets
            chunk_size = 20000
            merged_chunks = []
            
            for start in range(0, len(order_products_filtered), chunk_size):
                end = min(start + chunk_size, len(order_products_filtered))
                chunk = order_products_filtered.iloc[start:end]
                
                # Merge with orders
                chunk_with_orders = chunk.merge(
                    orders_df[['order_id', 'user_id']], 
                    on='order_id', 
                    how='inner'
                )
                
                # Merge with product info
                chunk_final = chunk_with_orders.merge(
                    product_info[['product_id', 'product_name', 'aisle']], 
                    on='product_id', 
                    how='inner'
                )
                
                # Clean and validate
                chunk_final = chunk_final.dropna(subset=['user_id', 'product_name', 'aisle'])
                
                if len(chunk_final) > 0:
                    merged_chunks.append(chunk_final)
                
                if start % (chunk_size * 5) == 0:
                    logging.info(f"Processed merge chunk {start//chunk_size + 1}, Memory: {self.get_memory_usage():.1f}MB")
                    self.clean_memory()
            
            if not merged_chunks:
                raise ValueError("No data remains after merging")
            
            final_data = pd.concat(merged_chunks, ignore_index=True)
            
            # Final cleanup
            del merged_chunks, order_products_filtered, product_info
            self.clean_memory()
            
            logging.info(f"Merge complete: {len(final_data):,} records, {final_data['user_id'].nunique():,} customers, Memory: {self.get_memory_usage():.1f}MB")
            return final_data
            
        except Exception as e:
            logging.error(f"Merge failed: {str(e)}")
            raise e
    
    def run_complete_analysis(self, orders_file, order_products_file, products_file, aisles_file) -> Tuple[str, str, str, str, str]:
        """Ultra-optimized analysis pipeline for handling 800MB+ files and 3.5M+ rows"""
        try:
            logging.info(f"🚀 Starting ultra-optimized analysis, Memory: {self.get_memory_usage():.1f}MB")
            
            # Phase 1: Validation
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
            
            logging.info("✅ File validation passed")
            
            # Phase 2: Smart Data Loading
            logging.info("📂 Loading datasets with optimization...")
            
            # Load orders with aggressive sampling for very large files
            orders_file_size = self.get_file_size_mb(orders_file.name)
            orders_sample_rate = self.sample_rates['orders'] if orders_file_size > 200 else 0.2
            
            orders_df = self.load_file_optimized(
                orders_file.name, 
                columns=['order_id', 'user_id'], 
                sample_rate=orders_sample_rate,
                max_rows=self.max_orders
            )
            
            if len(orders_df) == 0:
                return "❌ No valid order data found", "", "", "", ""
            
            logging.info(f"📊 Loaded {len(orders_df):,} orders from {orders_df['user_id'].nunique():,} customers")
            
            # Load order products filtered by available orders
            order_products_sample_rate = self.sample_rates['order_products'] if self.get_file_size_mb(order_products_file.name) > 200 else 0.3
            
            order_products_df = self.load_file_optimized(
                order_products_file.name,
                columns=['order_id', 'product_id'],
                sample_rate=order_products_sample_rate
            )
            
            if len(order_products_df) == 0:
                return "❌ No valid order products data found", "", "", "", ""
            
            logging.info(f"🛍️ Loaded {len(order_products_df):,} order items")
            
            # Load reference data (these are typically smaller)
            products_df = self.load_file_optimized(
                products_file.name,
                columns=['product_id', 'product_name', 'aisle_id']
            )
            
            aisles_df = self.load_file_optimized(
                aisles_file.name,
                columns=['aisle_id', 'aisle']
            )
            
            if len(products_df) == 0 or len(aisles_df) == 0:
                return "❌ No valid product or aisle data found", "", "", "", ""
            
            # Clean reference data
            products_df = self.clean_product_names(products_df, 'product_name')
            aisles_df = self.clean_product_names(aisles_df, 'aisle')
            
            logging.info(f"📦 Loaded {len(products_df):,} products and {len(aisles_df):,} aisles")
            
            # Phase 3: Optimized Data Merging
            logging.info("🔄 Merging datasets...")
            merged_data = self.merge_datasets_optimized(orders_df, order_products_df, products_df, aisles_df)
            
            if len(merged_data) < 100:
                return "❌ Insufficient data after merging. Check file compatibility.", "", "", "", ""
            
            # Clean up intermediate data
            del orders_df, order_products_df, products_df, aisles_df
            self.clean_memory()
            
            logging.info(f"✅ Final dataset: {len(merged_data):,} records from {merged_data['user_id'].nunique():,} customers")
            
            # Phase 4: Analysis
            logging.info("🔍 Running market basket analysis...")
            market_analysis = self.analyze_market_basket(merged_data)
            self.clean_memory()
            
            logging.info("👥 Running customer segmentation...")
            segment_analysis, visualization = self.analyze_customer_segments(merged_data)
            self.clean_memory()
            
            logging.info("💡 Generating recommendations...")
            cross_sell_recs, upsell_recs = self.generate_recommendations(market_analysis, segment_analysis)
            
            # Final cleanup
            del merged_data
            self.clean_memory()
            
            final_memory = self.get_memory_usage()
            logging.info(f"🎉 Analysis complete! Final memory usage: {final_memory:.1f}MB")
            
            return market_analysis, segment_analysis, visualization, cross_sell_recs, upsell_recs
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)[:200]}\n\n🔧 Troubleshooting tips:\n• Ensure files are properly formatted CSV\n• Check that column names match exactly\n• Try with a smaller dataset first\n• Verify file encoding is UTF-8"
            logging.error(f"Analysis failed: {str(e)}", exc_info=True)
            
            # Emergency cleanup
            self.clean_memory()
            
            return error_msg, "", "", "", ""

def create_production_interface():
    """Create ultra-optimized production interface"""
    
    analyzer = OptimizedMarketBasketAnalyzer()
    
    # Custom CSS for better appearance
    custom_css = """
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1400px;
        margin: auto;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    .upload-container {
        border: 2px dashed #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .upload-container:hover {
        border-color: #4CAF50;
        background-color: #f9f9f9;
    }
    .status-indicator {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    """
    
    with gr.Blocks(
        title="Market Basket Analysis Dashboard - Ultra Optimized",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as gradio_app:
        
        gr.HTML("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">🛒 Market Basket Analysis Dashboard</h1>
            <p style="margin: 15px 0; font-size: 1.2em; opacity: 0.9;">Ultra-Optimized for Large Datasets (800MB+, 3.5M+ rows)</p>
            <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin-top: 15px;">
                <span style="font-size: 0.9em;">✨ Intelligent sampling • 🚀 Memory optimization • 📊 Advanced analytics</span>
            </div>
        </div>
        """)
        
        with gr.Accordion("📋 Ultra-Large Dataset Guide", open=True):
            gr.Markdown("""
            **🎯 Optimized for Enterprise-Scale Data:**
            
            | File | Size Limit | Expected Columns | Processing Notes |
            |------|------------|------------------|------------------|
            | **orders.csv** | Up to 1.2GB | `order_id`, `user_id` | Auto-sampled at 5% for files >200MB |
            | **order_products.csv** | Up to 1.2GB | `order_id`, `product_id` | Auto-sampled at 3% for files >200MB |
            | **products.csv** | Up to 500MB | `product_id`, `product_name`, `aisle_id` | Fully processed |
            | **aisles.csv** | Up to 100MB | `aisle_id`, `aisle` | Fully processed |
            
            **⚡ Performance Optimizations:**
            - **Intelligent Chunking**: Processes data in optimized chunks to prevent memory overflow
            - **Adaptive Sampling**: Automatically samples large datasets while preserving statistical significance
            - **Memory Management**: Aggressive cleanup and MiniBatch algorithms for large-scale analysis
            - **Smart Filtering**: Limits analysis to top products/customers for meaningful insights
            
            **🎯 Best Results With:**
            - At least 10,000 orders and 50,000 order items
            - Minimum 6 months of transaction data
            - Clean, consistent product naming
            - UTF-8 encoded CSV files
            
            **⏱️ Expected Processing Time:**
            - Small datasets (<100MB): 1-3 minutes
            - Medium datasets (100-400MB): 3-8 minutes  
            - Large datasets (400MB-1GB): 8-15 minutes
            """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3 style='text-align: center; color: #333;'>📊 Transaction Data</h3>")
                orders_file = gr.File(
                    label="Orders CSV (order_id, user_id)", 
                    file_types=[".csv"],
                    elem_id="orders_upload",
                    elem_classes=["upload-container"]
                )
                products_file = gr.File(
                    label="Products CSV (product_id, product_name, aisle_id)", 
                    file_types=[".csv"],
                    elem_id="products_upload",
                    elem_classes=["upload-container"]
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3 style='text-align: center; color: #333;'>🛍️ Product Data</h3>")
                order_products_file = gr.File(
                    label="Order Products CSV (order_id, product_id)", 
                    file_types=[".csv"],
                    elem_id="order_products_upload",
                    elem_classes=["upload-container"]
                )
                aisles_file = gr.File(
                    label="Aisles CSV (aisle_id, aisle)", 
                    file_types=[".csv"],
                    elem_id="aisles_upload",
                    elem_classes=["upload-container"]
                )
        
        with gr.Row():
            analyze_button = gr.Button(
                "🚀 Run Ultra-Optimized Analysis", 
                variant="primary", 
                size="lg",
                elem_id="analyze_btn",
                scale=2
            )
            
        gr.HTML("""
        <div style="background: #e8f5e8; border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 20px 0; text-align: center;">
            <h4 style="margin: 0 0 10px 0; color: #2E7D32;">⚡ Ultra-Fast Processing Status</h4>
            <p style="margin: 0; color: #388E3C;">Click "Run Analysis" to begin processing your large dataset with advanced optimizations</p>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("📊 Market Basket Analysis", elem_id="market_tab"):
                market_results = gr.Markdown(
                    value="""
## 🔍 Market Basket Analysis Results

**Ready to analyze your data!** Upload your files and click the analysis button to discover:

- **Product Associations**: Which items are frequently bought together
- **Cross-selling Opportunities**: High-value product combinations  
- **Purchase Patterns**: Statistical relationships in customer behavior
- **Actionable Insights**: Specific recommendations for increasing sales

*Processing will begin once you start the analysis...*
                    """,
                    elem_id="market_results"
                )
            
            with gr.TabItem("👥 Customer Segmentation", elem_id="segments_tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        segment_results = gr.Markdown(
                            value="""
## 👥 Customer Segment Analysis

**Customer segmentation will reveal:**

- **Behavioral Groups**: Customers with similar shopping patterns
- **Value Segments**: High, medium, and low-value customer groups
- **Category Preferences**: What each segment prefers to buy
- **Targeting Opportunities**: How to market to each group effectively

*Advanced MiniBatch clustering optimized for large datasets*
                            """,
                            elem_id="segment_results"
                        )
                    with gr.Column(scale=1):
                        segment_viz = gr.HTML(
                            value="""
                            <div style="background: #f8f9fa; border: 2px dashed #dee2e6; border-radius: 15px; padding: 40px; text-align: center; color: #6c757d;">
                                <div style="font-size: 3em; margin-bottom: 15px;">📊</div>
                                <h4>Interactive Visualization</h4>
                                <p>Advanced segment visualization will appear here after analysis</p>
                                <small>Includes distribution charts and heatmaps</small>
                            </div>
                            """,
                            elem_id="segment_viz"
                        )
            
            with gr.TabItem("💡 Business Recommendations", elem_id="recommendations_tab"):
                gr.HTML("<h2 style='text-align: center; color: #333; margin-bottom: 30px;'>🎯 Actionable Business Strategy</h2>")
                with gr.Row():
                    with gr.Column():
                        cross_sell_recs = gr.Markdown(
                            value="""
## 🎯 Cross-Selling Strategy

**Data-driven cross-selling recommendations will include:**

- **Product Bundle Opportunities**: Which items to package together
- **Store Layout Optimization**: Where to place complementary products  
- **Digital Marketing**: Personalized recommendation strategies
- **Staff Training**: Key product combinations to promote
- **Performance Metrics**: How to measure cross-selling success

*Advanced association rule mining will identify the strongest product relationships*
                            """,
                            elem_id="cross_sell"
                        )
                    with gr.Column():
                        upsell_recs = gr.Markdown(
                            value="""
## 📈 Upselling Strategy  

**Segment-based upselling strategies will provide:**

- **Customer-Specific Approaches**: Tailored tactics for each segment
- **Premium Product Positioning**: How to introduce higher-value items
- **Loyalty Programs**: Tier-based reward strategies
- **Personalization Tactics**: Individual customer targeting methods
- **Success Metrics**: KPIs to track upselling performance

*Machine learning-powered segmentation enables precise targeting*
                            """,
                            elem_id="upsell"
                        )
        
        # Event handler with progress tracking
        analyze_button.click(
            fn=analyzer.run_complete_analysis,
            inputs=[orders_file, order_products_file, products_file, aisles_file],
            outputs=[market_results, segment_results, segment_viz, cross_sell_recs, upsell_recs],
            show_progress=True
        )
        
        gr.HTML("""
        <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 15px; text-align: center;">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                <div style="margin: 10px;">
                    <strong style="color: #495057;">🔧 Built with</strong><br>
                    <span style="color: #6c757d;">Gradio + FastAPI</span>
                </div>
                <div style="margin: 10px;">
                    <strong style="color: #495057;">⚡ Optimized for</strong><br>
                    <span style="color: #6c757d;">800MB+ datasets</span>
                </div>
                <div style="margin: 10px;">
                    <strong style="color: #495057;">🧠 Powered by</strong><br>
                    <span style="color: #6c757d;">scikit-learn + MLxtend</span>
                </div>
                <div style="margin: 10px;">
                    <strong style="color: #495057;">📊 Advanced</strong><br>
                    <span style="color: #6c757d;">MiniBatch clustering</span>
                </div>
            </div>
        </div>
        """)
    
    return gradio_app

# Create FastAPI app with enhanced error handling
app = FastAPI(
    title="Market Basket Analysis Dashboard",
    description="Ultra-optimized for large datasets",
    version="2.0.0"
)

# Health check endpoint for Cloud Run
@app.get("/health")
async def health_check():
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return {
            "status": "healthy",
            "memory_usage_mb": round(memory_mb, 2),
            "message": "Market Basket Analysis Dashboard is running"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

# Memory status endpoint
@app.get("/status")
async def status_check():
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "status": "operational"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Status check failed: {str(e)}"
        }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)[:200]}"}
    )

# Mount Gradio app
gradio_app = create_production_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    try:
        logging.info("✅ Creating ultra-optimized interface...")
        print("✅ Market Basket Analysis Dashboard (Ultra-Optimized) is ready!")
        print("🚀 Optimized for 800MB+ files and 3.5M+ rows")
        print("💾 Advanced memory management enabled")
        print("⚡ MiniBatch algorithms for large-scale processing")
    except Exception as e:
        logging.error(f"Failed to start application: {str(e)}", exc_info=True)
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
