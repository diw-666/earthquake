import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from earthquake_predictor import EarthquakePredictor
from earthquake_visualizer import EarthquakeVisualizer
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Earthquake Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Date conversion utility
def days_to_date(days):
    """Convert days since 1900-01-01 to actual calendar date"""
    try:
        base_date = datetime(1900, 1, 1)
        actual_date = base_date + timedelta(days=float(days))
        return actual_date.strftime('%Y-%m-%d')
    except:
        return f"Day {days:.1f}"

@st.cache_data
def load_earthquake_data():
    """Load earthquake data with caching"""
    try:
        df = pd.read_csv('data/Earthquakes_South_Asia_Cleaned.csv')
        return df
    except FileNotFoundError:
        st.error("Earthquake data file not found. Please ensure 'data/Earthquakes_South_Asia_Cleaned.csv' exists.")
        return None

@st.cache_resource
def initialize_predictor():
    """Initialize the earthquake predictor"""
    return EarthquakePredictor()

@st.cache_resource
def initialize_visualizer():
    """Initialize the earthquake visualizer"""
    return EarthquakeVisualizer()

def train_models_interface(predictor, df):
    """Interface for training models"""
    st.subheader("🤖 Model Training")
    
    if st.button("Train Prediction Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            progress_bar = st.progress(0)
            
            # Feature engineering
            progress_bar.progress(20)
            df_featured = predictor.engineer_features(df)
            
            # Prepare features
            progress_bar.progress(40)
            X, y = predictor.prepare_features(df_featured)
            
            # Train models
            progress_bar.progress(60)
            X_test, y_test = predictor.train_models(X, y)
            
            # Save models
            progress_bar.progress(80)
            predictor.save_models()
            
            progress_bar.progress(100)
            
            st.success("Models trained successfully!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance:**")
                for model_name, metrics in predictor.evaluation_results.items():
                    st.write(f"- {model_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
            
            with col2:
                st.write("**Dataset Info:**")
                st.write(f"- Total samples: {len(df):,}")
                st.write(f"- Training samples: {len(X) - len(X_test):,}")
                st.write(f"- Test samples: {len(X_test):,}")
                st.write(f"- Features: {len(predictor.feature_names)}")
            
            return df_featured
    
    return None

def prediction_interface(predictor, df_featured):
    """Interface for generating predictions"""
    st.subheader("🔮 Earthquake Predictions")
    
    # Check if models are trained
    if not predictor.is_trained:
        if not predictor.load_models():
            st.warning("No trained models found. Please train models first.")
            return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_ahead = st.slider("Prediction horizon (days)", 1, 90, 30)
    
    with col2:
        min_magnitude = st.slider("Minimum magnitude threshold", 3.0, 6.0, 4.0, 0.1)
    
    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions = predictor.predict_next_earthquakes(df_featured, days_ahead=days_ahead)
            
            # Filter by minimum magnitude
            predictions = predictions[predictions['predicted_magnitude'] >= min_magnitude]
            
            if len(predictions) > 0:
                st.success(f"Generated {len(predictions)} earthquake predictions!")
                
                # Save predictions
                predictions.to_csv('earthquake_predictions.csv', index=False)
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                
                with col2:
                    avg_mag = predictions['predicted_magnitude'].mean()
                    st.metric("Average Magnitude", f"{avg_mag:.2f}")
                
                with col3:
                    high_risk = len(predictions[predictions['risk_level'].isin(['High', 'Very High'])])
                    st.metric("High Risk Events", high_risk)
                
                with col4:
                    max_mag = predictions['predicted_magnitude'].max()
                    st.metric("Maximum Magnitude", f"{max_mag:.2f}")
                
                return predictions
            else:
                st.info(f"No earthquakes predicted above magnitude {min_magnitude} in the next {days_ahead} days.")
                return None
    
    return None

def display_interactive_map(historical_data, predictions, visualizer):
    """Display interactive map"""
    st.subheader("🗺️ Interactive Earthquake Map")
    
    # Create the map
    m = visualizer.create_interactive_map(historical_data, predictions, save_path='temp_map.html')
    
    # Display map in Streamlit
    with open('temp_map.html', 'r') as f:
        map_html = f.read()
    
    st.components.v1.html(map_html, height=600)
    
    # Cleanup
    if os.path.exists('temp_map.html'):
        os.remove('temp_map.html')

def display_analytics(historical_data, predictions, visualizer):
    """Display analytics dashboard"""
    st.subheader("📊 Analytics Dashboard")
    
    # Historical data analysis
    st.write("### Historical Earthquake Trends")
    time_fig = visualizer.create_time_series_analysis(historical_data)
    st.plotly_chart(time_fig, use_container_width=True)
    
    # Magnitude vs Depth analysis
    st.write("### Magnitude vs Depth Analysis")
    mag_depth_fig = visualizer.create_magnitude_depth_analysis(historical_data, predictions)
    st.plotly_chart(mag_depth_fig, use_container_width=True)
    
    if predictions is not None and len(predictions) > 0:
        # Prediction analysis
        st.write("### Prediction Analysis")
        pred_fig = visualizer.create_prediction_charts(predictions, historical_data)
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # Risk heatmap
        st.write("### Geographic Risk Distribution")
        risk_fig = visualizer.create_risk_heatmap(predictions)
        if risk_fig:
            st.plotly_chart(risk_fig, use_container_width=True)

def display_data_explorer(df):
    """Data exploration interface"""
    st.subheader("🔍 Data Explorer")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Overview:**")
        st.write(f"- Total earthquakes: {len(df):,}")
        st.write(f"- Date range: {days_to_date(df['days'].min())} to {days_to_date(df['days'].max())}")
        st.write(f"- Magnitude range: {df['mag'].min():.1f} to {df['mag'].max():.1f}")
        st.write(f"- Depth range: {df['depth'].min():.1f} to {df['depth'].max():.1f} km")
    
    with col2:
        st.write("**Magnitude Distribution:**")
        fig = px.histogram(df, x='mag', nbins=50, title="Earthquake Magnitude Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Filters
    st.write("### Data Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mag_range = st.slider("Magnitude Range", 
                            float(df['mag'].min()), 
                            float(df['mag'].max()), 
                            (4.0, 8.0))
    
    with col2:
        depth_range = st.slider("Depth Range (km)", 
                              float(df['depth'].min()), 
                              float(df['depth'].max()), 
                              (0.0, 100.0))
    
    with col3:
        sample_size = st.number_input("Sample Size", 1, len(df), min(1000, len(df)))
    
    # Filter data
    filtered_data = df[
        (df['mag'] >= mag_range[0]) & (df['mag'] <= mag_range[1]) &
        (df['depth'] >= depth_range[0]) & (df['depth'] <= depth_range[1])
    ]
    
    # Ensure we don't sample more than available data
    actual_sample_size = min(sample_size, len(filtered_data))
    
    if actual_sample_size > 0:
        filtered_df = filtered_data.sample(n=actual_sample_size, random_state=42).copy()
    else:
        filtered_df = filtered_data.copy()
    
    # Add proper date column for display
    filtered_df['date'] = filtered_df['days'].apply(days_to_date)
    
    # Reorder columns to show date first
    display_columns = ['date', 'mag', 'depth', 'x', 'y', 'days']
    filtered_df_display = filtered_df[display_columns]
    
    # Display filtered data
    st.write(f"**Filtered Data ({len(filtered_df):,} records):**")
    if len(filtered_df) > 0:
        st.dataframe(filtered_df_display.head(100), use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")
    
    return filtered_df

def display_predictions_table(predictions):
    """Display predictions in a table format"""
    if predictions is not None and len(predictions) > 0:
        st.subheader("📋 Detailed Predictions")
        
        # Sort by magnitude (highest first)
        predictions_sorted = predictions.sort_values('predicted_magnitude', ascending=False).copy()
        
        # Ensure proper date display - create or fix the predicted_date column
        if 'predicted_date' not in predictions_sorted.columns or predictions_sorted['predicted_date'].isna().any():
            predictions_sorted['predicted_date'] = predictions_sorted['predicted_day'].apply(days_to_date)
        
        # Select and reorder columns for display
        display_columns = [
            'predicted_date', 'predicted_magnitude', 'latitude', 'longitude', 
            'risk_level', 'region_x', 'region_y'
        ]
        predictions_display = predictions_sorted[display_columns].copy()
        
        # Color code based on risk level
        def color_risk(val):
            if val == 'Very High':
                return 'background-color: #ffcdd2'
            elif val == 'High':
                return 'background-color: #ffe0b2'
            elif val == 'Medium':
                return 'background-color: #fff9c4'
            else:
                return 'background-color: #c8e6c9'
        
        # Display styled dataframe
        styled_df = predictions_display.style.applymap(
            color_risk, subset=['risk_level']
        ).format({
            'predicted_magnitude': '{:.2f}',
            'latitude': '{:.4f}',
            'longitude': '{:.4f}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download button
        csv = predictions_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"earthquake_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">🌍 Earthquake Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Advanced machine learning system for earthquake prediction in South Asia")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🤖 Model Training", "🔮 Predictions", "🗺️ Interactive Map", "📊 Analytics", "🔍 Data Explorer"]
    )
    
    # Load data
    df = load_earthquake_data()
    if df is None:
        return
    
    # Initialize components
    predictor = initialize_predictor()
    visualizer = initialize_visualizer()
    
    # Load existing predictions if available
    predictions = None
    if os.path.exists('earthquake_predictions.csv'):
        try:
            predictions = pd.read_csv('earthquake_predictions.csv')
        except:
            pass
    
    # Page routing
    if page == "🏠 Overview":
        st.subheader("System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Earthquakes", f"{len(df):,}")
        
        with col2:
            recent_count = len(df[df['days'] > df['days'].quantile(0.9)])
            st.metric("Recent Events (Top 10%)", recent_count)
        
        with col3:
            major_count = len(df[df['mag'] >= 6.0])
            st.metric("Major Earthquakes (6.0+)", major_count)
        
        # System status
        st.subheader("System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_status = "✅ Ready" if predictor.load_models() else "❌ Not Trained"
            st.write(f"**Model Status:** {model_status}")
        
        with col2:
            pred_status = "✅ Available" if predictions is not None else "❌ No Predictions"
            st.write(f"**Predictions:** {pred_status}")
        
        # Quick insights
        st.subheader("Quick Insights")
        
        recent_data = df[df['days'] > df['days'].quantile(0.95)]
        avg_recent_mag = recent_data['mag'].mean()
        
        st.write(f"- Average magnitude in recent events: **{avg_recent_mag:.2f}**")
        st.write(f"- Most active region: **{recent_data.groupby(['x', 'y']).size().idxmax()}**")
        st.write(f"- Deepest recent earthquake: **{recent_data['depth'].max():.1f} km**")
        
    elif page == "🤖 Model Training":
        df_featured = train_models_interface(predictor, df)
        
        if df_featured is not None:
            st.subheader("Feature Engineering Results")
            st.write("New features created for better prediction accuracy:")
            new_features = [col for col in df_featured.columns if col not in df.columns]
            for feature in new_features:
                st.write(f"- {feature}")
    
    elif page == "🔮 Predictions":
        # Load featured data if models exist
        df_featured = df
        if predictor.load_models() or predictor.is_trained:
            with st.spinner("Preparing features..."):
                df_featured = predictor.engineer_features(df)
        
        new_predictions = prediction_interface(predictor, df_featured)
        
        if new_predictions is not None:
            predictions = new_predictions
        
        display_predictions_table(predictions)
    
    elif page == "🗺️ Interactive Map":
        display_interactive_map(df, predictions, visualizer)
    
    elif page == "📊 Analytics":
        display_analytics(df, predictions, visualizer)
    
    elif page == "🔍 Data Explorer":
        filtered_df = display_data_explorer(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.write("**Earthquake Prediction System**")
    st.sidebar.write("Powered by Machine Learning")
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 