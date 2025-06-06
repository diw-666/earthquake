import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EarthquakeVisualizer:
    def __init__(self):
        self.color_map = {
            'Very High': '#FF0000',  # Red
            'High': '#FF6600',       # Orange-Red
            'Medium': '#FFAA00',     # Orange
            'Low': '#FFDD00'         # Yellow
        }
        # Base date for day conversion (assuming days since 1900-01-01)
        self.base_date = datetime(1900, 1, 1)
        
    def days_to_date(self, days):
        """Convert days since base date to actual calendar date"""
        try:
            actual_date = self.base_date + timedelta(days=float(days))
            return actual_date.strftime('%Y-%m-%d')
        except:
            return f"Day {days:.1f}"
            
    def days_to_year(self, days):
        """Convert days since base date to year"""
        try:
            actual_date = self.base_date + timedelta(days=float(days))
            return actual_date.year
        except:
            return int(days / 365.25) + 1900
        
    def create_interactive_map(self, historical_data, predictions=None, save_path='earthquake_map.html'):
        """Create an interactive map showing historical earthquakes and predictions"""
        print("Creating interactive earthquake map...")
        
        # Calculate center point for South Asia
        center_lat = historical_data['y'].median()
        center_lon = historical_data['x'].median()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers with proper attribution
        folium.TileLayer(
            tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
        
        folium.TileLayer(
            tiles='CartoDB positron',
            name='Light Theme'
        ).add_to(m)
        
        # Historical earthquakes layer
        historical_group = folium.FeatureGroup(name='Historical Earthquakes')
        
        # Sample historical data to avoid overcrowding
        sample_size = min(5000, len(historical_data))
        historical_sample = historical_data.sample(n=sample_size, random_state=42)
        
        for _, row in historical_sample.iterrows():
            color = self._get_magnitude_color(row['mag'])
            date_str = self.days_to_date(row['days'])
            folium.CircleMarker(
                location=[row['y'], row['x']],
                radius=max(2, row['mag']),
                popup=f"Magnitude: {row['mag']:.2f}<br>Depth: {row['depth']:.1f} km<br>Date: {date_str}",
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                weight=1
            ).add_to(historical_group)
        
        historical_group.add_to(m)
        
        # Predictions layer if provided
        if predictions is not None and len(predictions) > 0:
            predictions_group = folium.FeatureGroup(name='Predicted Earthquakes')
            
            for _, row in predictions.iterrows():
                color = self.color_map.get(row['risk_level'], '#FFDD00')
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=max(4, row['predicted_magnitude'] * 2),
                    popup=f"Predicted Magnitude: {row['predicted_magnitude']:.2f}<br>"
                          f"Risk Level: {row['risk_level']}<br>"
                          f"Date: {row['predicted_date']}",
                    color=color,
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(predictions_group)
            
            predictions_group.add_to(m)
        
        # Add heatmap layer for earthquake density
        heat_data = [[row['y'], row['x'], row['mag']] for _, row in historical_sample.iterrows()]
        heatmap = plugins.HeatMap(
            heat_data,
            name='Earthquake Density',
            radius=15,
            blur=10,
            max_zoom=1
        )
        heatmap.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Levels</b></p>
        <p><i class="fa fa-circle" style="color:#FF0000"></i> Very High (7.0+)</p>
        <p><i class="fa fa-circle" style="color:#FF6600"></i> High (6.0-6.9)</p>
        <p><i class="fa fa-circle" style="color:#FFAA00"></i> Medium (5.0-5.9)</p>
        <p><i class="fa fa-circle" style="color:#FFDD00"></i> Low (4.0-4.9)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(save_path)
        print(f"Interactive map saved as {save_path}")
        
        return m
    
    def _get_magnitude_color(self, magnitude):
        """Get color based on earthquake magnitude"""
        if magnitude >= 7.0:
            return '#FF0000'  # Red
        elif magnitude >= 6.0:
            return '#FF6600'  # Orange-Red
        elif magnitude >= 5.0:
            return '#FFAA00'  # Orange
        elif magnitude >= 4.0:
            return '#FFDD00'  # Yellow
        else:
            return '#00FF00'  # Green
    
    def create_prediction_charts(self, predictions, historical_data):
        """Create various charts for earthquake predictions"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Predicted Magnitude Distribution',
                'Risk Level Distribution',
                'Geographic Distribution of Predictions',
                'Historical vs Predicted Magnitudes'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Magnitude distribution
        fig.add_trace(
            go.Histogram(
                x=predictions['predicted_magnitude'],
                name='Predicted Magnitude',
                nbinsx=20,
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Risk level distribution
        risk_counts = predictions['risk_level'].value_counts()
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                name='Risk Levels',
                marker_color=['red' if x == 'Very High' else 
                            'orange' if x == 'High' else 
                            'yellow' if x == 'Medium' else 'green' 
                            for x in risk_counts.index]
            ),
            row=1, col=2
        )
        
        # 3. Geographic scatter
        fig.add_trace(
            go.Scatter(
                x=predictions['longitude'],
                y=predictions['latitude'],
                mode='markers',
                marker=dict(
                    size=predictions['predicted_magnitude'] * 3,
                    color=predictions['predicted_magnitude'],
                    colorscale='Reds',
                    showscale=True
                ),
                name='Predicted Locations'
            ),
            row=2, col=1
        )
        
        # 4. Historical vs predicted comparison
        historical_sample = historical_data.sample(n=min(1000, len(historical_data)), random_state=42)
        fig.add_trace(
            go.Histogram(
                x=historical_sample['mag'],
                name='Historical',
                opacity=0.7,
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=predictions['predicted_magnitude'],
                name='Predicted',
                opacity=0.7,
                marker_color='red'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Earthquake Prediction Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_risk_heatmap(self, predictions):
        """Create a risk heatmap showing geographic risk distribution"""
        if len(predictions) == 0:
            return None
        
        # Create grid for heatmap
        lat_bins = np.linspace(predictions['latitude'].min(), predictions['latitude'].max(), 20)
        lon_bins = np.linspace(predictions['longitude'].min(), predictions['longitude'].max(), 20)
        
        # Calculate average risk in each grid cell
        risk_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
        
        for i in range(len(lat_bins)-1):
            for j in range(len(lon_bins)-1):
                mask = (
                    (predictions['latitude'] >= lat_bins[i]) & 
                    (predictions['latitude'] < lat_bins[i+1]) &
                    (predictions['longitude'] >= lon_bins[j]) & 
                    (predictions['longitude'] < lon_bins[j+1])
                )
                
                if mask.sum() > 0:
                    risk_grid[i, j] = predictions[mask]['predicted_magnitude'].mean()
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_grid,
            x=lon_bins[:-1],
            y=lat_bins[:-1],
            colorscale='Reds',
            colorbar=dict(title="Average Predicted Magnitude")
        ))
        
        fig.update_layout(
            title="Geographic Risk Distribution Heatmap",
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )
        
        return fig
    
    def create_time_series_analysis(self, historical_data):
        """Create time series analysis of historical earthquakes"""
        # Convert days to actual dates
        historical_data = historical_data.copy()
        historical_data['year'] = historical_data['days'].apply(self.days_to_year)
        
        # Filter to reasonable date range (remove any outlier years)
        historical_data = historical_data[
            (historical_data['year'] >= 1900) & (historical_data['year'] <= 2030)
        ]
        
        # Group by year and calculate statistics
        yearly_stats = historical_data.groupby('year').agg({
            'mag': ['count', 'mean', 'max'],
            'depth': 'mean'
        }).round(2)
        
        yearly_stats.columns = ['count', 'avg_magnitude', 'max_magnitude', 'avg_depth']
        yearly_stats = yearly_stats.reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Earthquake Count per Year',
                'Average Magnitude per Year',
                'Maximum Magnitude per Year',
                'Average Depth per Year'
            )
        )
        
        # Earthquake count
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['count'],
                mode='lines+markers',
                name='Count'
            ),
            row=1, col=1
        )
        
        # Average magnitude
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['avg_magnitude'],
                mode='lines+markers',
                name='Avg Magnitude',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        # Maximum magnitude
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['max_magnitude'],
                mode='lines+markers',
                name='Max Magnitude',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Average depth
        fig.add_trace(
            go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['avg_depth'],
                mode='lines+markers',
                name='Avg Depth',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Historical Earthquake Trends Analysis",
            showlegend=False
        )
        
        # Update x-axis labels to show years properly
        fig.update_xaxes(title_text="Year")
        
        return fig
    
    def create_magnitude_depth_analysis(self, historical_data, predictions=None):
        """Analyze relationship between magnitude and depth"""
        fig = go.Figure()
        
        # Historical data scatter
        sample_size = min(5000, len(historical_data))
        historical_sample = historical_data.sample(n=sample_size, random_state=42)
        
        fig.add_trace(go.Scatter(
            x=historical_sample['depth'],
            y=historical_sample['mag'],
            mode='markers',
            name='Historical',
            opacity=0.6,
            marker=dict(color='blue', size=4)
        ))
        
        # Predictions if available
        if predictions is not None and len(predictions) > 0:
            # Use average depth from historical data for predictions
            avg_depth = historical_data['depth'].mean()
            pred_depths = [avg_depth] * len(predictions)
            
            fig.add_trace(go.Scatter(
                x=pred_depths,
                y=predictions['predicted_magnitude'],
                mode='markers',
                name='Predictions',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='diamond'
                )
            ))
        
        fig.update_layout(
            title="Earthquake Magnitude vs Depth Analysis",
            xaxis_title="Depth (km)",
            yaxis_title="Magnitude",
            hovermode='closest'
        )
        
        return fig
    
    def save_all_visualizations(self, historical_data, predictions=None):
        """Save all visualizations to files"""
        print("Creating and saving all visualizations...")
        
        # Interactive map
        self.create_interactive_map(historical_data, predictions)
        
        # Prediction charts
        if predictions is not None and len(predictions) > 0:
            pred_fig = self.create_prediction_charts(predictions, historical_data)
            pred_fig.write_html("prediction_dashboard.html")
            
            # Risk heatmap
            risk_fig = self.create_risk_heatmap(predictions)
            if risk_fig:
                risk_fig.write_html("risk_heatmap.html")
        
        # Time series analysis
        time_fig = self.create_time_series_analysis(historical_data)
        time_fig.write_html("time_series_analysis.html")
        
        # Magnitude-depth analysis
        mag_depth_fig = self.create_magnitude_depth_analysis(historical_data, predictions)
        mag_depth_fig.write_html("magnitude_depth_analysis.html")
        
        print("All visualizations saved successfully!")

def main():
    """Test the visualizer with sample data"""
    visualizer = EarthquakeVisualizer()
    
    # Load sample data
    df = pd.read_csv('data/Earthquakes_South_Asia_Cleaned.csv')
    
    # Create sample predictions
    sample_predictions = pd.DataFrame({
        'predicted_magnitude': np.random.uniform(4.0, 7.0, 50),
        'latitude': np.random.uniform(df['y'].min(), df['y'].max(), 50),
        'longitude': np.random.uniform(df['x'].min(), df['x'].max(), 50),
        'predicted_day': np.random.uniform(df['days'].max(), df['days'].max() + 30, 50),
        'predicted_date': [f"Day {i}" for i in range(50)],
        'risk_level': np.random.choice(['Low', 'Medium', 'High', 'Very High'], 50)
    })
    
    # Create all visualizations
    visualizer.save_all_visualizations(df, sample_predictions)

if __name__ == "__main__":
    main() 