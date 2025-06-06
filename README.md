# 🌍 Earthquake Prediction System

Advanced machine learning system for predicting earthquakes in South Asia using historical seismic data. This comprehensive solution includes predictive modeling, interactive visualizations, and a user-friendly dashboard.

## 🚀 Features

### Machine Learning Models
- **Gradient Boosting Model**: Advanced gradient boosting regression for earthquake prediction
- **Feature Engineering**: Advanced temporal and spatial feature extraction
- **Time Series Analysis**: Historical pattern recognition and trend analysis
- **Risk Assessment**: Automated risk level classification (Low, Medium, High, Very High)

### Interactive Visualizations
- **Interactive Maps**: Folium-based maps with earthquake locations and predictions
- **Risk Heatmaps**: Geographic distribution of seismic risk
- **Time Series Charts**: Historical trends and patterns
- **Statistical Dashboards**: Comprehensive data analysis and insights

### Dashboard Features
- **Real-time Predictions**: Generate earthquake forecasts for the next 1-90 days
- **Model Training**: Train and evaluate prediction models with real-time progress
- **Data Explorer**: Interactive data filtering and exploration
- **Map Integration**: Seamless integration of maps within the dashboard
- **Export Capabilities**: Download predictions and visualizations

## 📁 Project Structure

```
earthquake/
├── data/
│   ├── Earthquakes_South_Asia_Cleaned.csv    # Processed earthquake data
│   └── column_descriptions.md                 # Data dictionary
├── models/                                    # Trained model storage
├── earthquake_predictor.py                    # Core prediction algorithms
├── earthquake_visualizer.py                   # Visualization components
├── dashboard.py                              # Streamlit dashboard
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd earthquake
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data availability**:
   Ensure the earthquake dataset is present at `data/Earthquakes_South_Asia_Cleaned.csv`

## 🚀 Usage

### Option 1: Interactive Dashboard (Recommended)

Launch the Streamlit dashboard for the complete interactive experience:

```bash
streamlit run dashboard.py
```

This opens a web interface at `http://localhost:8501` with the following features:
- **Overview**: System status and quick insights
- **Model Training**: Train prediction models with progress tracking
- **Predictions**: Generate and view earthquake forecasts
- **Interactive Map**: Explore historical and predicted earthquakes
- **Analytics**: Detailed statistical analysis and trends
- **Data Explorer**: Filter and examine the earthquake dataset

### Option 2: Command Line Training

Train models directly via command line:

```bash
python earthquake_predictor.py
```

### Option 3: Generate Visualizations

Create static visualizations:

```bash
python earthquake_visualizer.py
```

## 📊 Data Features

The system uses the following earthquake parameters for prediction:

### Input Features
- **Geographic**: Latitude, longitude, depth
- **Temporal**: Time-based patterns and cycles
- **Historical**: Past earthquake activity in the region
- **Magnitude**: Historical magnitude patterns
- **Spatial Clustering**: Regional earthquake groupings

### Engineered Features
- **Seismic Activity Windows**: 30, 90, and 365-day activity counts
- **Magnitude Statistics**: Rolling averages and maximums
- **Fault Line Proximity**: Distance to high-activity zones
- **Time Since Major Events**: Days since last significant earthquake
- **Regional Patterns**: Geographic clustering and trends

## 🧠 Machine Learning Model

### Model Architecture
The system uses an optimized Gradient Boosting Regressor:

**Gradient Boosting Regressor**
- Sequential error correction and optimization
- Advanced hyperparameter tuning with Grid Search
- Cross-validation for robust performance
- Feature importance analysis
- Handles non-linear relationships effectively

### Performance Metrics
Models are evaluated using:
- **RMSE (Root Mean Square Error)**: Overall prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² Score**: Explained variance ratio

## 🗺️ Visualization Features

### Interactive Maps
- **Historical Earthquakes**: Color-coded by magnitude
- **Predicted Events**: Risk-based visualization
- **Density Heatmaps**: Seismic activity concentration
- **Multiple Layers**: Toggle between different data views

### Statistical Charts
- **Time Series Analysis**: Historical trends and patterns
- **Magnitude Distribution**: Frequency analysis
- **Risk Assessment**: Geographic risk distribution
- **Model Comparison**: Performance metrics visualization

## ⚡ Quick Start Guide

1. **Install and Launch**:
   ```bash
   pip install -r requirements.txt
   streamlit run dashboard.py
   ```

2. **Train Models**:
   - Navigate to "Model Training" page
   - Click "Train Prediction Models"
   - Wait for training completion (~5-10 minutes)

3. **Generate Predictions**:
   - Go to "Predictions" page
   - Set prediction horizon (1-90 days)
   - Set minimum magnitude threshold
   - Click "Generate Predictions"

4. **Explore Results**:
   - View interactive maps
   - Analyze statistical trends
   - Export prediction data

## 📈 Model Performance

Expected performance metrics:
- **RMSE**: ~0.8-1.2 (magnitude units)
- **R² Score**: ~0.6-0.8 (explained variance)
- **Prediction Accuracy**: 70-80% for magnitude ±0.5

*Note: Earthquake prediction is inherently challenging, and results should be interpreted as risk assessments rather than definitive forecasts.*

## 🔧 Configuration

### Prediction Parameters
- **Time Horizon**: 1-90 days ahead
- **Magnitude Threshold**: 3.0-6.0+ minimum magnitude
- **Regional Focus**: South Asian tectonic regions
- **Risk Levels**: Low (4.0-4.9), Medium (5.0-5.9), High (6.0-6.9), Very High (7.0+)

### System Requirements
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB+ free space
- **Processing**: Multi-core CPU for optimal performance

## 🚨 Important Disclaimers

1. **Research Purpose**: This system is designed for research and educational purposes
2. **Prediction Limitations**: Earthquake prediction remains scientifically challenging
3. **Risk Assessment**: Results should be treated as risk indicators, not definitive predictions
4. **Professional Consultation**: For critical decisions, consult professional seismologists
5. **Data Quality**: Predictions are only as good as the input data quality

## 🔬 Technical Details

### Feature Engineering Process
1. **Temporal Features**: Convert time data to cyclical patterns
2. **Spatial Clustering**: Group earthquakes by geographic proximity
3. **Rolling Statistics**: Calculate activity metrics over multiple time windows
4. **Fault Line Analysis**: Identify high-activity seismic zones
5. **Pattern Recognition**: Extract recurring earthquake patterns

### Model Training Pipeline
1. **Data Preprocessing**: Handle missing values and outliers
2. **Feature Scaling**: Normalize numerical features
3. **Cross-Validation**: Time-series aware validation splits
4. **Hyperparameter Tuning**: Grid search optimization
5. **Ensemble Combination**: Weighted average of model predictions

## 📞 Support

For technical issues or questions:
1. Check the dashboard error messages
2. Verify data file existence and format
3. Ensure all dependencies are installed
4. Review system requirements

## 🔄 Future Enhancements

Planned improvements:
- Real-time data integration
- Deep learning models (LSTM, CNN)
- Satellite imagery analysis
- IoT sensor data integration
- Mobile application development
- API endpoints for external integration

---

**Built with**: Python, Scikit-learn, XGBoost, Plotly, Folium, Streamlit

**License**: Educational and Research Use 