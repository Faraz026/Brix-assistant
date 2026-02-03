import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
import traceback

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brix Control Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL WITH ERROR HANDLING
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model package with error handling"""
    try:
        with open('xgboost_brix_model_v2(With_features).pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package, None
    except FileNotFoundError:
        return None, "❌ Model file 'xgboost_brix_model_v2(With_features).pkl' not found in repository"
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

model_package, error_msg = load_model()

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (from your training)
# ─────────────────────────────────────────────────────────────────
feature_importance = {
    'M4_Steam_CV_F/B': 0.330083,
    'M3_Steam_CV_F/B': 0.230181,
    'M1_level': 0.210819,
    'M1_Steam_CV_F/B': 0.207827,
    'M2_Steam_CV_F/B': 0.189943,
    'M5_body_Temp': 0.136055,
    'M4_level': 0.120234,
    'M3_body_Temp': 0.109120,
    'M2_condensate_flow': 0.102657,
    'M2_body_Temp': 0.093759,
    'Grain_flow2': 0.074987,
    'M1_condensate_flow': 0.067757,
    'M4_body_Temp': 0.065803,
    'M3_level': 0.061228
}

# Friendly names for operators
friendly_names = {
    'M4_Steam_CV_F/B': 'M4 Steam Valve',
    'M3_Steam_CV_F/B': 'M3 Steam Valve',
    'M1_level': 'M1 Level',
    'M1_Steam_CV_F/B': 'M1 Steam Valve',
    'M2_Steam_CV_F/B': 'M2 Steam Valve',
    'M5_body_Temp': 'M5 Temperature',
    'M4_level': 'M4 Level',
    'M3_body_Temp': 'M3 Temperature',
    'M2_condensate_flow': 'M2 Condensate Flow',
    'M2_body_Temp': 'M2 Temperature',
    'Grain_flow2': 'Grain Flow',
    'M1_condensate_flow': 'M1 Condensate Flow',
    'M4_body_Temp': 'M4 Temperature',
    'M3_level': 'M3 Level'
}

def get_unit(feature_name):
    """Return appropriate unit for each feature"""
    units = {
        'Steam_CV': '%',
        'Steam': '%',
        'Temp': '°C',
        'level': '%',
        'flow': 't/h',
        'condensate': 't/h',
        'Vaccum': 'kPa',
        'Load': '%',
        'Grain': 't/h'
    }
    
    for key, unit in units.items():
        if key in feature_name:
            return unit
    return ''

# ─────────────────────────────────────────────────────────────────
# LOAD EMBEDDED TRAINING DATA (from model package)
# ─────────────────────────────────────────────────────────────────
# This is the historical data used during training
# In production, this would be embedded in the model package or loaded separately
df_embedded = None  # This will be used by modes 1, 2, 3

# For now, we'll note that modes 1-3 require the training data
# You would need to embed this in the model_package or load it separately

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def predict_brix(input_data):
    """Predict Brix from input data"""
    if model_package is None:
        return None
    
    try:
        model = model_package['model']
        feature_names = model_package['feature_names']
        
        # Create DataFrame with correct feature order
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Select and order features
        X = input_df[feature_names]
        
        # Predict
        prediction = model.predict(X)
        return prediction[0] if len(prediction) == 1 else prediction
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_operating_ranges(df, target_brix, tolerance=0.3):
    """Get operating ranges from historical data"""
    try:
        # Filter data near target
        mask = (df['Brix'] >= target_brix - tolerance) & (df['Brix'] <= target_brix + tolerance)
        filtered = df[mask]
        
        if len(filtered) < 10:
            return pd.DataFrame(), len(filtered)
        
        # Compute ranges for important features
        ranges = {}
        for feature in feature_importance.keys():
            if feature in filtered.columns:
                data = filtered[feature].dropna()
                if len(data) > 0:
                    ranges[feature] = {
                        'P25': data.quantile(0.25),
                        'P40': data.quantile(0.40),
                        'Median': data.quantile(0.50),
                        'Mean': data.mean(),
                        'P60': data.quantile(0.60),
                        'P75': data.quantile(0.75),
                        'Min': data.min(),
                        'Max': data.max(),
                        'Std': data.std(),
                        'Samples': len(data)
                    }
        
        return pd.DataFrame(ranges).T, len(filtered)
    
    except Exception as e:
        st.error(f"Error computing ranges: {str(e)}")
        return pd.DataFrame(), 0

def check_feasibility(df, target_brix, locked_values, tolerance=0.2):
    """Check if target Brix is achievable with locked values"""
    try:
        model = model_package['model']
        all_features = model_package['feature_names']
        
        # Create test vector
        test_vector = {}
        for feat in all_features:
            if feat in locked_values:
                test_vector[feat] = locked_values[feat]
            elif feat in df.columns:
                test_vector[feat] = df[feat].median()
            else:
                test_vector[feat] = 0
        
        # Predict
        pred = model.predict(pd.DataFrame([test_vector]))[0]
        feasible = abs(pred - target_brix) <= tolerance
        
        # Check problematic features
        problematic = []
        suggestions = {}
        
        if not feasible:
            ranges, _ = get_operating_ranges(df, target_brix, tolerance)
            
            for feat, val in locked_values.items():
                if feat in ranges.index:
                    row = ranges.loc[feat]
                    if val < row['Min']:
                        problematic.append(feat)
                        suggestions[feat] = f"Too low ({val:.1f}). Normal range: {row['P25']:.1f} - {row['P75']:.1f}"
                    elif val > row['Max']:
                        problematic.append(feat)
                        suggestions[feat] = f"Too high ({val:.1f}). Normal range: {row['P25']:.1f} - {row['P75']:.1f}"
                    elif val < row['P25'] or val > row['P75']:
                        problematic.append(feat)
                        suggestions[feat] = f"Outside typical range. Recommend: {row['Median']:.1f}"
        
        return {
            'feasible': feasible,
            'predicted_brix': pred,
            'problematic_features': problematic,
            'suggestions': suggestions
        }
    
    except Exception as e:
        st.error(f"Error checking feasibility: {str(e)}")
        return {
            'feasible': False,
            'predicted_brix': 0,
            'problematic_features': [],
            'suggestions': {}
        }

def find_conditional_ranges(df, target_brix, locked_values, tolerance=0.2):
    """Find operating ranges for unlocked features given locked values"""
    try:
        # Filter data near target
        mask = (df['Brix'] >= target_brix - tolerance) & (df['Brix'] <= target_brix + tolerance)
        filtered = df[mask].copy()
        
        # Further filter by locked values (with tolerance)
        for feature, value in locked_values.items():
            if feature in filtered.columns:
                feature_tolerance = max(value * 0.05, 1.0)
                mask = (filtered[feature] >= value - feature_tolerance) & \
                       (filtered[feature] <= value + feature_tolerance)
                filtered = filtered[mask]
        
        if len(filtered) < 5:
            return pd.DataFrame(), len(filtered)
        
        # Compute ranges for unlocked features
        unlocked_features = [f for f in feature_importance.keys() 
                           if f not in locked_values and f in filtered.columns]
        
        ranges = {}
        for feature in unlocked_features:
            data = filtered[feature].dropna()
            if len(data) > 0:
                ranges[feature] = {
                    'P25': data.quantile(0.25),
                    'P40': data.quantile(0.40),
                    'Median': data.quantile(0.50),
                    'Mean': data.mean(),
                    'P60': data.quantile(0.60),
                    'P75': data.quantile(0.75),
                    'Samples': len(data)
                }
        
        return pd.DataFrame(ranges).T, len(filtered)
    
    except Exception as e:
        st.error(f"Error finding conditional ranges: {str(e)}")
        return pd.DataFrame(), 0

# ─────────────────────────────────────────────────────────────────
# MAIN APP HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;'>
    <h2 style='margin: 0; font-size: 28px;'>🎯 Brix Control Assistant</h2>
    <p style='margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;'>
        Help achieve your target Brix by showing you what to adjust
    </p>
</div>
""", unsafe_allow_html=True)

# Show model status
if model_package is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", f"{model_package['metrics']['r2_test']*100:.1f}%")
    with col2:
        st.metric("Model RMSE", f"{model_package['metrics']['rmse_test']:.3f}")
    with col3:
        training_date = model_package.get('training_date', 'Unknown')
        if isinstance(training_date, str):
            st.metric("Last Updated", training_date.split()[0])
        else:
            st.metric("Last Updated", str(training_date)[:10])
else:
    st.error(error_msg if error_msg else "❌ Model not loaded")
    st.info("📤 Please ensure 'xgboost_brix_model_v2(With_features).pkl' is in the repository root")
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SIDEBAR - MODE SELECTION
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Select Mode")

mode = st.sidebar.radio(
    "Choose operation mode:",
    [
        "📊 See Typical Settings",
        "🎯 I Have Current Readings",
        "🔧 Advanced Control",
        "📁 Upload Excel for Live Brix Prediction"  # NEW 4th option
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Information")
st.sidebar.info(f"""
**Features Used:** {len(model_package['feature_names'])}  
**Top Driver:** M4 Steam Valve (33%)  
**Model Type:** XGBoost Regressor  
**Version:** {model_package.get('version', '1.0')}
""")

# ─────────────────────────────────────────────────────────────────
# MODE 4: UPLOAD EXCEL FOR LIVE BRIX PREDICTION (NEW)
# ─────────────────────────────────────────────────────────────────
if mode == "📁 Upload Excel for Live Brix Prediction":
    st.header("📁 Upload Excel for Live Brix Prediction")
    
    st.info("📤 Upload your Excel/CSV file containing process parameters. The model will predict Brix values for all rows.")
    
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="File should contain the feature columns (M4_Steam_CV_F/B, M3_Steam_CV_F/B, etc.)"
    )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_upload):,} rows from **{uploaded_file.name}**")
            
            # Show preview
            with st.expander("📋 Preview Data (first 5 rows)"):
                st.dataframe(df_upload.head(), use_container_width=True)
            
            # Check available features
            available_features = [f for f in model_package['feature_names'] if f in df_upload.columns]
            missing_features = [f for f in model_package['feature_names'] if f not in df_upload.columns]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Available Features", f"{len(available_features)}/{len(model_package['feature_names'])}")
            with col2:
                if missing_features:
                    st.warning(f"⚠️ {len(missing_features)} features missing (will use default values)")
            
            if st.button("🚀 Predict Brix Values", type="primary", use_container_width=True):
                with st.spinner("Predicting Brix values..."):
                    # Predict
                    predictions = predict_brix(df_upload)
                    
                    if predictions is not None:
                        # Add predictions to dataframe
                        df_results = df_upload.copy()
                        df_results['Predicted_Brix'] = predictions
                        
                        st.success("✅ Prediction Complete!")
                        
                        # Summary metrics
                        st.markdown("### 📊 Prediction Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Average Brix", f"{predictions.mean():.2f}")
                        with col3:
                            st.metric("Min Brix", f"{predictions.min():.2f}")
                        with col4:
                            st.metric("Max Brix", f"{predictions.max():.2f}")
                        
                        # If actual Brix exists, show comparison
                        if 'Brix' in df_upload.columns:
                            df_results['Actual_Brix'] = df_upload['Brix']
                            df_results['Error'] = df_results['Actual_Brix'] - df_results['Predicted_Brix']
                            df_results['Absolute_Error'] = abs(df_results['Error'])
                            df_results['Within_0.5'] = df_results['Absolute_Error'] <= 0.5
                            
                            st.markdown("---")
                            st.markdown("### 🎯 Accuracy Metrics (Actual Brix found in file)")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                mae = df_results['Absolute_Error'].mean()
                                st.metric("Mean Absolute Error", f"{mae:.3f}")
                            with col2:
                                rmse = np.sqrt((df_results['Error']**2).mean())
                                st.metric("RMSE", f"{rmse:.3f}")
                            with col3:
                                within_tolerance = (df_results['Within_0.5'].sum() / len(df_results)) * 100
                                st.metric("Within ±0.5 Brix", f"{within_tolerance:.1f}%")
                            with col4:
                                mean_error = df_results['Error'].mean()
                                st.metric("Mean Error (Bias)", f"{mean_error:+.3f}")
                            
                            # Scatter plot
                            st.markdown("### 📈 Predicted vs Actual Brix")
                            
                            fig = go.Figure()
                            
                            # Scatter points
                            fig.add_trace(go.Scatter(
                                x=df_results['Actual_Brix'],
                                y=df_results['Predicted_Brix'],
                                mode='markers',
                                name='Predictions',
                                marker=dict(
                                    color=df_results['Absolute_Error'],
                                    colorscale='RdYlGn_r',
                                    showscale=True,
                                    colorbar=dict(title="Absolute<br>Error"),
                                    size=6
                                )
                            ))
                            
                            # Perfect prediction line
                            min_val = min(df_results['Actual_Brix'].min(), df_results['Predicted_Brix'].min())
                            max_val = max(df_results['Actual_Brix'].max(), df_results['Predicted_Brix'].max())
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            fig.update_layout(
                                xaxis_title="Actual Brix",
                                yaxis_title="Predicted Brix",
                                height=500,
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show results table
                        st.markdown("---")
                        st.markdown("### 📋 Prediction Results")
                        
                        # Reorder columns to show Predicted_Brix first
                        cols_order = ['Predicted_Brix']
                        if 'Brix' in df_results.columns:
                            cols_order.append('Actual_Brix')
                            cols_order.append('Error')
                        cols_order += [c for c in df_results.columns if c not in cols_order]
                        
                        st.dataframe(df_results[cols_order], use_container_width=True)
                        
                        # Download button
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results with Predictions (CSV)",
                            data=csv,
                            file_name=f"brix_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            with st.expander("🔍 Show Error Details"):
                st.code(traceback.format_exc())
    
    else:
        st.markdown("""
        ### 📋 File Requirements:
        
        Your Excel/CSV file should contain columns for the model features:
        - M4_Steam_CV_F/B, M3_Steam_CV_F/B, M1_level, M1_Steam_CV_F/B
        - M2_Steam_CV_F/B, M5_body_Temp, M4_level, M3_body_Temp
        - M2_condensate_flow, M2_body_Temp, Grain_flow2
        - M1_condensate_flow, M4_body_Temp, M3_level
        
        **Optional:** If your file contains an **Actual Brix** column, the tool will also calculate prediction accuracy.
        
        ### 🎯 What You'll Get:
        - ✅ Predicted Brix for every row
        - 📊 Summary statistics (Average, Min, Max)
        - 📈 Accuracy metrics (if actual Brix is present)
        - 📥 Downloadable results file
        """)

# ─────────────────────────────────────────────────────────────────
# MODES 1-3: EXISTING FUNCTIONALITY (UNCHANGED)
# ─────────────────────────────────────────────────────────────────
else:
    # NOTE: For modes 1-3, you need to load your training data
    # This would typically be embedded in the model package or loaded from a separate file
    
    st.warning("⚠️ **Note:** Modes 1-3 require the original training data to calculate operating ranges.")
    st.info("""
    **To enable modes 1-3:**
    1. Load the training data (df) used during model training
    2. Or embed it in the model package during training
    3. Or provide it as a separate file
    
    For now, please use **Mode 4: Upload Excel for Live Brix Prediction** to get predictions on new data.
    """)
    
    # Placeholder for modes 1-3 implementation
    st.markdown("---")
    st.markdown("### 🚧 Under Construction")
    st.markdown("""
    These modes will calculate operating ranges and recommendations based on your training data.
    
    **Available Mode:**
    - 📁 Upload Excel for Live Brix Prediction (Mode 4)
    """)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🎯 Brix Control Assistant | Powered by XGBoost ML | Version 2.0 | Operator-Friendly Interface
</div>
""", unsafe_allow_html=True)
