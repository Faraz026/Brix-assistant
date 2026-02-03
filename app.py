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
    'M4_Steam_CV_F/B': 'M4 Steam Control',
    'M3_Steam_CV_F/B': 'M3 Steam Control',
    'M1_level': 'M1 Level',
    'M1_Steam_CV_F/B': 'M1 Steam Control',
    'M2_Steam_CV_F/B': 'M2 Steam Control',
    'M5_body_Temp': 'M5 Body Temperature',
    'M4_level': 'M4 Level',
    'M3_body_Temp': 'M3 Body Temperature',
    'M2_condensate_flow': 'M2 Condensate Flow',
    'M2_body_Temp': 'M2 Body Temperature',
    'Grain_flow2': 'Grain Flow',
    'M1_condensate_flow': 'M1 Condensate Flow',
    'M4_body_Temp': 'M4 Body Temperature',
    'M3_level': 'M3 Level'
}

def get_unit(feature_name):
    """Return appropriate unit for each feature"""
    if 'Steam' in feature_name or 'level' in feature_name:
        return '%'
    elif 'Temp' in feature_name:
        return '°C'
    elif 'flow' in feature_name:
        return 't/h'
    else:
        return ''

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
            st.warning(f"⚠️ Only {len(filtered)} samples found near target {target_brix}±{tolerance}")
            return pd.DataFrame()
        
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
        
        return pd.DataFrame(ranges).T
    
    except Exception as e:
        st.error(f"Error computing ranges: {str(e)}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────
# MAIN APP HEADER (Always show this)
# ─────────────────────────────────────────────────────────────────
st.title("🎯 Brix Control Assistant")
st.markdown("### AI-Powered Operating Range Advisor")

# Show model status
if model_package is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy (R²)", f"{model_package['metrics']['r2_test']:.3f}")
    with col2:
        st.metric("Model RMSE", f"{model_package['metrics']['rmse_test']:.3f}")
    with col3:
        training_date = model_package.get('training_date', 'Unknown')
        if isinstance(training_date, str):
            st.metric("Last Updated", training_date.split()[0])
        else:
            st.metric("Last Updated", str(training_date)[:10])
    
    st.success("✅ Model loaded successfully!")
else:
    st.error(error_msg if error_msg else "❌ Model not loaded")
    st.info("📤 Please ensure 'xgboost_brix_model_v2(With_features).pkl' is in the repository root")
    st.stop()  # Stop execution if model not loaded

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SIDEBAR - MODE SELECTION
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Select Mode")

mode = st.sidebar.radio(
    "Choose operation mode:",
    ["📊 See Typical Settings", "🔮 Predict Brix from Current Readings", "📁 Batch Prediction (Upload Excel)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Information")
st.sidebar.info(f"""
**Features Used:** {len(model_package['feature_names'])}  
**Top Driver:** M4 Steam Control  
**Model Type:** XGBoost Regressor  
**Version:** {model_package.get('version', '1.0')}
""")

# ─────────────────────────────────────────────────────────────────
# MODE 1: SEE TYPICAL SETTINGS
# ─────────────────────────────────────────────────────────────────
if mode == "📊 See Typical Settings":
    st.header("📊 Typical Operating Settings")
    
    st.info("👉 Upload historical data to see typical parameter ranges for your target Brix")
    
    # File uploader for historical data
    uploaded_historical = st.file_uploader(
        "Upload Historical Data (Excel)",
        type=['xlsx', 'xls'],
        key='historical_upload',
        help="Excel file should contain a 'Brix' column and feature columns"
    )
    
    if uploaded_historical:
        try:
            df_historical = pd.read_excel(uploaded_historical)
            
            if 'Brix' not in df_historical.columns:
                st.error("❌ Excel file must contain a 'Brix' column")
            else:
                st.success(f"✅ Loaded {len(df_historical)} rows of historical data")
                
                # Target Brix input
                col1, col2 = st.columns(2)
                with col1:
                    target_brix = st.number_input(
                        "Target Brix",
                        min_value=91.0,
                        max_value=95.0,
                        value=93.0,
                        step=0.1
                    )
                with col2:
                    tolerance = st.number_input(
                        "Tolerance (±)",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.3,
                        step=0.1
                    )
                
                if st.button("🔍 Calculate Ranges", type="primary"):
                    with st.spinner("Calculating operating ranges..."):
                        ranges = get_operating_ranges(df_historical, target_brix, tolerance)
                        
                        if not ranges.empty:
                            st.success(f"✅ Found operating ranges for Brix = {target_brix}±{tolerance}")
                            
                            # Display top features
                            st.subheader("🎯 Recommended Operating Ranges")
                            
                            for feature in list(feature_importance.keys())[:8]:
                                if feature in ranges.index:
                                    row = ranges.loc[feature]
                                    importance = feature_importance[feature]
                                    
                                    # Impact indicator
                                    if importance > 0.20:
                                        impact = "🔴 HIGH"
                                    elif importance > 0.10:
                                        impact = "🟡 MEDIUM"
                                    else:
                                        impact = "🟢 LOW"
                                    
                                    st.markdown(f"**{friendly_names.get(feature, feature)}** {impact}")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Low (P25)", f"{row['P25']:.1f} {get_unit(feature)}")
                                    with col2:
                                        st.metric("Normal (Median)", f"{row['Median']:.1f} {get_unit(feature)}")
                                    with col3:
                                        st.metric("High (P75)", f"{row['P75']:.1f} {get_unit(feature)}")
                                    
                                    st.markdown("---")
                        else:
                            st.warning("⚠️ Not enough data found near the target Brix")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.warning("👆 Please upload historical data Excel file to continue")

# ─────────────────────────────────────────────────────────────────
# MODE 2: PREDICT FROM CURRENT READINGS
# ─────────────────────────────────────────────────────────────────
elif mode == "🔮 Predict Brix from Current Readings":
    st.header("🔮 Predict Brix from Current Readings")
    
    st.info("Enter current parameter values to predict Brix")
    
    # Input fields for top features
    input_data = {}
    
    st.subheader("Enter Current Values")
    
    # Create input fields for top features
    num_cols = 2
    features_to_input = list(feature_importance.keys())[:10]  # Top 10 features
    
    for i in range(0, len(features_to_input), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(features_to_input):
                feature = features_to_input[i + j]
                with col:
                    value = st.number_input(
                        f"{friendly_names.get(feature, feature)} ({get_unit(feature)})",
                        value=0.0,
                        step=0.1,
                        key=f"input_{feature}"
                    )
                    input_data[feature] = value
    
    # Fill remaining features with zeros
    for feature in model_package['feature_names']:
        if feature not in input_data:
            input_data[feature] = 0.0
    
    if st.button("🎯 Predict Brix", type="primary"):
        with st.spinner("Predicting..."):
            predicted_brix = predict_brix(input_data)
            
            if predicted_brix is not None:
                st.success("✅ Prediction Complete!")
                
                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Predicted Brix",
                        f"{predicted_brix:.2f}",
                        delta=None
                    )
                with col2:
                    target = 93.5
                    delta = predicted_brix - target
                    st.metric(
                        f"vs Target ({target})",
                        f"{delta:+.2f}",
                        delta=f"{delta:+.2f}",
                        delta_color="inverse"
                    )
                
                # Status indicator
                if 93.0 <= predicted_brix <= 94.0:
                    st.success("✅ **Within Target Range** (93.0 - 94.0)")
                elif predicted_brix < 93.0:
                    st.warning(f"⚠️ **Below Target** - Need to increase Brix by {93.0 - predicted_brix:.2f}")
                else:
                    st.warning(f"⚠️ **Above Target** - Need to decrease Brix by {predicted_brix - 94.0:.2f}")

# ─────────────────────────────────────────────────────────────────
# MODE 3: BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────
elif mode == "📁 Batch Prediction (Upload Excel)":
    st.header("📁 Batch Prediction from Excel")
    
    st.info("Upload an Excel file with parameter columns to predict Brix for multiple rows")
    
    uploaded_batch = st.file_uploader(
        "Upload Batch Data (Excel)",
        type=['xlsx', 'xls'],
        key='batch_upload',
        help="Excel file should contain feature columns"
    )
    
    if uploaded_batch:
        try:
            df_batch = pd.read_excel(uploaded_batch)
            st.success(f"✅ Loaded {len(df_batch)} rows")
            
            # Show preview
            with st.expander("📋 Preview Data (first 5 rows)"):
                st.dataframe(df_batch.head())
            
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Running predictions..."):
                    # Predict
                    predictions = predict_brix(df_batch)
                    
                    if predictions is not None:
                        df_batch['Predicted_Brix'] = predictions
                        
                        # If actual Brix exists, compute error
                        if 'Brix' in df_batch.columns:
                            df_batch['Actual_Brix'] = df_batch['Brix']
                            df_batch['Error'] = df_batch['Actual_Brix'] - df_batch['Predicted_Brix']
                            df_batch['Within_0.5'] = (abs(df_batch['Error']) <= 0.5)
                            
                            # Summary metrics
                            st.subheader("📊 Batch Prediction Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Predictions", len(df_batch))
                            with col2:
                                st.metric("Avg Predicted Brix", f"{df_batch['Predicted_Brix'].mean():.2f}")
                            with col3:
                                st.metric("Avg Error", f"{df_batch['Error'].mean():.3f}")
                            with col4:
                                accuracy = (df_batch['Within_0.5'].sum() / len(df_batch)) * 100
                                st.metric("Within ±0.5 Brix", f"{accuracy:.1f}%")
                            
                            # Plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_batch['Actual_Brix'],
                                y=df_batch['Predicted_Brix'],
                                mode='markers',
                                name='Predictions',
                                marker=dict(color='blue', size=6)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[df_batch['Actual_Brix'].min(), df_batch['Actual_Brix'].max()],
                                y=[df_batch['Actual_Brix'].min(), df_batch['Actual_Brix'].max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(
                                title="Predicted vs Actual Brix",
                                xaxis_title="Actual Brix",
                                yaxis_title="Predicted Brix",
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.subheader("📊 Batch Prediction Summary")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Predictions", len(df_batch))
                            with col2:
                                st.metric("Avg Predicted Brix", f"{df_batch['Predicted_Brix'].mean():.2f}")
                        
                        # Show results
                        st.subheader("📋 Prediction Results")
                        st.dataframe(df_batch[['Predicted_Brix'] + [c for c in df_batch.columns if c != 'Predicted_Brix']])
                        
                        # Download button
                        csv = df_batch.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"brix_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code(traceback.format_exc())
    else:
        st.warning("👆 Please upload batch data Excel file to continue")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🎯 Brix Control Assistant | Powered by XGBoost ML | Version 2.0
</div>
""", unsafe_allow_html=True)
