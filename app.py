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
        
        # Extract training data if available
        df_embedded = model_package.get('training_data_sample', None)
        
        return model_package, df_embedded, None
    except FileNotFoundError:
        return None, None, "❌ Model file not found"
    except Exception as e:
        return None, None, f"❌ Error loading model: {str(e)}"

model_package, df_embedded, error_msg = load_model()

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
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
    units = {
        'Steam_CV': '%', 'Steam': '%', 'Temp': '°C', 'level': '%',
        'flow': 't/h', 'condensate': 't/h', 'Vaccum': 'kPa',
        'Load': '%', 'Grain': 't/h'
    }
    for key, unit in units.items():
        if key in feature_name:
            return unit
    return ''

# ─────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
def predict_brix(input_data):
    if model_package is None:
        return None
    try:
        model = model_package['model']
        feature_names = model_package['feature_names']
        
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        X = input_df[feature_names]
        prediction = model.predict(X)
        return prediction[0] if len(prediction) == 1 else prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def get_operating_ranges(df, target_brix, tolerance=0.3):
    try:
        mask = (df['Brix'] >= target_brix - tolerance) & (df['Brix'] <= target_brix + tolerance)
        filtered = df[mask]
        
        if len(filtered) < 10:
            return pd.DataFrame(), len(filtered)
        
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
        st.error(f"Error: {str(e)}")
        return pd.DataFrame(), 0

def check_feasibility(df, target_brix, locked_values, tolerance=0.2):
    try:
        model = model_package['model']
        all_features = model_package['feature_names']
        
        test_vector = {}
        for feat in all_features:
            if feat in locked_values:
                test_vector[feat] = locked_values[feat]
            elif feat in df.columns:
                test_vector[feat] = df[feat].median()
            else:
                test_vector[feat] = 0
        
        pred = model.predict(pd.DataFrame([test_vector]))[0]
        feasible = abs(pred - target_brix) <= tolerance
        
        problematic = []
        suggestions = {}
        
        if not feasible:
            ranges, _ = get_operating_ranges(df, target_brix, tolerance)
            
            for feat, val in locked_values.items():
                if feat in ranges.index:
                    row = ranges.loc[feat]
                    if val < row['Min']:
                        problematic.append(feat)
                        suggestions[feat] = f"Too low ({val:.1f}). Normal: {row['P25']:.1f}-{row['P75']:.1f}"
                    elif val > row['Max']:
                        problematic.append(feat)
                        suggestions[feat] = f"Too high ({val:.1f}). Normal: {row['P25']:.1f}-{row['P75']:.1f}"
                    elif val < row['P25'] or val > row['P75']:
                        problematic.append(feat)
                        suggestions[feat] = f"Outside typical. Recommend: {row['Median']:.1f}"
        
        return {
            'feasible': feasible,
            'predicted_brix': pred,
            'problematic_features': problematic,
            'suggestions': suggestions
        }
    except Exception as e:
        return {'feasible': False, 'predicted_brix': 0, 'problematic_features': [], 'suggestions': {}}

def find_conditional_ranges(df, target_brix, locked_values, tolerance=0.2):
    try:
        mask = (df['Brix'] >= target_brix - tolerance) & (df['Brix'] <= target_brix + tolerance)
        filtered = df[mask].copy()
        
        for feature, value in locked_values.items():
            if feature in filtered.columns:
                feature_tolerance = max(value * 0.05, 1.0)
                mask = (filtered[feature] >= value - feature_tolerance) & \
                       (filtered[feature] <= value + feature_tolerance)
                filtered = filtered[mask]
        
        if len(filtered) < 5:
            return pd.DataFrame(), len(filtered)
        
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
        return pd.DataFrame(), 0

# ─────────────────────────────────────────────────────────────────
# HEADER
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
    
    if df_embedded is not None:
        st.success(f"✅ Model loaded with {len(df_embedded):,} training samples")
    else:
        st.warning("⚠️ Training data not embedded - modes 1-3 limited")
else:
    st.error(error_msg if error_msg else "❌ Model not loaded")
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("🎛️ Select Mode")

if df_embedded is not None:
    mode_options = [
        "📊 See Typical Settings",
        "🎯 I Have Current Readings",
        "🔧 Advanced Control",
        "📁 Upload Excel for Live Brix"
    ]
else:
    mode_options = ["📁 Upload Excel for Live Brix"]
    st.sidebar.warning("⚠️ Only Mode 4 available (training data not embedded)")

mode = st.sidebar.radio("Choose mode:", mode_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Info")
st.sidebar.info(f"""
**Features:** {len(model_package['feature_names'])}  
**Top Driver:** M4 Steam Valve (33%)  
**Type:** XGBoost Regressor  
**Version:** {model_package.get('version', '1.0')}
""")

if df_embedded is not None:
    st.sidebar.success(f"**Training Samples:** {len(df_embedded):,}")

# ─────────────────────────────────────────────────────────────────
# MODE 4: UPLOAD EXCEL FOR LIVE BRIX
# ─────────────────────────────────────────────────────────────────
if mode == "📁 Upload Excel for Live Brix":
    st.header("📁 Upload Excel for Live Brix Prediction")
    
    st.info("📤 Upload Excel/CSV with process parameters")
    
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV",
        type=['xlsx', 'xls', 'csv'],
        help="File should contain feature columns"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_upload):,} rows from **{uploaded_file.name}**")
            
            with st.expander("📋 Preview (first 5 rows)"):
                st.dataframe(df_upload.head(), use_container_width=True)
            
            available = [f for f in model_package['feature_names'] if f in df_upload.columns]
            missing = [f for f in model_package['feature_names'] if f not in df_upload.columns]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Features", f"{len(available)}/{len(model_package['feature_names'])}")
            with col2:
                if missing:
                    st.warning(f"⚠️ {len(missing)} missing")
            
            if st.button("🚀 Predict Brix", type="primary", use_container_width=True):
                with st.spinner("Predicting..."):
                    predictions = predict_brix(df_upload)
                    
                    if predictions is not None:
                        df_results = df_upload.copy()
                        df_results['Predicted_Brix'] = predictions
                        
                        st.success("✅ Complete!")
                        
                        st.markdown("### 📊 Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", len(predictions))
                        with col2:
                            st.metric("Avg Brix", f"{predictions.mean():.2f}")
                        with col3:
                            st.metric("Min", f"{predictions.min():.2f}")
                        with col4:
                            st.metric("Max", f"{predictions.max():.2f}")
                        
                        if 'Brix' in df_upload.columns:
                            df_results['Actual_Brix'] = df_upload['Brix']
                            df_results['Error'] = df_results['Actual_Brix'] - df_results['Predicted_Brix']
                            df_results['Absolute_Error'] = abs(df_results['Error'])
                            df_results['Within_0.5'] = df_results['Absolute_Error'] <= 0.5
                            
                            st.markdown("---")
                            st.markdown("### 🎯 Accuracy")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("MAE", f"{df_results['Absolute_Error'].mean():.3f}")
                            with col2:
                                st.metric("RMSE", f"{np.sqrt((df_results['Error']**2).mean()):.3f}")
                            with col3:
                                pct = (df_results['Within_0.5'].sum() / len(df_results)) * 100
                                st.metric("Within ±0.5", f"{pct:.1f}%")
                            with col4:
                                st.metric("Bias", f"{df_results['Error'].mean():+.3f}")
                            
                            st.markdown("### 📈 Predicted vs Actual")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_results['Actual_Brix'],
                                y=df_results['Predicted_Brix'],
                                mode='markers',
                                name='Predictions',
                                marker=dict(color=df_results['Absolute_Error'], 
                                          colorscale='RdYlGn_r', showscale=True,
                                          colorbar=dict(title="Error"), size=6)
                            ))
                            min_v = min(df_results['Actual_Brix'].min(), df_results['Predicted_Brix'].min())
                            max_v = max(df_results['Actual_Brix'].max(), df_results['Predicted_Brix'].max())
                            fig.add_trace(go.Scatter(
                                x=[min_v, max_v], y=[min_v, max_v],
                                mode='lines', name='Perfect',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        st.markdown("### 📋 Results")
                        
                        cols = ['Predicted_Brix']
                        if 'Brix' in df_results.columns:
                            cols += ['Actual_Brix', 'Error']
                        cols += [c for c in df_results.columns if c not in cols]
                        
                        st.dataframe(df_results[cols], use_container_width=True)
                        
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"brix_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("Details"):
                st.code(traceback.format_exc())
    else:
        st.markdown("""
        ### 📋 Requirements:
        Excel/CSV with columns: M4_Steam_CV_F/B, M3_Steam_CV_F/B, M1_level, M1_Steam_CV_F/B, 
        M2_Steam_CV_F/B, M5_body_Temp, M4_level, M3_body_Temp, M2_condensate_flow, 
        M2_body_Temp, Grain_flow2, M1_condensate_flow, M4_body_Temp, M3_level
        
        **Optional:** 'Brix' column for accuracy
        """)

# ─────────────────────────────────────────────────────────────────
# MODES 1-3 (with training data)
# ─────────────────────────────────────────────────────────────────
elif df_embedded is not None:
    st.markdown("### 🎯 Target Settings")
    col1, col2 = st.columns(2)
    with col1:
        target_brix = st.number_input("Target Brix", 
            min_value=float(df_embedded['Brix'].min()),
            max_value=float(df_embedded['Brix'].max()),
            value=93.0, step=0.1)
    with col2:
        tolerance = st.number_input("± Range", 
            min_value=0.1, max_value=2.0, value=0.3, step=0.1)
    
    st.markdown("---")
    
    # MODE 1: SEE TYPICAL SETTINGS
    if mode == "📊 See Typical Settings":
        st.header("📊 Typical Operating Settings")
        st.markdown(f"### Parameters for Brix = **{target_brix}±{tolerance}**")
        
        with st.spinner("Calculating..."):
            ranges, sample_count = get_operating_ranges(df_embedded, target_brix, tolerance)
            
            if not ranges.empty:
                st.success(f"✅ Found **{sample_count}** similar situations")
                
                st.markdown("### 🎯 Recommended Settings")
                
                sorted_features = [f for f in feature_importance.keys() if f in ranges.index]
                
                for i, feature in enumerate(sorted_features, 1):
                    row = ranges.loc[feature]
                    importance = feature_importance[feature]
                    
                    if importance > 0.20:
                        impact = "🔴 HIGH IMPACT"
                        expanded = (i <= 5)
                    elif importance > 0.10:
                        impact = "🟡 MEDIUM"
                        expanded = (i <= 3)
                    else:
                        impact = "🟢 LOW"
                        expanded = False
                    
                    with st.expander(f"**{i}. {friendly_names.get(feature, feature)}** - {impact}", expanded=expanded):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Low (P25)", f"{row['P25']:.1f} {get_unit(feature)}")
                        with col2:
                            st.metric("Normal (Median)", f"{row['Median']:.1f} {get_unit(feature)}")
                        with col3:
                            st.metric("High (P75)", f"{row['P75']:.1f} {get_unit(feature)}")
                        
                        st.info(f"💡 Keep between **{row['P40']:.1f}** and **{row['P60']:.1f}** {get_unit(feature)}")
            else:
                st.warning(f"⚠️ Not enough data near {target_brix}±{tolerance}")
    
    # MODE 2: I HAVE CURRENT READINGS
    elif mode == "🎯 I Have Current Readings":
        st.header("🎯 I Have Current Readings")
        st.markdown(f"### Target: Brix = **{target_brix}±{tolerance}**")
        
        st.info("💡 Check parameters you can see now")
        
        st.markdown("### 📋 Enter Current Values")
        
        locked_values = {}
        sorted_features = [f for f in feature_importance.keys() if f in df_embedded.columns]
        
        for i in range(0, len(sorted_features[:10]), 2):
            col1, col2 = st.columns(2)
            
            for j, col in enumerate([col1, col2]):
                if i + j < len(sorted_features[:10]):
                    feature = sorted_features[i + j]
                    
                    with col:
                        st.markdown(f"**{friendly_names.get(feature, feature)}**")
                        
                        use_feature = st.checkbox(
                            "I have this value",
                            key=f"check_{feature}",
                            value=False
                        )
                        
                        if use_feature:
                            value = st.number_input(
                                f"Value ({get_unit(feature)})",
                                value=float(df_embedded[feature].median()),
                                min_value=float(df_embedded[feature].quantile(0.01)),
                                max_value=float(df_embedded[feature].quantile(0.99)),
                                step=0.1,
                                key=f"value_{feature}"
                            )
                            locked_values[feature] = value
        
        st.markdown("---")
        
        if st.button("🔍 Check Feasibility", type="primary", use_container_width=True):
            if len(locked_values) == 0:
                st.warning("⚠️ Please check at least one parameter")
            else:
                with st.spinner("Analyzing..."):
                    feasibility = check_feasibility(df_embedded, target_brix, locked_values, tolerance)
                    
                    st.markdown("### 📊 Status")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Predicted", f"{feasibility['predicted_brix']:.2f}")
                    with col2:
                        st.metric("Target", f"{target_brix:.2f}")
                    with col3:
                        delta = feasibility['predicted_brix'] - target_brix
                        st.metric("Difference", f"{delta:+.2f}")
                    
                    st.markdown("---")
                    
                    if feasibility['feasible']:
                        st.success("✅ **Target ACHIEVABLE!**")
                        
                        st.markdown("### 📌 Keep Steady:")
                        for feature, value in locked_values.items():
                            st.markdown(f"✓ **{friendly_names.get(feature, feature)}**: {value:.1f} {get_unit(feature)}")
                        
                        conditional_ranges, sample_count = find_conditional_ranges(df_embedded, target_brix, locked_values, tolerance)
                        
                        if not conditional_ranges.empty:
                            st.markdown(f"### 🎛️ Adjust These ({sample_count} samples)")
                            
                            unlocked_sorted = [f for f in feature_importance.keys() 
                                             if f in conditional_ranges.index]
                            
                            for feature in unlocked_sorted[:7]:
                                if feature in conditional_ranges.index:
                                    row = conditional_ranges.loc[feature]
                                    
                                    with st.expander(f"**{friendly_names.get(feature, feature)}**"):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Low", f"{row['P40']:.1f} {get_unit(feature)}")
                                        with col2:
                                            st.metric("Target", f"{row['Median']:.1f} {get_unit(feature)}")
                                        with col3:
                                            st.metric("High", f"{row['P60']:.1f} {get_unit(feature)}")
                    else:
                        st.error("❌ **NOT achievable**")
                        
                        if feasibility['problematic_features']:
                            st.markdown("### ⚠️ Problems:")
                            for feature in feasibility['problematic_features']:
                                st.warning(f"**{friendly_names.get(feature, feature)}**: {feasibility['suggestions'][feature]}")
    
    # MODE 3: ADVANCED CONTROL
    elif mode == "🔧 Advanced Control":
        st.header("🔧 Advanced Control")
        
        with st.spinner("Generating ranges..."):
            ranges, sample_count = get_operating_ranges(df_embedded, target_brix, tolerance)
            
            if not ranges.empty:
                st.success(f"✅ Ranges for Brix = {target_brix}±{tolerance} ({sample_count} samples)")
                
                display_df = ranges[['P25', 'P40', 'Median', 'Mean', 'P60', 'P75', 'Min', 'Max', 'Std', 'Samples']].copy()
                
                display_df.insert(0, 'Feature', [friendly_names.get(f, f) for f in display_df.index])
                display_df.insert(1, 'Unit', [get_unit(f) for f in ranges.index])
                display_df.insert(2, 'Importance %', [feature_importance.get(f, 0) * 100 for f in ranges.index])
                
                display_df = display_df.sort_values('Importance %', ascending=False)
                
                st.dataframe(
                    display_df.style.format({
                        'Importance %': '{:.1f}',
                        'P25': '{:.1f}', 'P40': '{:.1f}', 'Median': '{:.1f}',
                        'Mean': '{:.1f}', 'P60': '{:.1f}', 'P75': '{:.1f}',
                        'Min': '{:.1f}', 'Max': '{:.1f}', 'Std': '{:.2f}',
                        'Samples': '{:.0f}'
                    }).background_gradient(subset=['Importance %'], cmap='Reds'),
                    use_container_width=True
                )
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download (CSV)",
                    data=csv,
                    file_name=f"ranges_{target_brix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning(f"⚠️ Not enough data")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🎯 Brix Control Assistant | XGBoost ML | v2.0
</div>
""", unsafe_allow_html=True)
