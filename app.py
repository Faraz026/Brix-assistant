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

def check_feasibility(df, target_brix, locked_values, tolerance=0.2):
    """Check if target Brix is achievable with locked values"""
    try:
        # Get ranges near target
        ranges = get_operating_ranges(df, target_brix, tolerance)
        
        if ranges.empty:
            return False, ["Not enough historical data near target"]
        
        problems = []
        for feature, value in locked_values.items():
            if feature in ranges.index:
                row = ranges.loc[feature]
                if value < row['Min'] or value > row['Max']:
                    problems.append(f"{friendly_names.get(feature, feature)}: {value:.1f} is outside normal range ({row['Min']:.1f}-{row['Max']:.1f})")
                elif value < row['P25'] or value > row['P75']:
                    problems.append(f"{friendly_names.get(feature, feature)}: {value:.1f} is outside typical range ({row['P25']:.1f}-{row['P75']:.1f})")
        
        return len(problems) == 0, problems
    
    except Exception as e:
        return False, [f"Error checking feasibility: {str(e)}"]

def find_conditional_ranges(df, target_brix, locked_values, tolerance=0.2, n_solutions=50):
    """Find operating ranges for unlocked features given locked values"""
    try:
        # Filter data near target
        mask = (df['Brix'] >= target_brix - tolerance) & (df['Brix'] <= target_brix + tolerance)
        filtered = df[mask].copy()
        
        # Further filter by locked values (with tolerance)
        for feature, value in locked_values.items():
            if feature in filtered.columns:
                feature_tolerance = value * 0.05  # 5% tolerance
                mask = (filtered[feature] >= value - feature_tolerance) & \
                       (filtered[feature] <= value + feature_tolerance)
                filtered = filtered[mask]
        
        if len(filtered) < 5:
            return pd.DataFrame(), filtered
        
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
        
        return pd.DataFrame(ranges).T, filtered
    
    except Exception as e:
        st.error(f"Error finding conditional ranges: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# ─────────────────────────────────────────────────────────────────
# MAIN APP HEADER (Always show this)
# ─────────────────────────────────────────────────────────────────
st.title("🎯 Brix Control Assistant")
st.markdown("### AI-Powered Operating Range Advisor for Operators")

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
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# DATA UPLOAD SECTION
# ─────────────────────────────────────────────────────────────────
st.sidebar.header("📁 Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload Historical Data (Excel)",
    type=['xlsx', 'xls'],
    help="Excel file should contain 'Brix' column and all feature columns"
)

df = None
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        
        # Validate required columns
        if 'Brix' not in df.columns:
            st.sidebar.error("❌ Excel must contain 'Brix' column")
            df = None
        else:
            # Check for required features
            missing_features = [f for f in feature_importance.keys() if f not in df.columns]
            if missing_features:
                st.sidebar.warning(f"⚠️ Missing features: {', '.join(missing_features[:3])}...")
            
            st.sidebar.success(f"✅ Loaded {len(df):,} rows")
            st.sidebar.info(f"📊 Brix range: {df['Brix'].min():.1f} - {df['Brix'].max():.1f}")
    
    except Exception as e:
        st.sidebar.error(f"❌ Error loading file: {str(e)}")
        df = None

if df is None:
    st.info("👈 **Please upload historical data Excel file in the sidebar to begin**")
    st.markdown("""
    ### 📋 Required Data Format:
    
    Your Excel file should contain:
    - **Brix** column (target variable)
    - **Feature columns**: M4_Steam_CV_F/B, M3_Steam_CV_F/B, M1_level, M1_Steam_CV_F/B, M2_Steam_CV_F/B, 
      M5_body_Temp, M4_level, M3_body_Temp, M2_condensate_flow, M2_body_Temp, Grain_flow2, 
      M1_condensate_flow, M4_body_Temp, M3_level
    
    ### 🎯 What This Tool Does:
    
    1. **See Typical Settings** - View normal operating ranges for achieving target Brix
    2. **I Have Current Readings** - Input current values and get recommendations to reach target
    3. **Advanced Control** - Detailed technical view with all operating parameters
    """)
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR - MODE SELECTION
# ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Select Mode")

mode = st.sidebar.radio(
    "Choose operation mode:",
    ["📊 Simple - See Typical Settings", "🔧 Guided - I Have Current Readings", "⚙️ Advanced - Full Control"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Information")
st.sidebar.info(f"""
**Features Used:** {len(model_package['feature_names'])}  
**Top Driver:** M4 Steam Control (33%)  
**Model Type:** XGBoost Regressor  
**Version:** {model_package.get('version', '1.0')}
""")

# ─────────────────────────────────────────────────────────────────
# COMMON CONTROLS
# ─────────────────────────────────────────────────────────────────
st.header("🎯 Target Settings")

col1, col2 = st.columns(2)
with col1:
    target_brix = st.number_input(
        "🎯 Target Brix",
        min_value=91.0,
        max_value=95.0,
        value=93.5,
        step=0.1,
        help="Your desired Brix value"
    )
with col2:
    tolerance = st.number_input(
        "📏 Tolerance (±)",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Acceptable deviation from target"
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────
# MODE 1: SIMPLE - SEE TYPICAL SETTINGS
# ─────────────────────────────────────────────────────────────────
if mode == "📊 Simple - See Typical Settings":
    st.header("📊 Typical Operating Settings")
    st.markdown(f"### What parameters should I use to get Brix = **{target_brix}±{tolerance}**?")
    
    with st.spinner("Calculating typical settings..."):
        ranges = get_operating_ranges(df, target_brix, tolerance)
        
        if not ranges.empty:
            st.success(f"✅ Found {ranges.loc[list(ranges.index)[0], 'Samples']:.0f} similar situations in your data")
            
            # Sort by importance
            sorted_features = [f for f in feature_importance.keys() if f in ranges.index]
            
            st.markdown("### 🎯 Recommended Settings (sorted by impact)")
            
            for i, feature in enumerate(sorted_features[:10], 1):
                row = ranges.loc[feature]
                importance = feature_importance[feature]
                
                # Impact indicator
                if importance > 0.20:
                    impact = "🔴 HIGH IMPACT"
                    impact_text = "Adjust this first if Brix is off-target"
                elif importance > 0.10:
                    impact = "🟡 MEDIUM IMPACT"
                    impact_text = "Important for fine-tuning"
                else:
                    impact = "🟢 LOW IMPACT"
                    impact_text = "Keep stable, don't change often"
                
                with st.expander(f"**{i}. {friendly_names.get(feature, feature)}** - {impact}", expanded=(i <= 5)):
                    st.markdown(f"*{impact_text}*")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔽 Low (P25)", f"{row['P25']:.1f} {get_unit(feature)}")
                        st.caption("Minimum typical value")
                    with col2:
                        st.metric("🎯 Normal (Median)", f"{row['Median']:.1f} {get_unit(feature)}")
                        st.caption("Most common value")
                    with col3:
                        st.metric("🔼 High (P75)", f"{row['P75']:.1f} {get_unit(feature)}")
                        st.caption("Maximum typical value")
                    
                    st.info(f"💡 **Operator Tip:** Keep {friendly_names.get(feature, feature)} between **{row['P40']:.1f}** and **{row['P60']:.1f}** {get_unit(feature)} for best results")
        
        else:
            st.warning(f"⚠️ Not enough data found near Brix = {target_brix}±{tolerance}")
            st.info("💡 Try increasing the tolerance or check if this target is within your historical range")

# ─────────────────────────────────────────────────────────────────
# MODE 2: GUIDED - I HAVE CURRENT READINGS
# ─────────────────────────────────────────────────────────────────
elif mode == "🔧 Guided - I Have Current Readings":
    st.header("🔧 Guided Mode - Current Readings")
    st.markdown(f"### I want to reach Brix = **{target_brix}±{tolerance}**. What should I adjust?")
    
    st.info("💡 Enter your current readings below. Check only the parameters you can see right now.")
    
    # Input current values
    locked_values = {}
    
    st.subheader("📋 Enter Current Parameter Values")
    
    sorted_features = [f for f in feature_importance.keys() if f in df.columns]
    
    for i, feature in enumerate(sorted_features[:10], 1):
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{i}. {friendly_names.get(feature, feature)}**")
        
        with col2:
            use_feature = st.checkbox(
                f"I have this value",
                key=f"check_{feature}",
                value=False
            )
        
        with col3:
            if use_feature:
                value = st.number_input(
                    f"Value ({get_unit(feature)})",
                    value=0.0,
                    step=0.1,
                    key=f"value_{feature}",
                    label_visibility="collapsed"
                )
                locked_values[feature] = value
    
    st.markdown("---")
    
    if st.button("🔍 Check Feasibility & Get Recommendations", type="primary"):
        if len(locked_values) == 0:
            st.warning("⚠️ Please check at least one parameter and enter its value")
        else:
            with st.spinner("Analyzing your current readings..."):
                # Check feasibility
                is_feasible, problems = check_feasibility(df, target_brix, locked_values, tolerance)
                
                # Predict current Brix
                input_data = locked_values.copy()
                for feature in model_package['feature_names']:
                    if feature not in input_data:
                        input_data[feature] = 0
                
                current_brix = predict_brix(input_data)
                
                if current_brix is not None:
                    st.subheader("📊 Current Status")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Predicted Brix", f"{current_brix:.2f}")
                    with col2:
                        st.metric("Target Brix", f"{target_brix:.2f}")
                    with col3:
                        delta = current_brix - target_brix
                        st.metric("Difference", f"{delta:+.2f}", delta=f"{delta:+.2f}", delta_color="inverse")
                
                st.markdown("---")
                
                if is_feasible:
                    st.success("✅ **Target is achievable!** Your current readings are in good ranges.")
                    
                    # Get conditional ranges
                    conditional_ranges, filtered = find_conditional_ranges(df, target_brix, locked_values, tolerance)
                    
                    if not conditional_ranges.empty:
                        st.subheader("🎯 Recommended Settings for Other Parameters")
                        st.markdown(f"*Based on {len(filtered)} similar situations where Brix = {target_brix}±{tolerance}*")
                        
                        st.markdown("### 📌 Keep These Steady:")
                        for feature, value in locked_values.items():
                            st.markdown(f"- **{friendly_names.get(feature, feature)}**: {value:.1f} {get_unit(feature)} ✅")
                        
                        st.markdown("### 🎛️ Adjust These To:")
                        
                        # Sort unlocked features by importance
                        unlocked_sorted = [f for f in feature_importance.keys() 
                                         if f in conditional_ranges.index]
                        
                        for feature in unlocked_sorted[:5]:
                            if feature in conditional_ranges.index:
                                row = conditional_ranges.loc[feature]
                                importance = feature_importance[feature]
                                
                                if importance > 0.15:
                                    priority = "🔴 HIGH PRIORITY"
                                else:
                                    priority = "🟡 MEDIUM PRIORITY"
                                
                                st.markdown(f"**{friendly_names.get(feature, feature)}** - {priority}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Low", f"{row['P40']:.1f} {get_unit(feature)}")
                                with col2:
                                    st.metric("Target", f"{row['Median']:.1f} {get_unit(feature)}")
                                with col3:
                                    st.metric("High", f"{row['P60']:.1f} {get_unit(feature)}")
                    
                    else:
                        st.warning("⚠️ Not enough similar data to give specific recommendations")
                
                else:
                    st.error("❌ **Target may NOT be achievable with current readings**")
                    
                    st.subheader("⚠️ Problems Detected:")
                    for problem in problems:
                        st.warning(f"- {problem}")
                    
                    st.info("💡 **Suggestion:** Adjust the problem parameters first, then check again")

# ─────────────────────────────────────────────────────────────────
# MODE 3: ADVANCED - FULL CONTROL
# ─────────────────────────────────────────────────────────────────
elif mode == "⚙️ Advanced - Full Control":
    st.header("⚙️ Advanced Control - Full Technical View")
    
    with st.spinner("Generating full operating ranges..."):
        ranges = get_operating_ranges(df, target_brix, tolerance)
        
        if not ranges.empty:
            st.success(f"✅ Operating ranges for Brix = {target_brix}±{tolerance}")
            
            # Full table view
            st.subheader("📊 Complete Operating Ranges Table")
            
            # Prepare display dataframe
            display_df = ranges[['P25', 'P40', 'Median', 'Mean', 'P60', 'P75', 'Min', 'Max', 'Std', 'Samples']].copy()
            display_df.index = [friendly_names.get(f, f) for f in display_df.index]
            
            # Add units
            units = [get_unit(f) for f in ranges.index]
            display_df.insert(0, 'Unit', units)
            
            # Add importance
            importances = [feature_importance.get(f, 0) * 100 for f in ranges.index]
            display_df.insert(0, 'Importance %', importances)
            
            # Sort by importance
            display_df = display_df.sort_values('Importance %', ascending=False)
            
            st.dataframe(
                display_df.style.format({
                    'Importance %': '{:.1f}',
                    'P25': '{:.1f}',
                    'P40': '{:.1f}',
                    'Median': '{:.1f}',
                    'Mean': '{:.1f}',
                    'P60': '{:.1f}',
                    'P75': '{:.1f}',
                    'Min': '{:.1f}',
                    'Max': '{:.1f}',
                    'Std': '{:.2f}',
                    'Samples': '{:.0f}'
                }).background_gradient(subset=['Importance %'], cmap='Reds'),
                use_container_width=True
            )
            
            # Download button
            csv = display_df.to_csv()
            st.download_button(
                label="📥 Download Operating Ranges (CSV)",
                data=csv,
                file_name=f"brix_operating_ranges_{target_brix}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        else:
            st.warning(f"⚠️ Not enough data found near Brix = {target_brix}±{tolerance}")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🎯 Brix Control Assistant | Powered by XGBoost ML | Version 2.0 | Operator-Friendly Interface
</div>
""", unsafe_allow_html=True)
