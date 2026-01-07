import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import StringIO
# FIX: calibration_curve is in sklearn.calibration, not sklearn.metrics
from sklearn.calibration import calibration_curve

# Page Configuration
st.set_page_config(
    page_title="DBS Pipeline Visualizer",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #262730;
    }
    h1, h2, h3 {
        color: #4DB6AC;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

def generate_brain_phantom(n_points=2000):
    """Generates synthetic data for demo purposes."""
    # Right hemisphere
    u = np.random.uniform(0, np.pi, n_points // 2)
    v = np.random.uniform(0, 2 * np.pi, n_points // 2)
    x1 = 10 * np.sin(u) * np.cos(v) + 5
    y1 = 8 * np.sin(u) * np.sin(v)
    z1 = 8 * np.cos(u)
    
    # Left hemisphere
    x2 = 10 * np.sin(u) * np.cos(v) - 5
    y2 = 8 * np.sin(u) * np.sin(v)
    z2 = 8 * np.cos(u)
    
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    z = np.concatenate([z1, z2])
    
    # Simulate e-field (stronger in center)
    dist_to_center = np.sqrt(x**2 + y**2 + z**2)
    e_field = np.exp(-dist_to_center / 10)
    
    # Simulate anatomy labels (0: STN, 1: GPi, etc)
    anatomy = (y > 0).astype(int)
    
    return np.column_stack((x, y, z)), {"e_field": e_field, "anatomy": anatomy}

def load_data(uploaded_file):
    """
    Parses uploaded files. 
    Supports specific .npz format from your DBS pipeline.
    Returns: points (N, 3), attributes (dict of arrays)
    """
    if uploaded_file is None:
        return None, None
        
    filename = uploaded_file.name
    attributes = {}
    
    try:
        if filename.endswith('.npz'):
            data = np.load(uploaded_file)
            
            # 1. Handle DBS Pipeline Format ('coords', 'e_field', etc.)
            if 'coords' in data:
                points = data['coords']
                if 'e_field' in data:
                    attributes['Electric Field (V/m)'] = data['e_field'].flatten()
                if 'anatomy' in data:
                    attributes['Anatomy Label'] = data['anatomy'].flatten()
                return points, attributes
                
            # 2. Handle Processed PointNet++ Format ('points' N,8)
            elif 'points' in data:
                raw_points = data['points'] # Shape (N, 8) or (N, C)
                points = raw_points[:, :3]
                # Assuming standard format from your script: [x,y,z, anat, e_field, vec...]
                if raw_points.shape[1] >= 4:
                    attributes['Anatomy Label'] = raw_points[:, 3]
                if raw_points.shape[1] >= 5:
                    attributes['Electric Field (V/m)'] = raw_points[:, 4]
                return points, attributes
                
            # Fallback for generic .npz
            else:
                st.warning("Unknown .npz structure. Keys found: " + str(list(data.keys())))
                return None, None

        elif filename.endswith('.npy'):
            data = np.load(uploaded_file)
            if data.shape[1] != 3 and data.shape[0] == 3:
                data = data.T
            return data[:, :3], {}
            
        elif filename.endswith('.xyz') or filename.endswith('.txt'):
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio, sep=" ", header=None, comment='#')
            return df.iloc[:, :3].values, {}
            
        elif filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            cols = [c.lower() for c in df.columns]
            if 'x' in cols and 'y' in cols and 'z' in cols:
                return df[['x', 'y', 'z']].values, {}
            elif 'X' in df.columns and 'Y' in df.columns and 'Z' in df.columns:
                return df[['X', 'Y', 'Z']].values, {}
            else:
                return df.iloc[:, :3].values, {}
                
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

# --- Plotting Functions ---

def plot_calibration_curve(y_true, y_prob):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig = go.Figure()
    # Perfectly calibrated line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated',
                             line=dict(dash='dash', color='gray')))
    # Model calibration
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Model (PointNet++)',
                             line=dict(color='#4DB6AC', width=3)))
    
    fig.update_layout(
        title="Calibration Curve (TRIPOD-AI)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template="plotly_dark",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig

def plot_dca(y_true, y_prob, thresholds=np.arange(0.01, 1.0, 0.01)):
    n = len(y_true)
    net_benefits_model = []
    net_benefits_all = []
    
    for thresh in thresholds:
        # Model
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits_model.append(max(0, net_benefit))
        
        # Treat All
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        net_benefit_all = (tp_all / n) - (fp_all / n) * (thresh / (1 - thresh))
        net_benefits_all.append(max(0, net_benefit_all))
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=net_benefits_model, mode='lines', name='Model (PointNet++)',
                             line=dict(color='#4DB6AC', width=3)))
    fig.add_trace(go.Scatter(x=thresholds, y=net_benefits_all, mode='lines', name='Treat All',
                             line=dict(dash='dot', color='orange')))
    fig.add_trace(go.Scatter(x=thresholds, y=[0]*len(thresholds), mode='lines', name='Treat None',
                             line=dict(color='white')))
                             
    fig.update_layout(
        title="Decision Curve Analysis (DCA)",
        xaxis_title="Threshold Probability",
        yaxis_title="Net Benefit",
        template="plotly_dark",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[-0.05, max(max(net_benefits_model), max(net_benefits_all)) + 0.1])
    )
    return fig

# --- Main App Layout ---

st.title("🧠 DBS Integrated Pipeline Visualizer")
st.markdown("Visualize both the **3D Data Processing** (Lead-DBS/Point Clouds) and **Clinical Validation** (TRIPOD-AI metrics).")

tab1, tab2 = st.tabs(["🧬 Process 1: 3D Point Cloud", "📈 Process 2: Validation (TRIPOD-AI)"])

# === TAB 1: 3D Visualization ===
with tab1:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Data Input")
        uploaded_file = st.file_uploader("Upload Pipeline Output (.npz)", type=["npz", "npy", "xyz", "csv"])
        
        points, attributes = None, None
        if uploaded_file:
            points, attributes = load_data(uploaded_file)
            if points is not None:
                st.success(f"Loaded {len(points)} points")
        else:
            if st.button("Load Demo Data"):
                points, attributes = generate_brain_phantom()
                
        # Style Controls
        st.subheader("Style")
        point_size = st.slider("Point Size", 1, 10, 2)
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8)
        
        # Color Selection
        color_option = "Depth (Z)"
        if attributes:
            color_options = ["Depth (Z)"] + list(attributes.keys())
            color_option = st.selectbox("Color By", color_options)
        
        colorscale = st.selectbox("Color Scale", ["Viridis", "Plasma", "Jet", "RdBu", "Magma"])

    with col2:
        if points is not None:
            # Determine Color
            if color_option == "Depth (Z)":
                color_values = points[:, 2]
                cbar_title = "Depth (Z)"
            else:
                color_values = attributes[color_option]
                cbar_title = color_option

            # Subsampling
            if len(points) > 20000:
                st.info(f"Subsampling view to 20k points (Original: {len(points)})")
                idx = np.random.choice(len(points), 20000, replace=False)
                plot_points = points[idx]
                plot_colors = color_values[idx]
            else:
                plot_points = points
                plot_colors = color_values

            fig = go.Figure(data=[go.Scatter3d(
                x=plot_points[:, 0], y=plot_points[:, 1], z=plot_points[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=plot_colors,
                    colorscale=colorscale,
                    opacity=opacity,
                    colorbar=dict(title=cbar_title)
                )
            )])

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis=dict(backgroundcolor="#0e1117", gridcolor='#444'),
                    yaxis=dict(backgroundcolor="#0e1117", gridcolor='#444'),
                    zaxis=dict(backgroundcolor="#0e1117", gridcolor='#444'),
                    aspectmode='data'
                ),
                paper_bgcolor="#0e1117",
                height=650
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metadata display
            if uploaded_file and attributes:
                st.caption("Attributes found in file: " + ", ".join(attributes.keys()))

# === TAB 2: Validation Visualization ===
with tab2:
    st.header("TRIPOD-AI Clinical Validation")
    st.markdown("Visualize model performance metrics described in `dbs_external_validation.py`.")
    
    # Input simulation data
    col_v1, col_v2 = st.columns([1, 3])
    
    with col_v1:
        st.subheader("Simulation Params")
        n_samples = st.slider("Test Samples", 100, 1000, 500)
        auc_sim = st.slider("Simulated AUC", 0.5, 0.99, 0.85)
        calibration_err = st.slider("Calibration Error", 0.0, 0.5, 0.1)
        
        st.info("💡 Adjust these sliders to simulate how different model performances would look in the validation report.")

    with col_v2:
        # Generate dummy data for visualization
        np.random.seed(42)
        y_true = np.random.randint(0, 2, n_samples)
        noise = np.random.normal(0, (1-auc_sim)*2, n_samples)
        # Create probabilities correlated with truth but with noise
        y_logits = y_true * 2.0 - 1.0 + noise
        y_prob = 1 / (1 + np.exp(-y_logits))
        
        # Add calibration error
        if calibration_err > 0:
            y_prob = np.power(y_prob, 1.0 - calibration_err) # Skew probabilities

        # 1. Calibration Curve
        st.subheader("1. Calibration Curve")
        st.markdown("Assesses the agreement between predicted probabilities and observed outcomes.")
        fig_cal = plot_calibration_curve(y_true, y_prob)
        st.plotly_chart(fig_cal, use_container_width=True)
        
        # 2. Decision Curve Analysis
        st.subheader("2. Decision Curve Analysis (DCA)")
        st.markdown("Evaluates the clinical utility of the model across different decision thresholds.")
        fig_dca = plot_dca(y_true, y_prob)
        st.plotly_chart(fig_dca, use_container_width=True)