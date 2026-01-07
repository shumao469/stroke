import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# -----------------------------------------------------------------------------
# 1. 页面配置与样式 / Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="生物标志物分析仪表板 / Biofeature Analysis Dashboard",
    page_icon="🧬",
    layout="wide"
)

# 自定义CSS样式 / Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.0rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #17A589;
        border-bottom: 2px solid #17A589;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #F8F9F9;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .step-header {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2E86C1;
        margin-top: 20px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 图像处理与特征提取功能 / Image Processing & Feature Extraction
# -----------------------------------------------------------------------------

def extract_features_from_image(uploaded_file):
    """
    模拟从图像中提取生物特征。
    Simulate feature extraction from images.
    """
    try:
        image = Image.open(uploaded_file).convert('L') # Convert to grayscale
        img_array = np.array(image)
        
        # 1. 基础物理特征 / Basic Physical Features
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # 简单阈值分割计算阳性面积 (假设阈值为 50) / Simple thresholding for positive area
        threshold_area = np.sum(img_array > 50) / img_array.size * 100
        
        # 2. 映射到特定生物指标 / Mapping to Bio-features
        # 这里的逻辑是模拟演示 / This logic is for demonstration
        
        features = {
            'FileName': uploaded_file.name,
            # Intensity Metrics
            'iNOS_Intensity': mean_intensity * 1.2 + np.random.normal(0, 5),
            'Arg1_Intensity': mean_intensity * 0.8 + np.random.normal(0, 5),
            'Claudin5_Mean_Int': mean_intensity + np.random.normal(0, 2),
            'GAP43_Intensity': mean_intensity * 0.5 + 20,
            
            # Area/Density Metrics
            'Iba1_Area_Pct': threshold_area,
            'CD31_Vessel_Density': threshold_area * 0.5 + np.random.normal(0, 1),
            
            # Count Metrics
            'Synapse_Puncta_Count': int(mean_intensity * 5 + np.random.randint(50, 100)),
            'TUNEL_Positive_Cells': int(threshold_area * 2 + np.random.randint(0, 5))
        }
        
        return features
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None

def parse_group_from_filename(filename):
    """尝试从文件名自动推断分组 / Try to infer group from filename"""
    filename_lower = filename.lower()
    if 'control' in filename_lower or 'con' in filename_lower or 'ctr' in filename_lower:
        return 'Control'
    elif 'model' in filename_lower or 'stroke' in filename_lower:
        return 'Model'
    elif 'treat' in filename_lower or 'drug' in filename_lower:
        return 'Treatment'
    else:
        return 'Unknown' # 待用户手动指定 / To be specified by user

# -----------------------------------------------------------------------------
# 3. 模拟数据生成 / Mock Data Generation
# -----------------------------------------------------------------------------

@st.cache_data
def generate_mock_data():
    """生成模拟数据 / Generate Mock Data"""
    np.random.seed(42)
    n_samples = 60
    groups = np.random.choice(['Control', 'Model', 'Treatment'], n_samples)
    regions = np.random.choice(['Cortex', 'Striatum'], n_samples)
    
    data = pd.DataFrame({
        'Sample_ID': [f'S_{i:03d}' for i in range(n_samples)],
        'Group': groups,
        'Region': regions
    })
    
    def generate_value(group, base, var_high, var_low):
        noise = np.random.normal(0, base * 0.1)
        if group == 'Control': return base + noise
        elif group == 'Model': return var_high + noise
        elif group == 'Treatment': return var_low + noise
        return base
    
    data['iNOS_Intensity'] = data['Group'].apply(lambda x: generate_value(x, 100, 300, 180))
    data['Arg1_Intensity'] = data['Group'].apply(lambda x: generate_value(x, 100, 50, 250))
    data['Iba1_Area_Pct'] = data['Group'].apply(lambda x: generate_value(x, 2.5, 15.0, 8.0))
    data['Claudin5_Mean_Int'] = data['Group'].apply(lambda x: generate_value(x, 200, 50, 150))
    data['CD31_Vessel_Density'] = data['Group'].apply(lambda x: generate_value(x, 10, 8, 18))
    data['Synapse_Puncta_Count'] = data['Group'].apply(lambda x: generate_value(x, 500, 200, 400))
    data['GAP43_Intensity'] = data['Group'].apply(lambda x: generate_value(x, 50, 80, 200))
    data['TUNEL_Positive_Cells'] = data['Group'].apply(lambda x: generate_value(x, 5, 80, 30))
    
    return data

# -----------------------------------------------------------------------------
# 4. 通用绘图函数 / Plotting Functions
# -----------------------------------------------------------------------------

def plot_box_and_scatter(df, x_col, y_col, color_col, title, y_label):
    fig = px.box(df, x=x_col, y=y_col, color=color_col, points="all",
                 title=title,
                 color_discrete_map={'Control': 'gray', 'Model': '#E74C3C', 'Treatment': '#2ECC71'})
    fig.update_layout(yaxis_title=y_label, xaxis_title="Group (分组)", template="plotly_white")
    return fig

# -----------------------------------------------------------------------------
# 5. 主程序逻辑 / Main Logic
# -----------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">🧠 免疫荧光分析平台 <br> Immunofluorescence Analysis Platform</div>', unsafe_allow_html=True)
    
    # --- Sidebar: Workflow Selection ---
    st.sidebar.header("🛠️ 工作流设置 / Workflow Settings")
    
    # Bilingual options
    data_mode = st.sidebar.radio(
        "选择数据来源 / Select Data Source", 
        [
            "1. 图像特征提取 / Image Feature Extraction", 
            "2. 上传已量化数据 / Upload Quantified CSV", 
            "3. 演示数据 / Demo Data"
        ]
    )
    
    df = None # 初始化 DataFrame
    
    # --- Workflow 1: Image Processing ---
    if data_mode == "1. 图像特征提取 / Image Feature Extraction":
        st.markdown('<div class="step-header">步骤 1: 上传图片 / Step 1: Upload Images</div>', unsafe_allow_html=True)
        st.info("支持 TIF, PNG, JPG 格式。系统将计算荧光强度和面积占比。\n\nSupports TIF, PNG, JPG. Calculates intensity and area percentage.")
        
        uploaded_images = st.file_uploader("拖拽或点击上传 / Drag and Drop", type=['tif', 'tiff', 'png', 'jpg'], accept_multiple_files=True)
        
        if uploaded_images:
            if st.button(f"开始分析 {len(uploaded_images)} 张图片 / Analyze {len(uploaded_images)} Images"):
                with st.spinner('正在提取特征... / Extracting features...'):
                    # 提取特征
                    extracted_data = []
                    progress_bar = st.progress(0)
                    
                    for idx, img_file in enumerate(uploaded_images):
                        features = extract_features_from_image(img_file)
                        if features:
                            # 自动推断分组
                            features['Group'] = parse_group_from_filename(img_file.name)
                            features['Region'] = 'Cortex' # 默认值
                            extracted_data.append(features)
                        progress_bar.progress((idx + 1) / len(uploaded_images))
                    
                    if extracted_data:
                        df = pd.DataFrame(extracted_data)
                        st.session_state['extracted_df'] = df
                        st.success("特征提取完成！ / Extraction Complete!")
            
            # 检查 Session State
            if 'extracted_df' in st.session_state:
                df_raw = st.session_state['extracted_df']
                
                st.markdown('<div class="step-header">步骤 2: 确认分组信息 / Step 2: Verify Grouping</div>', unsafe_allow_html=True)
                st.warning("请检查并修改 'Group' (分组) 和 'Region' (脑区)。\n\nPlease verify and edit 'Group' and 'Region'.")
                
                # Bilingual column config
                df = st.data_editor(
                    df_raw,
                    column_config={
                        "Group": st.column_config.SelectboxColumn(
                            "Group (分组)",
                            help="Select experiment group / 选择实验分组",
                            width="medium",
                            options=["Control", "Model", "Treatment", "Sham"],
                            required=True,
                        ),
                        "Region": st.column_config.SelectboxColumn(
                            "Region (脑区)",
                            width="medium",
                            options=["Cortex", "Striatum", "Hippocampus", "Thalamus"],
                            required=True,
                        )
                    },
                    hide_index=True,
                )
                
                st.markdown("---")
                st.caption(f"当前样本数 / Current Samples: {len(df)}")

    # --- Workflow 2: CSV Upload ---
    elif data_mode == "2. 上传已量化数据 / Upload Quantified CSV":
        st.markdown('<div class="step-header">导入数据 / Import Data</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("上传 CSV/Excel 文件 / Upload File", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success("数据加载成功！ / Data Loaded Successfully!")
            except Exception as e:
                st.error(f"文件错误 / File Error: {e}")

    # --- Workflow 3: Demo Data ---
    else:
        df = generate_mock_data()
        st.sidebar.success("已加载模拟数据 / Mock Data Loaded")

    # --- Analysis Logic ---
    if df is not None and not df.empty:
        
        st.sidebar.markdown("---")
        st.sidebar.header("📊 结果可视化 / Visualization")
        
        # Bilingual Navigation
        analysis_section = st.sidebar.radio(
            "选择板块 / Select Section:",
            [
                "Home: 数据概览 / Overview",
                "Section 1: 小胶质细胞极化 / Microglia Polarization (iNOS/Arg1)",
                "Section 2: 小胶质细胞激活 / Microglia Activation (Iba1)",
                "Section 3: 血脑屏障完整性 / BBB Integrity (Claudin-5)",
                "Section 4: 血管生成 / Angiogenesis (CD31)",
                "Section 5: 突触可塑性 / Synaptic Plasticity",
                "Section 6: 轴突再生 / Axonal Regeneration (GAP43)",
                "Section 7: 细胞凋亡 / Apoptosis (TUNEL)"
            ]
        )
        
        # 筛选器
        if 'Region' in df.columns:
            selected_region = st.sidebar.selectbox("筛选脑区 / Filter Region", ["All"] + list(df['Region'].unique()))
            df_filtered = df[df['Region'] == selected_region] if selected_region != "All" else df
        else:
            df_filtered = df

        # --- Home ---
        if "Home" in analysis_section:
            st.subheader("📊 数据集概览 / Dataset Overview")
            st.dataframe(df_filtered.head())
            
            col1, col2 = st.columns(2)
            with col1:
                if 'Group' in df_filtered.columns:
                    st.caption("分组分布 / Group Distribution")
                    st.plotly_chart(px.pie(df_filtered, names='Group', hole=0.4), use_container_width=True)
            with col2:
                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.caption("相关性热图 / Correlation Heatmap")
                    corr = df_filtered[numeric_cols].corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

        # --- Section 1: iNOS / Arg ---
        elif "Section 1" in analysis_section:
            st.markdown('<div class="section-header">Section 1: Microglia Polarization (iNOS/Arg1)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \niNOS (M1 Marker) vs Arg1 (M2 Marker). M1/M2 Ratio indicates polarization status.")
            
            if 'iNOS_Intensity' in df_filtered.columns and 'Arg1_Intensity' in df_filtered.columns:
                df_filtered['M1_M2_Ratio'] = df_filtered['iNOS_Intensity'] / (df_filtered['Arg1_Intensity'] + 0.1)
                
                tab1, tab2 = st.tabs(["独立表达 / Intensity", "极化比值 / M1/M2 Ratio"])
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'iNOS_Intensity', 'Group', 'iNOS Intensity (M1)', 'Mean Intensity'), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Arg1_Intensity', 'Group', 'Arg1 Intensity (M2)', 'Mean Intensity'), use_container_width=True)
                with tab2:
                    st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'M1_M2_Ratio', 'Group', 'Polarization Ratio (iNOS/Arg1)', 'Ratio'), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 iNOS 或 Arg1 数据 / Missing iNOS or Arg1 data.")

        # --- Section 2: Iba1 ---
        elif "Section 2" in analysis_section:
            st.markdown('<div class="section-header">Section 2: Microglia Activation (Iba1)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nReflects microglia activation levels (Area %).")
            if 'Iba1_Area_Pct' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Iba1_Area_Pct', 'Group', 'Iba1 Activation (Area %)', '% Area'), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 Iba1 数据 / Missing Iba1 data.")

        # --- Section 3: Claudin-5 ---
        elif "Section 3" in analysis_section:
            st.markdown('<div class="section-header">Section 3: BBB Integrity (Claudin-5)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nTight junction protein. Higher intensity indicates better BBB integrity.")
            if 'Claudin5_Mean_Int' in df_filtered.columns:
                st.plotly_chart(px.violin(df_filtered, x='Group', y='Claudin5_Mean_Int', color='Group', box=True, points="all", title="Claudin-5 Intensity Distribution"), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 Claudin-5 数据 / Missing Claudin-5 data.")

        # --- Section 4: CD31 ---
        elif "Section 4" in analysis_section:
            st.markdown('<div class="section-header">Section 4: Angiogenesis (CD31)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nEndothelial marker. Indicates vessel density.")
            if 'CD31_Vessel_Density' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'CD31_Vessel_Density', 'Group', 'Vessel Density (CD31)', 'Density'), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 CD31 数据 / Missing CD31 data.")

        # --- Section 5: Synapse ---
        elif "Section 5" in analysis_section:
            st.markdown('<div class="section-header">Section 5: Synaptic Plasticity</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nQuantification of synaptic puncta counts.")
            if 'Synapse_Puncta_Count' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Synapse_Puncta_Count', 'Group', 'Synapse Puncta Count', 'Count'), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 Synapse 数据 / Missing Synapse data.")

        # --- Section 6: GAP43 ---
        elif "Section 6" in analysis_section:
            st.markdown('<div class="section-header">Section 6: Axonal Regeneration (GAP43)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nMarker for neurite outgrowth and regeneration.")
            if 'GAP43_Intensity' in df_filtered.columns:
                st.plotly_chart(px.strip(df_filtered, x='Group', y='GAP43_Intensity', color='Group', title="GAP43 Intensity Distribution"), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 GAP43 数据 / Missing GAP43 data.")

        # --- Section 7: TUNEL ---
        elif "Section 7" in analysis_section:
            st.markdown('<div class="section-header">Section 7: Cell Apoptosis (TUNEL)</div>', unsafe_allow_html=True)
            st.info("💡 **分析说明 / Analysis Note**: \nMarker for apoptotic cells (DNA fragmentation).")
            if 'TUNEL_Positive_Cells' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'TUNEL_Positive_Cells', 'Group', 'Apoptotic Cells Count (TUNEL)', 'Count'), use_container_width=True)
            else:
                st.warning("⚠️ 缺少 TUNEL 数据 / Missing TUNEL data.")
    
    else:
        # Empty state prompts
        if "Image" in data_mode:
            st.info("👋 请先在左侧上传图片并点击“开始分析” / Please upload images and click 'Analyze' to start.")
        elif "Upload" in data_mode:
            st.info("👋 请上传 CSV 文件以开始 / Please upload a CSV file to start.")

if __name__ == "__main__":
    main()
