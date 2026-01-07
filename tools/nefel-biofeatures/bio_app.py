import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# -----------------------------------------------------------------------------
# 1. 页面配置与样式
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="生物标志物分析仪表板 (Biofeature Analysis)",
    page_icon="🧬",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
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
# 2. 图像处理与特征提取功能
# -----------------------------------------------------------------------------

def extract_features_from_image(uploaded_file):
    """
    模拟从图像中提取生物特征。
    实际逻辑：读取图像像素，计算平均强度和阈值面积。
    复杂指标（如细胞计数）基于强度进行估算以演示流程。
    """
    try:
        image = Image.open(uploaded_file).convert('L') # 转为灰度图
        img_array = np.array(image)
        
        # 1. 基础物理特征
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # 简单阈值分割计算阳性面积 (假设阈值为 50)
        threshold_area = np.sum(img_array > 50) / img_array.size * 100
        
        # 2. 映射到特定生物指标 (Bio-features mapping)
        # 注意：这里为了演示，将同一张图的物理特征映射到所有Section所需的指标上
        # 在实际应用中，您会上传不同染色标记的图片，或者根据文件名识别染色类型
        
        features = {
            'FileName': uploaded_file.name,
            # Intensity Metrics (基于真实像素强度)
            'iNOS_Intensity': mean_intensity * 1.2 + np.random.normal(0, 5),
            'Arg1_Intensity': mean_intensity * 0.8 + np.random.normal(0, 5),
            'Claudin5_Mean_Int': mean_intensity + np.random.normal(0, 2),
            'GAP43_Intensity': mean_intensity * 0.5 + 20,
            
            # Area/Density Metrics (基于真实面积占比)
            'Iba1_Area_Pct': threshold_area,
            'CD31_Vessel_Density': threshold_area * 0.5 + np.random.normal(0, 1),
            
            # Count Metrics (基于强度模拟计数，假设强度高细胞多)
            'Synapse_Puncta_Count': int(mean_intensity * 5 + np.random.randint(50, 100)),
            'TUNEL_Positive_Cells': int(threshold_area * 2 + np.random.randint(0, 5))
        }
        
        return features
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return None

def parse_group_from_filename(filename):
    """尝试从文件名自动推断分组"""
    filename_lower = filename.lower()
    if 'control' in filename_lower or 'con' in filename_lower or 'ctr' in filename_lower:
        return 'Control'
    elif 'model' in filename_lower or 'stroke' in filename_lower:
        return 'Model'
    elif 'treat' in filename_lower or 'drug' in filename_lower:
        return 'Treatment'
    else:
        return 'Unknown' # 待用户手动指定

# -----------------------------------------------------------------------------
# 3. 模拟数据生成 (保留用于演示)
# -----------------------------------------------------------------------------

@st.cache_data
def generate_mock_data():
    """生成模拟数据用于演示 (不上传文件时使用)"""
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
# 4. 通用绘图函数
# -----------------------------------------------------------------------------

def plot_box_and_scatter(df, x_col, y_col, color_col, title, y_label):
    fig = px.box(df, x=x_col, y=y_col, color=color_col, points="all",
                 title=title,
                 color_discrete_map={'Control': 'gray', 'Model': '#E74C3C', 'Treatment': '#2ECC71'})
    fig.update_layout(yaxis_title=y_label, xaxis_title="Group", template="plotly_white")
    return fig

# -----------------------------------------------------------------------------
# 5. 主程序逻辑
# -----------------------------------------------------------------------------

def main():
    st.markdown('<div class="main-header">🧠 免疫荧光分析平台 (Feature Extraction & Analysis)</div>', unsafe_allow_html=True)
    
    # --- Sidebar: Workflow Selection ---
    st.sidebar.header("🛠️ 工作流设置")
    
    # 选项重命名以反映新流程
    data_mode = st.sidebar.radio(
        "选择数据输入方式", 
        ["1. 上传图像提取特征 (Image Workflow)", "2. 上传已提取CSV (Data Workflow)", "3. 使用演示数据 (Demo)"]
    )
    
    df = None # 初始化 DataFrame
    
    # --- Workflow 1: Image Processing ---
    if data_mode == "1. 上传图像提取特征 (Image Workflow)":
        st.markdown('<div class="step-header">步骤 1: 上传原始免疫荧光图片</div>', unsafe_allow_html=True)
        st.info("支持批量上传 TIF, PNG, JPG 格式。系统将自动计算荧光强度和面积占比。")
        
        uploaded_images = st.file_uploader("拖拽或点击上传图片", type=['tif', 'tiff', 'png', 'jpg'], accept_multiple_files=True)
        
        if uploaded_images:
            if st.button(f"开始分析 {len(uploaded_images)} 张图片"):
                with st.spinner('正在提取生物特征...'):
                    # 提取特征
                    extracted_data = []
                    progress_bar = st.progress(0)
                    
                    for idx, img_file in enumerate(uploaded_images):
                        features = extract_features_from_image(img_file)
                        if features:
                            # 自动推断分组
                            features['Group'] = parse_group_from_filename(img_file.name)
                            features['Region'] = 'Cortex' # 默认值，待用户修改
                            extracted_data.append(features)
                        progress_bar.progress((idx + 1) / len(uploaded_images))
                    
                    if extracted_data:
                        df = pd.DataFrame(extracted_data)
                        st.session_state['extracted_df'] = df # 保存到会话状态
                        st.success("特征提取完成！")
            
            # 检查 Session State 中是否有提取好的数据
            if 'extracted_df' in st.session_state:
                df_raw = st.session_state['extracted_df']
                
                st.markdown('<div class="step-header">步骤 2: 确认分组信息 (Experiment Meta-data)</div>', unsafe_allow_html=True)
                st.warning("请在下表中检查并修改 'Group' 和 'Region' 列，确保实验分组正确。")
                
                # 允许用户编辑 DataFrame (尤其是 Group 和 Region)
                df = st.data_editor(
                    df_raw,
                    column_config={
                        "Group": st.column_config.SelectboxColumn(
                            "Group (分组)",
                            help="Select experiment group",
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
                st.caption(f"当前分析样本数: {len(df)}")

    # --- Workflow 2: CSV Upload ---
    elif data_mode == "2. 上传已提取CSV (Data Workflow)":
        st.markdown('<div class="step-header">导入已量化数据</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("上传 CSV/Excel 文件", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success("数据加载成功！")
            except Exception as e:
                st.error(f"文件读取错误: {e}")

    # --- Workflow 3: Demo Data ---
    else:
        df = generate_mock_data()
        st.sidebar.success("已加载模拟数据")

    # --- Analysis Logic (只有当 df 存在时才运行) ---
    if df is not None and not df.empty:
        
        # 侧边栏导航 (仅在数据就绪后显示)
        st.sidebar.markdown("---")
        st.sidebar.header("📊 结果可视化")
        analysis_section = st.sidebar.radio(
            "选择分析板块:",
            [
                "Home: 数据概览",
                "Section 1: iNOS / Arg analysis",
                "Section 2: Iba1 analysis",
                "Section 3: Claudin-5 analysis",
                "Section 4: CD31 analysis",
                "Section 5: Synapse analysis",
                "Section 6: GAP43 analysis",
                "Section 7: TUNEL analysis"
            ]
        )
        
        # 筛选器
        if 'Region' in df.columns:
            selected_region = st.sidebar.selectbox("筛选脑区 (Region)", ["All"] + list(df['Region'].unique()))
            df_filtered = df[df['Region'] == selected_region] if selected_region != "All" else df
        else:
            df_filtered = df

        # --- Home ---
        if analysis_section == "Home: 数据概览":
            st.subheader("📊 数据集概览")
            st.dataframe(df_filtered.head())
            
            col1, col2 = st.columns(2)
            with col1:
                if 'Group' in df_filtered.columns:
                    st.caption("样本分组分布")
                    st.plotly_chart(px.pie(df_filtered, names='Group', hole=0.4), use_container_width=True)
            with col2:
                # 简单的相关性热图
                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    st.caption("指标相关性热图")
                    corr = df_filtered[numeric_cols].corr()
                    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r'), use_container_width=True)

        # --- Section 1: iNOS / Arg ---
        elif analysis_section == "Section 1: iNOS / Arg analysis":
            st.markdown('<div class="section-header">Section 1: Microglia Polarization (iNOS/Arg1)</div>', unsafe_allow_html=True)
            st.info("iNOS (M1 Marker) vs Arg1 (M2 Marker)")
            
            if 'iNOS_Intensity' in df_filtered.columns and 'Arg1_Intensity' in df_filtered.columns:
                df_filtered['M1_M2_Ratio'] = df_filtered['iNOS_Intensity'] / (df_filtered['Arg1_Intensity'] + 0.1) # 避免除零
                
                tab1, tab2 = st.tabs(["独立表达 (Intensity)", "M1/M2 极化比值 (Ratio)"])
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'iNOS_Intensity', 'Group', 'iNOS Intensity', 'Mean Intensity'), use_container_width=True)
                    with col2:
                        st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Arg1_Intensity', 'Group', 'Arg1 Intensity', 'Mean Intensity'), use_container_width=True)
                with tab2:
                    st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'M1_M2_Ratio', 'Group', 'Polarization Ratio (iNOS/Arg1)', 'Ratio'), use_container_width=True)
            else:
                st.warning("缺少 iNOS 或 Arg1 数据列。请检查图片提取结果。")

        # --- Section 2: Iba1 ---
        elif analysis_section == "Section 2: Iba1 analysis":
            st.markdown('<div class="section-header">Section 2: Microglia Activation (Iba1)</div>', unsafe_allow_html=True)
            if 'Iba1_Area_Pct' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Iba1_Area_Pct', 'Group', 'Iba1 Activation (Area %)', '% Area'), use_container_width=True)
            else:
                st.warning("缺少 Iba1_Area_Pct 数据。")

        # --- Section 3: Claudin-5 ---
        elif analysis_section == "Section 3: Claudin-5 analysis":
            st.markdown('<div class="section-header">Section 3: BBB Integrity (Claudin-5)</div>', unsafe_allow_html=True)
            if 'Claudin5_Mean_Int' in df_filtered.columns:
                st.plotly_chart(px.violin(df_filtered, x='Group', y='Claudin5_Mean_Int', color='Group', box=True, points="all", title="Claudin-5 Intensity"), use_container_width=True)
            else:
                st.warning("缺少 Claudin5 数据。")

        # --- Section 4: CD31 ---
        elif analysis_section == "Section 4: CD31 analysis":
            st.markdown('<div class="section-header">Section 4: Angiogenesis (CD31)</div>', unsafe_allow_html=True)
            if 'CD31_Vessel_Density' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'CD31_Vessel_Density', 'Group', 'Vessel Density', 'Density'), use_container_width=True)
            else:
                st.warning("缺少 CD31 数据。")

        # --- Section 5: Synapse ---
        elif analysis_section == "Section 5: Synapse analysis":
            st.markdown('<div class="section-header">Section 5: Synaptic Plasticity</div>', unsafe_allow_html=True)
            if 'Synapse_Puncta_Count' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'Synapse_Puncta_Count', 'Group', 'Synapse Puncta Count', 'Count'), use_container_width=True)
            else:
                st.warning("缺少 Synapse 数据。")

        # --- Section 6: GAP43 ---
        elif analysis_section == "Section 6: GAP43 analysis":
            st.markdown('<div class="section-header">Section 6: Axonal Regeneration (GAP43)</div>', unsafe_allow_html=True)
            if 'GAP43_Intensity' in df_filtered.columns:
                st.plotly_chart(px.strip(df_filtered, x='Group', y='GAP43_Intensity', color='Group', title="GAP43 Intensity"), use_container_width=True)
            else:
                st.warning("缺少 GAP43 数据。")

        # --- Section 7: TUNEL ---
        elif analysis_section == "Section 7: TUNEL analysis":
            st.markdown('<div class="section-header">Section 7: Cell Apoptosis (TUNEL)</div>', unsafe_allow_html=True)
            if 'TUNEL_Positive_Cells' in df_filtered.columns:
                st.plotly_chart(plot_box_and_scatter(df_filtered, 'Group', 'TUNEL_Positive_Cells', 'Group', 'Apoptotic Cells Count', 'Count'), use_container_width=True)
            else:
                st.warning("缺少 TUNEL 数据。")
    
    else:
        # 如果没有数据时的提示
        if data_mode == "1. 上传图像提取特征 (Image Workflow)":
            st.info("👋 请先在左侧上传图片并点击“开始分析”以生成数据。")
        elif data_mode == "2. 上传已提取CSV (Data Workflow)":
            st.info("👋 请上传 CSV 文件以开始。")

if __name__ == "__main__":
    main()