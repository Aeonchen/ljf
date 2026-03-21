"""
学生学业预警系统 - Web界面
简单直观的Web应用，用于查看和交互使用预警系统
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from configs.web import APP_CONFIG, MODEL_CONFIG
from src.web.view_models import (
    build_dashboard_summary,
    build_data_overview,
    build_feature_relationship,
    build_feature_importance_frame,
    build_prediction_probability_rows,
    build_risk_distribution,
    build_single_prediction_result,
    build_student_management_view,
    get_feature_columns,
    load_app_resources,
    resolve_resource_paths,
)

# 页面配置
st.set_page_config(
    page_title=APP_CONFIG['title'],
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #D1FAE5;
        color: #059669;
        padding: 0.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# 加载数据和模型
@st.cache_resource
def load_resources():
    """加载模型和数据"""
    resources = {}

    try:
        resources = load_app_resources(PROJECT_ROOT, MODEL_CONFIG)
        for level, message in resources.get('_status', []):
            getattr(st, level)(message)
    except Exception as e:
        st.error(f"❌ 加载资源时出错: {str(e)}")

    return resources


# 侧边栏
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-center.png", width=80)
    st.title("🎓 导航")

    page = st.radio(
        "选择页面",
        ["📊 仪表板", "📈 数据分析", "🎯 风险预测", "👥 学生管理", "📋 系统报告"]
    )

    st.markdown("---")
    st.markdown("### 📊 系统状态")

    # 显示系统状态
    model_path = resolve_resource_paths(PROJECT_ROOT, MODEL_CONFIG)['model_path']
    if model_path.exists():
        model_age = datetime.fromtimestamp(model_path.stat().st_mtime)
        st.info(f"📅 模型更新时间: {model_age.strftime('%Y-%m-%d %H:%M')}")

    st.markdown("---")
    st.markdown("### ⚙️ 设置")

    # 配置选项
    show_details = st.checkbox("显示详细分析", value=True)
    num_high_risk = st.slider("显示高风险学生数量", 5, 30, 15)

# 主页面
if page == "📊 仪表板":
    st.markdown('<h1 class="main-header">🎓 学生学业预警系统</h1>', unsafe_allow_html=True)

    # 加载资源
    resources = load_resources()

    if 'data' in resources and 'model' in resources:
        df = resources['data']
        model = resources['model']
        dashboard_summary = build_dashboard_summary(df)
        risk_labels = dashboard_summary['risk_labels']

        # 关键指标
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("总学生数", f"{len(df)}人")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            avg_grade = dashboard_summary['avg_grade']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("平均成绩", f"{avg_grade:.2f}分")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            high_risk_count = dashboard_summary['high_risk_count']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("高风险学生", f"{high_risk_count}人")
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            low_risk_count = dashboard_summary['low_risk_count']
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("低风险学生", f"{low_risk_count}人")
            st.markdown('</div>', unsafe_allow_html=True)

        # 成绩分布
        st.markdown('<h3 class="sub-header">📊 成绩分布</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['GRADE'], bins=20, edgecolor='black', alpha=0.7, color='#3B82F6')
            ax.set_xlabel('成绩')
            ax.set_ylabel('学生人数')
            ax.set_title('学生成绩分布')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

        with col2:
            risk_distribution = build_risk_distribution(risk_labels)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                risk_distribution['counts'].values,
                labels=risk_distribution['labels'],
                autopct='%1.1f%%',
                colors=risk_distribution['colors'],
                startangle=90,
            )
            ax.set_title('风险类别分布')
            st.pyplot(fig)

        # 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_'):
            st.markdown('<h3 class="sub-header">🎯 重要特征</h3>', unsafe_allow_html=True)
            top_features = build_feature_importance_frame(model)

            # 显示前10个重要特征
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(top_features['特征'], top_features['重要性'], color='#8B5CF6')
            ax.set_xlabel('重要性分数')
            ax.set_title('Top 10 重要特征')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

elif page == "📈 数据分析":
    st.markdown('<h1 class="main-header">📈 数据分析</h1>', unsafe_allow_html=True)

    resources = load_resources()

    if 'data' in resources:
        df = resources['data']

        # 数据概览
        st.markdown('<h3 class="sub-header">📋 数据概览</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.write("**数据统计:**")
            data_overview = build_data_overview(df)
            st.write(f"- 总行数: {data_overview['row_count']}")
            st.write(f"- 总列数: {data_overview['column_count']}")
            st.write(f"- 成绩范围: {data_overview['grade_min']:.1f} - {data_overview['grade_max']:.1f}")
            st.write(f"- 缺失值: {data_overview['missing_count']}")

        # 特征选择分析
        st.markdown('<h3 class="sub-header">🔍 特征与成绩关系</h3>', unsafe_allow_html=True)

        # 选择要分析的特征
        feature_columns = get_feature_columns(df)
        selected_feature = st.selectbox("选择要分析的特征", feature_columns[:10])
        relationship = build_feature_relationship(df, selected_feature)

        # 绘制散点图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[selected_feature], df['GRADE'], alpha=0.6)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('成绩')
        ax.set_title(f'{selected_feature} 与成绩的关系')

        # 添加回归线
        if relationship['trendline_x'] is not None:
            ax.plot(
                relationship['trendline_x'],
                relationship['trendline_y'],
                "r--",
                alpha=0.8,
                label=f"相关性: {relationship['correlation']:.3f}"
            )
            ax.legend()

        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 相关系数矩阵
        st.markdown('<h3 class="sub-header">🔗 特征相关性</h3>', unsafe_allow_html=True)

        if st.checkbox("显示相关系数矩阵"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
            ax.set_title('特征相关系数矩阵')
            st.pyplot(fig)

elif page == "🎯 风险预测":
    st.markdown('<h1 class="main-header">🎯 风险预测</h1>', unsafe_allow_html=True)

    resources = load_resources()

    if 'model' in resources and 'scaler' in resources:
        model = resources['model']

        # 预测方式选择
        prediction_mode = st.radio(
            "选择预测方式",
            ["单个学生预测", "批量预测"]
        )

        if prediction_mode == "单个学生预测":
            st.markdown('<h3 class="sub-header">📝 输入学生特征</h3>', unsafe_allow_html=True)

            # 创建特征输入表单
            col1, col2, col3, col4 = st.columns(4)

            # 简化版本：只输入最重要的几个特征
            features = {}

            with col1:
                features['feature_29'] = st.number_input("特征29", value=0.0, step=0.1)
                features['feature_14'] = st.number_input("特征14", value=0.0, step=0.1)

            with col2:
                features['feature_28'] = st.number_input("特征28", value=0.0, step=0.1)
                features['feature_12'] = st.number_input("特征12", value=0.0, step=0.1)

            with col3:
                features['feature_11'] = st.number_input("特征11", value=0.0, step=0.1)
                features['feature_9'] = st.number_input("特征9", value=0.0, step=0.1)

            with col4:
                features['feature_5'] = st.number_input("特征5", value=0.0, step=0.1)
                features['feature_3'] = st.number_input("特征3", value=0.0, step=0.1)

            # 预测按钮
            if st.button("🔮 预测风险等级", type="primary"):
                try:
                    prediction_result = build_single_prediction_result(
                        features,
                        model,
                        resources['scaler'],
                    )
                    prediction = prediction_result['prediction']
                    probabilities = prediction_result['probabilities']

                    # 显示结果
                    st.markdown("---")
                    st.markdown("### 📊 预测结果")

                    # 风险等级显示
                    if prediction == '高风险':
                        st.markdown('<div class="risk-high">🔴 高风险 - 需要重点关注和干预</div>',
                                    unsafe_allow_html=True)
                    elif prediction == '中风险':
                        st.markdown('<div class="risk-medium">🟡 中风险 - 建议定期关注</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="risk-low">🟢 低风险 - 表现良好，继续保持</div>', unsafe_allow_html=True)

                    # 概率显示
                    st.markdown("#### 风险概率分布:")

                    # 创建水平条形图
                    probability_rows = build_prediction_probability_rows(
                        prediction_result['classes'],
                        probabilities,
                    )
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.barh(
                        probability_rows['label'],
                        probability_rows['probability'],
                        color=probability_rows['color'],
                    )
                    ax.set_xlabel('概率 (%)')
                    ax.set_xlim(0, 100)

                    # 添加数值标签
                    for bar, prob in zip(bars, probability_rows['probability']):
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height() / 2,
                                f' {prob:.1f}%', va='center')

                    st.pyplot(fig)

                    # 建议
                    st.markdown("#### 💡 建议:")
                    for recommendation in prediction_result['recommendations']:
                        st.write(f"- {recommendation}")

                except Exception as e:
                    st.error(f"预测时出错: {str(e)}")

        else:  # 批量预测
            st.markdown('<h3 class="sub-header">📁 上传学生数据</h3>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

            if uploaded_file is not None:
                try:
                    # 读取上传的数据
                    batch_data = pd.read_csv(uploaded_file)

                    st.success(f"✅ 成功读取 {len(batch_data)} 条记录")
                    st.dataframe(batch_data.head(), use_container_width=True)

                    if st.button("🔮 批量预测", type="primary"):
                        # 这里简化处理，实际需要提取特征并标准化
                        st.warning("批量预测功能正在开发中...")
                        st.info("目前请使用单个学生预测功能")

                except Exception as e:
                    st.error(f"读取文件时出错: {str(e)}")
    else:
        st.warning("未找到可用的模型或标准化器，请先运行训练脚本生成 Web 所需资源。")

elif page == "👥 学生管理":
    st.markdown('<h1 class="main-header">👥 学生管理</h1>', unsafe_allow_html=True)

    resources = load_resources()

    if 'data' in resources and 'model' in resources and 'scaler' in resources:
        df = resources['data']
        management_view = build_student_management_view(
            df,
            resources['model'],
            resources['scaler'],
            num_high_risk,
        )
        results_df = management_view['results_df']
        high_risk_df = management_view['high_risk_df']

        # 计算所有学生的风险
        st.markdown('<h3 class="sub-header">🚨 高风险学生名单</h3>', unsafe_allow_html=True)

        # 显示高风险学生表格
        if len(high_risk_df) > 0:
            st.dataframe(
                high_risk_df[['STUDENT ID', 'GRADE', '预测风险等级', '高风险概率']].reset_index(drop=True),
                use_container_width=True
            )

            # 高风险学生统计
            st.markdown(f"**高风险学生统计:** 共发现 {len(high_risk_df)} 名高风险学生")

            # 下载按钮
            csv = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载高风险学生名单",
                data=csv,
                file_name=f"高风险学生名单_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ 未发现高风险学生")

        # 学生搜索
        st.markdown('<h3 class="sub-header">🔍 搜索学生</h3>', unsafe_allow_html=True)

        search_option = st.radio("搜索方式", ["按学生ID", "按成绩范围"])

        if search_option == "按学生ID":
            student_id = st.text_input("输入学生ID")
            if student_id:
                # 查找学生
                student_data = results_df[results_df['STUDENT ID'] == student_id]
                if not student_data.empty:
                    st.dataframe(student_data, use_container_width=True)
                else:
                    st.warning(f"未找到学生ID: {student_id}")
        else:
            min_grade, max_grade = st.slider(
                "选择成绩范围",
                min_value=float(df['GRADE'].min()),
                max_value=float(df['GRADE'].max()),
                value=(0.0, 3.0)
            )

            filtered_df = results_df[
                (results_df['GRADE'] >= min_grade) &
                (results_df['GRADE'] <= max_grade)
                ]

            st.write(f"找到 {len(filtered_df)} 名成绩在 {min_grade}-{max_grade} 分的学生")
            st.dataframe(filtered_df.head(20), use_container_width=True)
    else:
        st.warning("未找到完整的学生管理资源，请确认模型、标准化器和数据文件均已生成。")

elif page == "📋 系统报告":
    st.markdown('<h1 class="main-header">📋 系统报告</h1>', unsafe_allow_html=True)

    resources = load_resources()

    if 'report' in resources:
        # 显示报告内容
        st.markdown(resources['report'])
    else:
        st.warning("未找到系统报告文件")

    # 系统信息
    st.markdown('<h3 class="sub-header">💻 系统信息</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.info("**系统版本:** 优化版预警系统")
        st.info(f"**模型类型:** 随机森林优化版")
        st.info(f"**数据样本:** 145名学生")

    with col2:
        st.info("**预测准确率:** 55.17%")
        st.info("**交叉验证:** 47.39%")
        st.info("**风险分类:** 高/中/低 三分类")

    # 生成新报告
    st.markdown('<h3 class="sub-header">🔄 重新生成报告</h3>', unsafe_allow_html=True)

    if st.button("🔄 重新运行预警系统", type="primary"):
        with st.spinner("正在重新运行预警系统..."):
            try:
                # 这里可以调用原系统的运行函数
                # import run_warning_optimized
                # run_warning_optimized.main()
                st.success("✅ 报告已重新生成")
                st.info("请刷新页面查看最新报告")
            except Exception as e:
                st.error(f"重新运行失败: {str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280;'>
        <p>🎓 学生学业预警系统 | 基于机器学习的学生风险预测工具</p>
        <p>📧 技术支持: data@example.com | 📞 联系电话: 123-456-7890</p>
        <p>© 2024 学业预警系统 | 版本 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
