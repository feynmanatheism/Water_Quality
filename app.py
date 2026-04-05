import importlib
import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, average_precision_score, precision_recall_curve)
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Water Quality Prediction", layout="wide")

PROJECT_TITLE = "Dự đoán nước an toàn dựa trên các chỉ số lý hóa bằng thuật toán RandomForest nhằm tối ưu hóa quy trình kiểm định chất lượng nước tại nguồn"
PROJECT_AUTHOR = "Hồ Trọng Nghĩa. - 22T1020683"
MODELS_DIR = Path("models")
DATA_FILE = Path("data/water_potability.csv")
FEATURE_COLUMNS = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
TARGET_COLUMN = 'Potability'


def patch_legacy_numpy_pickle():
    """Patch numpy module references to support loading pickles from older numpy versions."""
    try:
        import numpy as np
        
        # Alias legacy numpy._core to numpy.core
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = importlib.import_module('numpy.core')
        if 'numpy._core.multiarray' not in sys.modules:
            sys.modules['numpy._core.multiarray'] = importlib.import_module('numpy.core.multiarray')
        if 'numpy._core._multiarray_umath' not in sys.modules:
            sys.modules['numpy._core._multiarray_umath'] = importlib.import_module('numpy.core._multiarray_umath')

        # Ensure numpy.random._mt19937 is importable as a fallback
        try:
            sys.modules['numpy.random._mt19937']
        except (KeyError, AttributeError):
            try:
                sys.modules['numpy.random._mt19937'] = importlib.import_module('numpy.random._mt19937')
            except Exception:
                pass

        # Patch numpy.random._pickle constructors to handle class objects
        try:
            import numpy.random._pickle as nrp

            def make_safe(original):
                """Wrap constructor to handle both string names and class objects."""
                def wrapper(bit_generator_name='MT19937'):
                    if not isinstance(bit_generator_name, str):
                        try:
                            bit_generator_name = bit_generator_name.__name__
                        except Exception:
                            bit_generator_name = str(bit_generator_name)
                    return original(bit_generator_name)
                return wrapper

            for name in ('__bit_generator_ctor', '__generator_ctor', '__randomstate_ctor'):
                if hasattr(nrp, name):
                    setattr(nrp, name, make_safe(getattr(nrp, name)))
        except Exception:
            pass
    except Exception:
        pass


@st.cache_resource
def load_models():
    patch_legacy_numpy_pickle()

    model = None
    threshold = 0.5  # Ngưỡng mặc định
    
    if not MODELS_DIR.exists():
        st.error(f"❌ Không tìm thấy thư mục mô hình: {MODELS_DIR}")
        return None, threshold

    def safe_load(path):
        try:
            obj = joblib.load(path)
            return obj
        except Exception as exc:
            st.warning(f"⚠️ Không thể tải {path.name}: {type(exc).__name__}")
            return None

    model = safe_load(MODELS_DIR / "water_rf_model_best.pkl")
    loaded_threshold = safe_load(MODELS_DIR / "optimal_threshold.pkl")
    
    if loaded_threshold is not None:
        threshold = loaded_threshold

    return model, threshold

@st.cache_data
def load_dataset():
    if not DATA_FILE.exists():
        return None

    df = pd.read_csv(DATA_FILE, na_values=['', 'NA', 'NaN'])
    if TARGET_COLUMN not in df.columns:
        return None

    return df


def impute_dataframe(df, imputer_obj=None):
    if not df.isna().any().any():
        return df

    if imputer_obj is not None:
        try:
            return pd.DataFrame(imputer_obj.transform(df), columns=df.columns)
        except Exception:
            pass

    fallback = SimpleImputer(strategy='mean')
    return pd.DataFrame(fallback.fit_transform(df), columns=df.columns)


def format_score(value: float) -> str:
    return f"{value * 100:.2f}%"


# 1. ĐỊNH NGHĨA HÀM VẼ BIỂU ĐỒ (Có dùng cache để tăng tốc)
@st.cache_resource
def plot_pairplot(data):
    # Khởi tạo ma trận lưới, cắt bỏ nửa trên (corner=True)
    g = sns.PairGrid(data, hue="Potability", height=2, aspect=1.5, corner=True)
    
    # Chỉ vẽ đồ thị phân tán ở nửa dưới
    g.map_lower(sns.scatterplot)
    
    # Thêm chú giải (legend)
    g.add_legend()
    
    # ---------------------------------------------------------
    # TÙY CHỈNH CHÚ GIẢI (LEGEND) "Potability"
    # ---------------------------------------------------------
    if g.legend is not None:
        # Tăng kích thước tiêu đề "Potability"
        g.legend.set_title("Potability", prop={'size': 28, 'weight': 'bold'})
        
        # Tăng kích thước cho các nhãn giá trị (0 và 1)
        for text in g.legend.get_texts():
            text.set_fontsize(24) 
            
        # Tăng kích thước các dấu chấm màu (Đã fix lỗi AttributeError)
        for handle in g.legend.legend_handles:
            # Dành cho phiên bản dùng PathCollection
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([200]) 
            # Dành cho phiên bản dùng Line2D (Thường gặp trên môi trường Cloud)
            elif hasattr(handle, 'set_markersize'):
                handle.set_markersize(15)  # Lưu ý: markersize 15 nhìn sẽ tương đương với sizes 200
            
    # Tiêu đề tổng của biểu đồ
    g.fig.suptitle("Biểu đồ pair plot của các đặc trưng trong tập dữ liệu", y=1.00, fontsize=48, fontweight='bold')
    
    # ---------------------------------------------------------
    # TÙY CHỈNH KÍCH THƯỚC VÀ ĐỘ NGHIÊNG CỦA CHỮ TRÊN TRỤC
    # ---------------------------------------------------------
    for ax in g.axes.flatten():
        # Kiểm tra xem ô biểu đồ có tồn tại không
        if ax is not None:
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            
            # Trục X: Chữ to, in đậm, xoay 45 độ
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=24, fontweight='bold', rotation=45) 
                
            # Trục Y: Chữ to, in đậm, xoay 45 độ, và đẩy ra xa (labelpad)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=24, fontweight='bold', rotation=45, labelpad=40)
                
            # Xoay nghiêng và tăng kích thước các con số chia vạch (tick labels)
            ax.tick_params(axis='x', labelsize=24, rotation=45) 
            ax.tick_params(axis='y', labelsize=24, rotation=45) 

    return g.fig


st.sidebar.title("Navigation")
page = st.sidebar.radio("Chọn trang:", ["EDA", "Model Deployment", "Evaluation"])

if page == "EDA":
    st.title(PROJECT_TITLE)
    st.markdown("---")

    st.markdown(
        """
        **Tên đề tài:** Dự đoán nước an toàn dựa trên các chỉ số lý hóa bằng thuật toán RandomForest nhằm tối ưu hóa quy trình kiểm định chất lượng nước tại nguồn.\n 
        **Sinh viên:** Hồ Trọng Nghĩa. \n 
        **MSSV:** 22T1020683. \n
        **Giá trị thực tiễn:** Đưa ra cảnh báo sớm để tạm dừng cấp nước hoặc sục rửa hệ thống ngay khi phát hiện dấu hiệu bất thường, ngăn ngừa dịch bệnh tiêu hóa.
        """
    )
    df = load_dataset()
    if df is None:
        st.warning("⚠️ Không tìm thấy dữ liệu mẫu. Vui lòng thêm file `data/water_potability.csv`.")
    else:
        st.subheader("1. Tổng quan tập dữ liệu")
        st.write(f"- Số lượng bản ghi: **{len(df)}**")
        st.write(f"- Số lượng cột: **{len(df.columns)}**")

        st.write("**1.1. Dữ liệu thô (mẫu)**")
        st.dataframe(df, height=300, use_container_width=True)

        missing = df.isna().sum()
        st.write("**1.2. Giá trị thiếu theo cột:**")
        st.dataframe(missing[missing > 0].to_frame("Missing Count"))

        st.write("**1.3. Thông tin cơ bản các thuộc tính**")
        st.markdown("""
* **pH value (Giá trị pH):** Đánh giá sự cân bằng acid-base và xác định tính kiềm hay tính acid của nước. Ngưỡng khuyến nghị của WHO là từ **6.5 đến 8.5**.
* **Hardness (Độ cứng):** Gây ra bởi các muối Canxi và Magiê hòa tan từ địa tầng địa chất. Chỉ số này phản ánh khả năng kết tủa xà phòng của nước.
* **Solids (Tổng chất rắn hòa tan - TDS):** Đo lường nồng độ các khoáng chất, muối vô cơ và hữu cơ hòa tan (như Kali, Natri, Clorua, Sulfate...). Giới hạn lý tưởng là **500 mg/l** và tối đa là **1000 mg/l** cho nước uống.
* **Chloramines:** Chất khử trùng chính trong hệ thống nước công cộng, hình thành khi thêm Amoniac vào Clo. Mức an toàn tối đa là **4 mg/L (4 ppm)**.
* **Sulfate (Sunfat):** Chất tự nhiên tìm thấy trong khoáng vật và đá. Nồng độ trong nước ngọt thường từ **3 - 30 mg/L**, tuy nhiên có thể lên tới 1000 mg/L ở một số khu vực địa chất đặc thù.
* **Conductivity (Độ dẫn điện):** Đo lường nồng độ ion thông qua khả năng truyền dòng điện của nước. Theo tiêu chuẩn WHO, giá trị này không được vượt quá **400 μS/cm**.
* **Organic_carbon (Cacbon hữu cơ tổng số - TOC):** Đo lường tổng lượng cacbon trong các hợp chất hữu cơ từ tự nhiên hoặc tổng hợp. Tiêu chuẩn US EPA yêu cầu **< 2 mg/L** đối với nước uống đã qua xử lý.
* **Trihalomethanes (THMs):** Các phụ phẩm hóa học phát sinh khi xử lý nước bằng Clo. Nồng độ lên tới **80 ppm** được coi là an toàn cho người sử dụng.
* **Turbidity (Độ đục):** Đo lường lượng chất rắn ở trạng thái lơ lửng dựa trên tính chất phát xạ ánh sáng. Tiêu chuẩn của WHO khuyến nghị độ đục phải thấp hơn **5.00 NTU**.
* **Potability (Khả năng uống được):** Biến mục tiêu phân loại mức độ an toàn cho con người.
    * **1**: Có thể uống được (Potable).
    * **0**: Không thể uống được (Not potable).
        """)

        st.subheader("2. Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()
        
        fig2 = px.imshow(corr, 
                        title="Biểu đồ tương quan của tập dữ liệu",
                        labels=dict(x="Biến", y="Biến", color="Tương quan"),
                        x=corr.columns,
                        y=corr.columns,
                        color_continuous_scale='RdBu_r',
                        text_auto='.2f',
                        aspect="auto",
                        zmin=-1,
                        zmax=1)
        fig2.update_layout(
            # 1. Chỉnh kích thước và ép màu ĐEN cho tên biến ở trục X và Y
            xaxis_tickfont_size=24,
            xaxis_tickfont_color='black',  # Thêm dòng này để chữ trục X đậm hơn
            yaxis_tickfont_size=24,
            yaxis_tickfont_color='black',  # Thêm dòng này để chữ trục Y đậm hơn
            
            # 2. Chỉnh màu đen cho thanh chú thích (colorbar) bên phải
            coloraxis_colorbar=dict(
                yanchor="bottom",
                y=0,
                title_font=dict(color='black', size=14), # Làm đậm chữ "Tương quan"
                tickfont=dict(color='black', size=14)    # Làm đậm các con số 1, 0.5, 0...
            ),
            width=800,
            height=800,
            xaxis_title="",
            yaxis_title=""
        )
        fig2.update_traces(
        text=corr.values, 
        texttemplate='<b>%{text:.2f}</b>',  # Thêm thẻ <b>...</b> để in đậm
        textfont=dict(
            color='black',                  # Ép màu chữ thành đen (hoặc 'darkblue', 'darkred' tùy ý)
            size=16                         # Bạn có thể tăng giảm size chữ ở đây cho phù hợp
        )
    )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
**Nhận xét:**
Chỉ có Sulfate tương quan nghịch kém với Hardness (-0.11) và Solids (-0.17). Và Solids tương quan nghịch kém với pH (-0.09). Tương quan không mạnh nên không cần loại bỏ đặc trưng nào cả.
        """)


        # 2. PHẦN HIỂN THỊ LÊN STREAMLIT (Nơi bạn gọi hàm)
        # Lưu ý: 'df' là biến chứa DataFrame dữ liệu của bạn
        st.subheader("3. Pair Plot")
        with st.spinner("Đang tạo biểu đồ Pair Plot... (có thể mất vài giây)"):
            # Gọi hàm để lấy figure
            fig_pairplot_fig = plot_pairplot(df)
            
            # Hiển thị lên Streamlit
            st.pyplot(fig_pairplot_fig)
        st.markdown("""
**Nhận xét:**
Đám mây điểm ngẫu nhiên, không theo một trật tự hình khối nào cả. Hai biến của từng cặp hoàn toàn độc lập, không ảnh hưởng gì đến nhau. Điều này có nghĩa là chúng không trùng lặp thông tin với nhau, nên không cần xử lý gì cả.
        """)

        st.subheader("4. Skewness Check")
        selected_col = st.selectbox("Chọn cột để xem thống kê:", numeric_cols)

        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Tần suất")
        ax.set_title(f"Phân bố {selected_col}")
        st.pyplot(fig)
        st.markdown("""
**Nhận xét:**
* Bị lệch phải: **Solids**
* Hơi bị lệch phải: **Conductivity** 
    * Nếu dữ liệu ở 2 biến này bị thiếu thì điền median thay cho mean. 
        """)
        

        st.subheader("5. Outliers")

        features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        target = 'Potability'

        # 1. Khởi tạo biểu đồ
        df_melted = df.melt(id_vars=target, value_vars=features)

        fig_outliers = px.box(
            df_melted, 
            x=target,                
            y="value",              
            color=target,            
            facet_col="variable",
            facet_col_wrap=3,
            color_discrete_map={
                0: "#FF9999",  
                1: "#66b3ff"
            },
            title="<b>So sánh phân bố đặc trưng giữa nhóm nước không uống được và nước uống được</b>" 
        )

        # 2. ÉP TẤT CẢ FONT CHỮ THÀNH MÀU ĐEN (Phương án mạnh tay)
        fig_outliers.update_layout(
            height=900,
            font=dict(color="black"), # Dòng này sẽ bắt mọi chữ chưa được gán màu phải chuyển sang Đen
            title_font=dict(size=24, color='black'),          
            legend_title_font=dict(size=20, color='black'),   
            legend_font=dict(size=18, color='black')          
        )

        # 3. Tùy chỉnh TÊN CÁC ĐẶC TRƯNG (ph, Hardness...)
        fig_outliers.for_each_annotation(
            lambda a: a.update(
                text=f'<b>{a.text.split("=")[-1]}</b>',  
                font=dict(size=20, color='black')        
            )
        )

        # 4. Tùy chỉnh TRỤC X
        fig_outliers.update_xaxes(
            matches=None, 
            showticklabels=True,
            tickfont=dict(size=18, color='black'),            
            title_font=dict(size=20, color='black'),          
            color="black"  # Bắt buộc các đường kẻ vạch và nhãn mặc định phải màu đen
        )

        # 5. Tùy chỉnh TRỤC Y
        fig_outliers.update_yaxes(
            matches=None, 
            showticklabels=True,
            tickfont=dict(size=14, color='black'),            
            title_font=dict(size=18, color='black'),
            color="black"  # Bắt buộc các đường kẻ vạch và nhãn mặc định phải màu đen
        )

        # Bắt buộc giữ theme=None để tránh Streamlit nhúng tay vào
        st.plotly_chart(fig_outliers, use_container_width=True, theme=None)

        st.markdown("""
**Nhận xét:**
Giá trị outlier của các biến này đều là tự nhiên chứ không phải lỗi khi thu thập dữ liệu. Có thể thử ép outlier thay bằng giá trị râu (capping), để xem hiệu suất có tốt hơn không?
        """)

elif page == "Model Deployment":
    model, threshold = load_models()
    st.title("Model Deployment")
    st.markdown("---")
    st.write("Nhập thông số nước để dự đoán chất lượng nước uống.")

    if model is None:
        st.error("❌ Mô hình chưa được tải. Kiểm tra thư mục `models/`.")
    else:
        # BỎ st.form ĐI, DÙNG TRỰC TIẾP CÁC CỘT
        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input("pH", value=7.0, step=0.1)
            hardness = st.number_input("Hardness", value=200.0, step=1.0)
            solids = st.number_input("Solids", value=20000.0, step=100.0)
        with col2:
            chloramines = st.number_input("Chloramines", value=7.0, step=0.1)
            sulfate = st.number_input("Sulfate", value=350.0, step=1.0)
            conductivity = st.number_input("Conductivity", value=400.0, step=1.0)
        with col3:
            organic_carbon = st.number_input("Organic_carbon", value=15.0, step=0.1)
            trihalomethanes = st.number_input("Trihalomethanes", value=65.0, step=1.0)
            turbidity = st.number_input("Turbidity", value=3.5, step=0.1)

        # DÙNG NÚT BẤM BÌNH THƯỜNG (Bỏ st.form_submit_button)
        if st.button("🎯 Dự đoán"):
            input_data = pd.DataFrame({
                'ph': [ph],
                'Hardness': [hardness],
                'Solids': [solids],
                'Chloramines': [chloramines],
                'Sulfate': [sulfate],
                'Conductivity': [conductivity],
                'Organic_carbon': [organic_carbon],
                'Trihalomethanes': [trihalomethanes],
                'Turbidity': [turbidity]
            })

            try:
                # Xử lý giá trị thiếu
                input_data_imputed = impute_dataframe(input_data, None)
                
                # Lấy xác suất dự đoán
                probabilities = model.predict_proba(input_data_imputed)[:, 1]
                
                # Áp dụng ngưỡng tối ưu
                prediction = (probabilities[0] >= threshold).astype(int)
                
                # Hiển thị kết quả
                st.markdown("---")
                st.subheader("📊 Kết quả dự đoán")
                
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    st.metric("Độ tin cậy", f"{probabilities[0]*100:.2f}%")
                    st.metric("Ngưỡng quyết định", f"{threshold*100:.2f}%")
                
                with col_result2:
                    if prediction == 1:
                        st.success(f"✅ **NƯỚC AN TOÀN**\nĐủ tiêu chuẩn uống")
                    else:
                        st.error(f"❌ **NƯỚC KHÔNG AN TOÀN**\nKhông đủ tiêu chuẩn uống")
                
                # Hiển thị chi tiết thông số
                st.subheader("📋 Thông số đầu vào")
                st.dataframe(input_data)
                
            except Exception as exc:
                st.error(f"❌ Lỗi khi dự đoán: {exc}")



elif page == "Evaluation":
    # Load model và threshold (giống bên trang Deployment)
    model, threshold = load_models()
    
    st.title("Evaluation")
    st.markdown("---")

    if model is None:
        st.error("❌ Không thể đánh giá vì mô hình chưa được tải.")
    else:
        df = load_dataset()
        if df is None:
            st.warning("⚠️ Dữ liệu đánh giá chưa có. Vui lòng thêm file `data/water_potability.csv`.")
        else:
            st.subheader("1. Chuẩn bị dữ liệu đánh giá")
            st.write("Dữ liệu sẽ được điền khuyết theo giá trị trung bình của từng nhóm Potability, sau đó phân chia thành tập Train/Test.")

            # ---------------------------------------------------------
            # BƯỚC MỚI: ĐIỀN KHUYẾT DỮ LIỆU THEO NHÓM POTABILITY
            # ---------------------------------------------------------
            cols_to_fill = ['ph', 'Sulfate', 'Trihalomethanes']
            df[cols_to_fill] = df.groupby('Potability')[cols_to_fill].transform(lambda x: x.fillna(x.mean()))

            # Tách biến độc lập (X) và biến mục tiêu (y)
            X = df[FEATURE_COLUMNS].copy()
            y = df[TARGET_COLUMN].copy()
            
            # Impute tổng quát cho các cột khác nếu vẫn còn sót giá trị thiếu
            X = impute_dataframe(X, None)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(y.unique()) > 1 else None,
            )

            # ---------------------------------------------------------
            # BƯỚC 1: DỰ ĐOÁN XÁC SUẤT VÀ ÁP DỤNG NGƯỠNG TỐI ƯU
            # ---------------------------------------------------------
            y_test_proba = model.predict_proba(X_test)[:, 1]
            y_test_pred_custom = (y_test_proba >= threshold).astype(int)

            # Tính toán các độ đo
            pr_auc = average_precision_score(y_test, y_test_proba)
            accuracy = accuracy_score(y_test, y_test_pred_custom)
            precision = precision_score(y_test, y_test_pred_custom, zero_division=0)
            recall = recall_score(y_test, y_test_pred_custom, zero_division=0)
            f1 = f1_score(y_test, y_test_pred_custom, zero_division=0)
            report = classification_report(y_test, y_test_pred_custom, output_dict=True, zero_division=0)

            st.subheader(f"2. Kết quả đánh giá (Áp dụng ngưỡng T = {threshold:.2f})")
            
            metrics_df = pd.DataFrame({
                'PR-AUC': [pr_auc],
                'Precision': [precision],
                'Recall': [recall],
            })
            metrics_df = metrics_df.T.rename(columns={0: 'Value'})
            st.dataframe(metrics_df.style.format('{:.4f}'))

            # ---------------------------------------------------------
            # BƯỚC 2: VẼ MA TRẬN NHẦM LẪN VÀ PR CURVE (1 HÀNG, 2 CỘT)
            # ---------------------------------------------------------
            st.subheader("3. Biểu đồ đánh giá chi tiết")
            
            test_precisions, test_recalls, _ = precision_recall_curve(y_test, y_test_proba)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # --- BIỂU ĐỒ 1: CONFUSION MATRIX ---
            cm = confusion_matrix(y_test, y_test_pred_custom)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Dự đoán Bẩn (0)', 'Dự đoán Sạch (1)'],
                        yticklabels=['Thực tế Bẩn (0)', 'Thực tế Sạch (1)'],
                        ax=axes[0])
            axes[0].set_title(f'Confusion Matrix (Ngưỡng = {threshold:.2f})')
            axes[0].set_ylabel('Nhãn Thực tế')
            axes[0].set_xlabel('Nhãn Dự đoán')

            # --- BIỂU ĐỒ 2: PRECISION-RECALL CURVE ---
            axes[1].plot(test_recalls, test_precisions, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
            # Vẽ một điểm đỏ đánh dấu ngưỡng đã chọn
            axes[1].plot(recall, precision, marker='o', color='red', markersize=8, 
                         label=f'Ngưỡng chọn ({threshold:.2f})')

            axes[1].set_title('Precision-Recall Curve (Tập Test)')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].legend(loc='lower left')
            axes[1].grid(True, linestyle='--', alpha=0.6)

            # Hiển thị biểu đồ lên Streamlit
            st.pyplot(fig)

            # Hiển thị cảnh báo False Positive
            st.warning(f"⚠️ **Chú ý:** Có **{cm[0, 1]}** mẫu 'Nước Bẩn' bị mô hình dự đoán nhầm là 'Nước Sạch' (False Positives).")

            # ---------------------------------------------------------
            # BƯỚC 3: HIỂN THỊ NHẬN XÉT ĐÁNH GIÁ (MARKDOWN)
            # ---------------------------------------------------------
            st.markdown(f"""
### 💡 Nhận xét đánh giá mô hình (Tập Test)
* **Hiệu suất phân loại tốt:** Chỉ số PR-AUC đạt **{pr_auc:.4f}** khẳng định mô hình đã học được các quy luật thực tế để phân biệt nước an toàn và không an toàn, thay vì chỉ dự đoán ngẫu nhiên.
* **Đạt mục tiêu an toàn tuyệt đối:** Việc tinh chỉnh ngưỡng lên **{threshold:.2f}** đã đẩy độ chính xác (Precision) lên mức rất cao (**{precision*100:.1f}%**). Nghĩa là khi mô hình báo "Nước Sạch", độ tin cậy đạt trên {precision*100:.1f}%.
* **Kiểm soát rủi ro cực thấp:** Ma trận nhầm lẫn cho thấy chỉ có **{cm[0, 1]}** mẫu nước bẩn bị nhận diện sai (False Positives). Đây là con số rất nhỏ, giúp hạn chế tối đa rủi ro gây hại cho sức khỏe người dùng.
* **Sự đánh đổi tất yếu (Trade-off):** Để đạt mức độ an toàn cao, mô hình phải đánh đổi bằng độ bao phủ (Recall giảm còn **{recall*100:.1f}%**). Mô hình chấp nhận "chê nhầm" **{cm[1, 0]}** mẫu nước sạch (False Negatives) để đảm bảo không bỏ lọt nước bẩn.
            """)