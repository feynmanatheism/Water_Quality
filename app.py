import importlib
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
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



model, threshold = load_models()

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
        st.subheader("Tổng quan tập dữ liệu")
        st.write(f"- Số lượng bản ghi: **{len(df)}**")
        st.write(f"- Số lượng cột: **{len(df.columns)}**")

        missing = df.isna().sum()
        st.write("**Giá trị thiếu theo cột:**")
        st.dataframe(missing[missing > 0].to_frame("Missing Count"))

        st.subheader("Biểu đồ phân bố dữ liệu")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Chọn cột để xem thống kê:", numeric_cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        df[selected_col].hist(bins=30, ax=ax, edgecolor='black', color='skyblue')
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Tần suất")
        ax.set_title(f"Phân bố {selected_col}")
        st.pyplot(fig)

        st.subheader("Ma trận tương quan")
        corr = df[numeric_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
        ax2.set_title("Heatmap tương quan giữa các biến")
        st.pyplot(fig2)

        if st.checkbox("Xem mẫu dữ liệu" ):
            st.dataframe(df.head(10))

elif page == "Model Deployment":
    st.title("Model Deployment")
    st.markdown("---")
    st.write("Nhập thông số nước để dự đoán chất lượng nước uống.")

    if model is None:
        st.error("❌ Mô hình chưa được tải. Kiểm tra thư mục `models/`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            hardness = st.number_input("Hardness", min_value=0.0, max_value=500.0, value=200.0, step=1.0)
            solids = st.number_input("Solids", min_value=0.0, max_value=50000.0, value=20000.0, step=100.0)
        with col2:
            chloramines = st.number_input("Chloramines", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
            sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=350.0, step=1.0)
            conductivity = st.number_input("Conductivity", min_value=0.0, max_value=1000.0, value=400.0, step=1.0)
        with col3:
            organic_carbon = st.number_input("Organic_carbon", min_value=0.0, max_value=30.0, value=15.0, step=0.1)
            trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=150.0, value=65.0, step=1.0)
            turbidity = st.number_input("Turbidity", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

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
    st.title("Evaluation")
    st.markdown("---")

    if model is None:
        st.error("❌ Không thể đánh giá vì mô hình chưa được tải.")
    else:
        df = load_dataset()
        if df is None:
            st.warning("⚠️ Dữ liệu đánh giá chưa có. Vui lòng thêm file `data/water_potability.csv`.")
        else:
            st.subheader("Chuẩn bị dữ liệu đánh giá")
            st.write("Dữ liệu sẽ được impute nếu tồn tại giá trị thiếu và chuẩn hóa bằng scaler.")

            X = df[FEATURE_COLUMNS].copy()
            y = df[TARGET_COLUMN].copy()
            X = impute_dataframe(X, None)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(y.unique()) > 1 else None,
            )

            y_pred = model.predict(X_test)

            st.subheader("Kết quả đánh giá")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            metrics_df = pd.DataFrame({
                'Accuracy': [accuracy],
                'Precision': [precision],
                'Recall': [recall],
                'F1 Score': [f1],
            })
            metrics_df = metrics_df.T.rename(columns={0: 'Value'})
            st.dataframe(metrics_df.style.format('{:.4f}'))

            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['Không sạch', 'Sạch'], columns=['Dự đoán Không sạch', 'Dự đoán Sạch'])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_ylabel('Thực tế')
            ax.set_xlabel('Dự đoán')
            st.pyplot(fig)

            st.subheader("Chi tiết báo cáo phân loại")
            st.dataframe(pd.DataFrame(report).transpose().round(4))
