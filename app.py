import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Cấu hình trang
st.set_page_config(page_title="Water Quality Prediction", layout="wide")

# Tiêu đề
st.title("🌊 Water Quality Prediction System")
st.markdown("---")

# Đường dẫn thư mục models
models_dir = Path("models")

# Load các mô hình
@st.cache_resource
def load_models():
    try:
        # Thử load với encoding để tương thích nhiều Python version
        with open(models_dir / "water_imputer.pkl", "rb") as f:
            imputer = pickle.load(f, encoding='latin1')
        with open(models_dir / "water_scaler.pkl", "rb") as f:
            scaler = pickle.load(f, encoding='latin1')
        with open(models_dir / "water_rf_model.pkl", "rb") as f:
            model = pickle.load(f, encoding='latin1')
        return imputer, scaler, model
    except FileNotFoundError as e:
        st.error(f"❌ Lỗi: Không tìm thấy mô hình. {e}")
        st.error(f"📁 Đường dẫn cần: {models_dir}")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Lỗi khi load mô hình: {str(e)}")
        st.error("💡 **Giải pháp:**")
        st.error("- File pickle có thể không tương thích với Python version")
        st.error("- Vui lòng tạo lại pickle bằng: `pickle.dump(model, f, protocol=2)`")
        return None, None, None

imputer, scaler, model = load_models()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Chọn trang:", ["Home", "Prediction", "Analysis"])

if page == "Home":
    st.header("Welcome to Water Quality Prediction!")
    st.write("""
    Hệ thống này được xây dựng để dự đoán chất lượng nước dựa trên các thông số không khí và hoá học.
    
    **Thông số đầu vào:**
    - pH: Độ axit hoặc kiềm của nước
    - Nhiệt độ (Temperature): Độ nóng của nước (°C)
    - Oxy hòa tan (Dissolved Oxygen): Lượng oxy trong nước (mg/L)
    - Độ đục (Turbidity): Độ mờ của nước (NTU)
    
    Sử dụng tab "Prediction" để dự đoán chất lượng nước.
    """)
    
elif page == "Prediction":
    if model is None:
        st.error("❌ Không thể tải mô hình. Vui lòng kiểm tra file trong thư mục models/.")
    else:
        st.header("🔮 Dự đoán chất lượng nước")
        st.write("Nhập thông số nước của bạn:")
        
        col1, col2 = st.columns(2)
        with col1:
            ph = st.slider("pH", 0.0, 14.0, 7.0, step=0.1)
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, step=0.5)
        
        with col2:
            dissolved_oxygen = st.slider("Dissolved Oxygen (mg/L)", 0.0, 15.0, 8.0, step=0.1)
            turbidity = st.slider("Turbidity (NTU)", 0.0, 100.0, 5.0, step=0.5)
        
        if st.button("🎯 Dự đoán", key="predict_btn"):
            # Tạo dataframe từ input
            input_data = pd.DataFrame({
                'ph': [ph],
                'Tempreture': [temperature],  # Giữ nguyên tên cột nếu model dùng tên này
                'Dissolved_oxygen': [dissolved_oxygen],
                'Turbidity': [turbidity]
            })
            
            # Xử lý dữ liệu
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            # Dự đoán
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Hiển thị kết quả
            st.markdown("---")
            st.subheader("📊 Kết quả dự đoán:")
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1:
                    st.success("✅ **Chất lượng: TỐT (Sạch)**")
                else:
                    st.error("❌ **Chất lượng: XẤU (Không sạch)**")
            
            with col2:
                st.metric("Độ tin cậy", f"{max(probability) * 100:.2f}%")
            
            # Chi tiết xác suất
            st.write("**Xác suất chi tiết:**")
            prob_df = pd.DataFrame({
                'Chất lượng': ['Sạch', 'Không sạch'],
                'Xác suất (%)': [probability[1] * 100, probability[0] * 100]
            })
            st.bar_chart(prob_df.set_index('Chất lượng'))

elif page == "Analysis":
    st.header("📊 Phân tích dữ liệu")
    
    data_file = Path("data/water_potability.csv")
    if data_file.exists():
        try:
            df = pd.read_csv(data_file)
            
            st.write(f"**Số dòng dữ liệu:** {len(df)}")
            st.write(f"**Số cột:** {len(df.columns)}")
            
            # Hiển thị thông tin cơ bản
            st.subheader("Mô tả dữ liệu:")
            st.dataframe(df.describe())
            
            # Biểu đồ phân bố
            st.subheader("📈 Phân bố dữ liệu:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            selected_col = st.selectbox("Chọn cột để xem biểu đồ:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            df[selected_col].hist(bins=30, ax=ax, edgecolor='black', color='skyblue')
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Tần suất")
            ax.set_title(f"Phân bố {selected_col}")
            st.pyplot(fig)
            
            # Hiển thị dữ liệu
            if st.checkbox("Xem dữ liệu thô"):
                st.dataframe(df.head(100))
                
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {e}")
    else:
        st.warning(f"⚠️ Không tìm thấy file {data_file}. Vui lòng đặt file CSV vào thư mục data/")
