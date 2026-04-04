import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(page_title="Water Quality Prediction", layout="wide")

# Tiêu đề
st.title("🌊 Water Quality Prediction System")
st.markdown("---")

# Đường dẫn thư mục models
models_dir = Path("models")

# Load các mô hình bằng joblib (tương thích scikit-learn tốt hơn)
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load(models_dir / "water_scaler.pkl")
        model = joblib.load(models_dir / "water_rf_model.pkl")
        
        # ⚠️ KHÔNG load imputer - nó bị lỗi version incompatibility
        # Vì user input từ st.number_input đã đầy đủ, không có missing values
        
        return scaler, model
    except FileNotFoundError as e:
        st.error(f"❌ Lỗi: Không tìm thấy mô hình. {e}")
        st.error(f"📁 Đường dẫn cần: {models_dir}")
        return None, None
    except Exception as e:
        st.error(f"❌ Lỗi khi load mô hình: {str(e)}")
        return None, None

scaler, model = load_models()

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
        
        # Tạo 3 cột input
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
        
        if st.button("🎯 Dự đoán", key="predict_btn"):
            # Tạo dataframe với đúng tên cột từ training (9 tính năng)
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
                # ⚠️ KHÔNG impute - dữ liệu từ st.number_input đã đầy đủ, không có missing values
                # Chỉ cần scale & predict
                input_scaled = scaler.transform(input_data)
                
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
                
            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {str(e)}")

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
