import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi & Analisis Harga Laptop 💰",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Variabel Global ---
CSV_FILE = 'laptop_prices.csv'
FEATURE_COLUMNS = ['Brand', 'Processor_Type', 'RAM_GB', 'Storage_GB', 'Storage_Type', 'GPU_Type', 'Screen_Size_Inch']
TARGET_COLUMN = 'Price_Million_IDR'
PRICE_CORRECTION_FACTOR = 10 

# --- Fungsi Pemuatan Data dan Pelatihan Model ---
@st.cache_data
def load_data(file_path):
    """Memuat data dan melakukan koreksi harga."""
    try:
        data = pd.read_csv(file_path)
        data['RAM_GB'] = data['RAM_GB'].astype(int)
        data['Storage_GB'] = data['Storage_GB'].astype(int)
        data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors='coerce')
        data.dropna(subset=[TARGET_COLUMN], inplace=True)
        
        # *** KOREKSI HARGA: Dibagi 10 sesuai permintaan Anda ***
        data[TARGET_COLUMN] = data[TARGET_COLUMN] / PRICE_CORRECTION_FACTOR
        # *******************************************************
        
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memuat data: {e}")
        return None

@st.cache_resource
def train_model(data):
    """Melatih model Machine Learning."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['Brand', 'Processor_Type', 'Storage_Type', 'GPU_Type']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    
    model.fit(X_train, y_train)
    return model

# Muat data
df = load_data(CSV_FILE)
if df is not None:
    # Latih model
    model = train_model(df)
    data_count = len(df)
    
    # Ambil opsi unik untuk UI
    options = {
        'brand': sorted(df['Brand'].unique()), 
        'processor': sorted(df['Processor_Type'].unique()), 
        'ram': sorted(df['RAM_GB'].unique().astype(str)), 
        'storage_type': sorted(df['Storage_Type'].unique()), 
        'gpu': sorted(df['GPU_Type'].unique()), 
        'screen_size': sorted(df['Screen_Size_Inch'].unique().round(1).astype(str)), 
        'storage_size': sorted(df['Storage_GB'].unique().astype(str))
    }
    
    # --- Styling Custom ---
    st.markdown("""
        <style>
            .big-font { font-size:30px !important; font-weight: bold; }
            div[data-testid="stMetricValue"] { font-size: 24px; font-weight: bold; color: #1E88E5; }
        </style>
        """, unsafe_allow_html=True)

    # --- TABS: Prediksi dan Analisis ---
    st.title("💻 Aplikasi Prediksi Harga Laptop")
    pred_tab, anal_tab = st.tabs(["💰 Prediksi Harga", "📈 Analisis Data & Grafik"])

    # ----------------------------------------------------
    #                       TAB PREDIKSI
    # ----------------------------------------------------

    with pred_tab:
        st.subheader("Temukan perkiraan harga rata-rata laptop Anda berdasarkan spesifikasi.")
        st.markdown("---")
        
        # Ringkasan Metrik
        col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
        with col_metric_1: st.metric(label="📊 Jumlah Data Latih", value=f"{data_count} Baris")
        with col_metric_2: st.metric(label="🧠 Model Digunakan", value="Random Forest Regressor")
        with col_metric_3: st.metric(label="🎯 Harga Dikoreksi", value=f"Dibagi {PRICE_CORRECTION_FACTOR}")

        st.markdown("---")

        st.header("Masukkan Spesifikasi Laptop 🛠️")
        
        with st.container(border=True):
            st.markdown("#### Detail Komponen Utama")
            
            # Form Input
            col1, col2 = st.columns(2)
            with col1:
                company = st.selectbox("1. Merek Laptop (Brand) 🏷️", options['brand'])
            with col2:
                cpu_name = st.selectbox("2. Tipe Prosesor (CPU) 🚀", options['processor'])

            st.markdown("---")
            col3, col4, col5 = st.columns(3)
            with col3:
                ram_gb_str = st.selectbox("3. Ukuran RAM (GB) 💾", options['ram'])
            with col4:
                storage_size_str = st.selectbox("4. Ukuran Penyimpanan (GB) 📦", options['storage_size'])
            with col5:
                storage_type = st.selectbox("5. Tipe Penyimpanan (HDD/SSD) 🗃️", options['storage_type'])

            st.markdown("---")
            col6, col7 = st.columns(2)
            with col6:
                vga_name = st.selectbox("6. Tipe GPU (Graphics Card) 🖼️", options['gpu'])
            with col7:
                screen_size_str = st.selectbox("7. Ukuran Layar (Inch) 🖥️", options['screen_size'])
                
        # Konversi input
        try:
            ram_gb = int(ram_gb_str)
            storage_size = int(storage_size_str)
            screen_size = float(screen_size_str)
        except ValueError:
            st.error("Pastikan semua input numerik valid.")
            st.stop()

        input_data = pd.DataFrame([[company, cpu_name, ram_gb, storage_size, storage_type, vga_name, screen_size]], 
                                  columns=FEATURE_COLUMNS)

        st.markdown("---")
        result_placeholder = st.empty()
        
        if st.button("PREDIKSI HARGA", type="primary", use_container_width=True):
            with st.spinner('⏳ Model sedang menghitung prediksi...'):
                time.sleep(1.5)
                predicted_price_million_corrected = model.predict(input_data)[0] 
            
            with result_placeholder.container():
                st.success(f"## ✅ Prediksi Harga Rata-rata Laptop Anda:")
                
                harga_rupiah_penuh = predicted_price_million_corrected * 1_000_000 
                formatted_price = "Rp {:,.0f}".format(harga_rupiah_penuh).replace(',', '#').replace('.', ',').replace('#', '.')

                st.markdown(f"""
                    <div style='background-color: #e6ffe6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                        <p class='big-font' style='color: #2e8b57;'>{formatted_price}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.caption(f"Angka ini setara dengan **{predicted_price_million_corrected:,.2f} Juta IDR**.") 
                st.caption(f"Perkiraan dihasilkan berdasarkan {data_count} data yang telah dikoreksi (dibagi {PRICE_CORRECTION_FACTOR}).")

            st.balloons()
            
    # ----------------------------------------------------
    #                       TAB ANALISIS
    # ----------------------------------------------------
    
    with anal_tab:
        st.header("📈 Analisis Harga dan Korelasi Data")
        st.info("Visualisasi ini menunjukkan hubungan antara spesifikasi laptop dan harga jualnya, yang digunakan untuk melatih model prediksi.")
        
        st.markdown("---")

        # 1. Histogram Distribusi Harga (dari 'dari ipynb.py')
        st.subheader("1. Distribusi Harga Laptop")
        st.markdown("Menunjukkan sebaran harga laptop (dalam Juta IDR yang sudah dikoreksi) di seluruh dataset.")
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        sns.histplot(df[TARGET_COLUMN], kde=True, ax=ax_hist, bins=15, color='#1E88E5')
        ax_hist.set_title(f'Distribusi Harga Laptop (Juta IDR, Dikoreksi)', fontsize=14)
        ax_hist.set_xlabel('Harga (Juta IDR)')
        ax_hist.set_ylabel('Frekuensi')
        st.pyplot(fig_hist)
        
        st.markdown("---")

        # 2. Scatter Plot RAM vs Harga (dari 'dari ipynb.py')
        st.subheader("2. Korelasi RAM vs Harga")
        st.markdown("Melihat apakah peningkatan RAM berhubungan positif dengan harga jual.")
        
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        sns.regplot(x='RAM_GB', y=TARGET_COLUMN, data=df, ax=ax_scatter, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        ax_scatter.set_title('Hubungan RAM vs Harga', fontsize=14)
        ax_scatter.set_xlabel('RAM (GB)')
        ax_scatter.set_ylabel('Harga (Juta IDR)')
        st.pyplot(fig_scatter)
        
        st.markdown("---")

        # 3. Box Plot GPU Type vs Harga (dari 'dari ipynb.py')
        st.subheader("3. Perbandingan Harga berdasarkan Tipe GPU")
        st.markdown("Membandingkan rata-rata harga laptop dengan GPU *Integrated* (terintegrasi) dan *Dedicated* (terpisah).")
        
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='GPU_Type', y=TARGET_COLUMN, data=df, ax=ax_box, palette=['#FFC107', '#4CAF50'])
        ax_box.set_title('Perbandingan Harga berdasarkan Tipe GPU', fontsize=14)
        ax_box.set_xlabel('Tipe GPU')
        ax_box.set_ylabel('Harga (Juta IDR)')
        st.pyplot(fig_box)

else:
    st.error("Gagal memuat atau memproses data. Mohon periksa file `laptop_prices.csv` Anda.")