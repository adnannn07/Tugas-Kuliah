import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Analisis Produksi Padi Sumatera", layout="wide")

# Load data sintetis
@st.cache_data
def load_data():
    np.random.seed(42)
    years = list(range(1993, 2021))
    provinces = ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 
                 'Sumatera Selatan', 'Bengkulu', 'Lampung', 
                 'Kepulauan Bangka Belitung', 'Kepulauan Riau']
    
    data = []
    for year in years:
        for province in provinces:
            luas_panen = np.random.uniform(50000, 250000)  # Diubah menjadi 250000
            curah_hujan = np.random.uniform(1500, 3000)
            kelembapan = np.random.uniform(70, 90)
            suhu_rata = np.random.uniform(25, 32)
            produksi = luas_panen * np.random.uniform(4, 7) + np.random.normal(0, 10000)
            data.append({
                'Tahun': year,
                'Provinsi': province,
                'Luas Panen': luas_panen,
                'Curah hujan': curah_hujan,
                'Kelembapan': kelembapan,
                'Suhu rata-rata': suhu_rata,
                'Produksi': produksi
            })
    return pd.DataFrame(data)

# Load & filter data
df = load_data()
st.title("🌾 Analisis Produksi Padi di Sumatera")

# Sidebar filter
st.sidebar.header("Filter")
tahun_range = st.sidebar.slider("Rentang Tahun", int(df.Tahun.min()), int(df.Tahun.max()), (1993, 2020))
provinsi = st.sidebar.multiselect("Pilih Provinsi", df.Provinsi.unique(), default=list(df.Provinsi.unique()))

df_filtered = df[(df['Tahun'] >= tahun_range[0]) & (df['Tahun'] <= tahun_range[1]) & (df['Provinsi'].isin(provinsi))]

# Statistik utama
st.subheader("Statistik Utama")
col1, col2, col3, col4 = st.columns(4)  
col1.metric("Total Produksi (Ton)", f"{df_filtered['Produksi'].sum():,.0f}")
col2.metric("Rata-rata Luas Panen", f"{df_filtered['Luas Panen'].mean():,.0f} Ha")
col3.metric("Rata-rata Curah Hujan", f"{df_filtered['Curah hujan'].mean():,.0f} mm")
col4.metric("Rata-rata Suhu", f"{df_filtered['Suhu rata-rata'].mean():.1f} °C")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Tren Waktu", "Perbandingan Provinsi", "Korelasi", "Prediksi"])

with tab1:
    st.subheader("📈 Tren Produksi Padi per Tahun")
    produksi_tahunan = df_filtered.groupby("Tahun")["Produksi"].sum().reset_index()
    st.plotly_chart(px.line(produksi_tahunan, x="Tahun", y="Produksi", markers=True), use_container_width=True)

with tab2:
    st.subheader("📊 Produksi per Provinsi")
    prov_produksi = df_filtered.groupby("Provinsi")["Produksi"].sum().sort_values()
    st.plotly_chart(px.bar(prov_produksi, orientation='h', title="Total Produksi per Provinsi"), use_container_width=True)

with tab3:
    st.subheader("📌 Korelasi Variabel")
    corr = df_filtered[['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata', 'Produksi']].corr()
    fig = px.imshow(corr, text_auto=True, title="Matriks Korelasi")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("🤖 Prediksi Produksi Padi")

    fitur = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
    X = df_filtered[fitur]
    y = df_filtered['Produksi']

    # Pastikan tidak ada nilai NaN dalam data
    if X.isnull().any().any() or y.isnull().any():
        st.warning("⚠️ Terdapat nilai kosong (NaN) dalam data. Melakukan pembersihan data...")
        df_clean = df_filtered.dropna(subset=fitur+['Produksi'])
        X = df_clean[fitur]
        y = df_clean['Produksi']

    if X.empty or y.empty:
        st.warning("⚠️ Tidak ada data yang tersedia untuk prediksi. Silakan pilih rentang tahun atau provinsi lain.")
        st.stop()

    if len(X) < 2:
        st.warning("⚠️ Data yang dipilih terlalu sedikit untuk membagi data train-test. Minimal diperlukan 2 baris data.")
        st.stop()

    # Model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Tabel hasil prediksi vs aktual
    df_prediksi = pd.DataFrame({
        "Aktual": y_test.values,
        "Prediksi": y_pred
    })
    df_prediksi["Selisih"] = df_prediksi["Aktual"] - df_prediksi["Prediksi"]
    df_prediksi["Error (%)"] = (abs(df_prediksi["Selisih"]) / df_prediksi["Aktual"]) * 100
    
        # Evaluasi model
    st.subheader("📊 Evaluasi Model")
    y_pred = model.predict(X_test)
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    col2.metric("MSE", f"{mean_squared_error(y_test, y_pred):,.0f}")

    # Grafik perbandingan
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df_prediksi["Aktual"], mode='lines+markers', name='Aktual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=df_prediksi["Prediksi"], mode='lines+markers', name='Prediksi', line=dict(color='red')))
    fig.update_layout(
        title="Perbandingan Prediksi vs Aktual",
        xaxis_title="Index",
        yaxis_title="Produksi (Ton)",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Grafik Error
    st.subheader("📉 Grafik Persentase Error")
    fig_error = px.line(df_prediksi, y="Error (%)", title="Persentase Error per Index")
    st.plotly_chart(fig_error, use_container_width=True)

    # Tabel Data
    st.dataframe(df_prediksi.round(2))

    # Download button
    st.download_button(
        label="📥 Unduh Hasil Prediksi",
        data=df_prediksi.to_csv(index=False).encode('utf-8'),
        file_name="hasil_prediksi.csv",
        mime="text/csv"
    )

    # Prediksi untuk tahun berikutnya (2021-2030)
    st.markdown("### Prediksi Produksi Tahun 2021-2030")
    
    try:
        # Hitung rata-rata fitur 5 tahun terakhir (maksimal sampai 2020)
        tahun_terakhir = min(2020, df_filtered['Tahun'].max())
        tahun_awal = max(tahun_terakhir - 4, df_filtered['Tahun'].min())  # 5 tahun terakhir
        
        if tahun_terakhir - tahun_awal < 4:
            st.warning(f"⚠️ Data hanya tersedia dari {tahun_awal} sampai {tahun_terakhir}. Prediksi mungkin kurang akurat.")
        
        last_years_avg = df_filtered[df_filtered['Tahun'].between(tahun_awal, tahun_terakhir)]\
                        .groupby('Tahun')[fitur].mean().mean().values
        
        # Buat dataframe untuk prediksi
        future_years = list(range(2021, 2031))
        prediksi_data = []
        
        for year in future_years:
            # Asumsikan pertumbuhan linier 2% per tahun untuk variabel input
            growth_factor = 1.02 ** (year - tahun_terakhir)
            adjusted_features = last_years_avg * growth_factor
            
            # Pastikan tidak ada nilai NaN sebelum prediksi
            if np.isnan(adjusted_features).any():
                st.error(f"❌ Nilai NaN terdeteksi dalam fitur untuk tahun {year}")
                continue
                
            prediksi = model.predict([adjusted_features])[0]
            
            prediksi_data.append({
                'Tahun': year,
                'Luas Panen': adjusted_features[0],
                'Curah Hujan': adjusted_features[1],
                'Kelembapan': adjusted_features[2],
                'Suhu Rata-rata': adjusted_features[3],
                'Produksi Prediksi': prediksi
            })
        
        df_prediksi_future = pd.DataFrame(prediksi_data)
        
        # Tampilkan grafik prediksi
        fig_future = px.line(df_prediksi_future, x='Tahun', y='Produksi Prediksi', 
                            title=f'Prediksi Produksi Padi {future_years[0]}-{future_years[-1]}', 
                            markers=True)
        st.plotly_chart(fig_future, use_container_width=True)
        
        # Tampilkan tabel prediksi
        st.subheader("📋 Detail Prediksi {}-{}".format(future_years[0], future_years[-1]))
        st.dataframe(df_prediksi_future.style.format({
            'Luas Panen': '{:,.0f} Ha',
            'Curah Hujan': '{:,.0f} mm',
            'Kelembapan': '{:.1f}%',
            'Suhu Rata-rata': '{:.1f} °C',
            'Produksi Prediksi': '{:,.0f} Ton'
        }))
        
    except Exception as e:
        st.error(f"❌ Gagal membuat prediksi: {str(e)}")
        st.error("Pastikan data tersedia untuk tahun-tahun terakhir (minimal 5 tahun terakhir)")
    
    # Prediksi manual
    st.subheader("🔮 Prediksi Manual")
    st.write("Masukkan parameter untuk memprediksi produksi padi:")
    
    col1, col2, col3, col4 = st.columns(4)
    luas = col1.number_input("Luas Panen (Ha)", min_value=0.0, max_value=500000.0, value=100000.0, step=1000.0)
    hujan = col2.number_input("Curah Hujan (mm)", min_value=0.0, max_value=5000.0, value=2000.0, step=100.0)
    lembap = col3.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    suhu = col4.number_input("Suhu Rata-rata (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.1)
    
    if st.button("Prediksi Sekarang"):
        hasil = model.predict([[luas, hujan, lembap, suhu]])
        st.success(f"Prediksi Produksi: {hasil[0]:,.0f} Ton")
        st.balloons()