import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Models
from sklearn.ensemble import RandomForestRegressor

# Misc
from haversine import haversine
import pickle
import os

import folium
import branca.colormap as cm
from streamlit_folium import st_folium


# Random state
rs = 42

# Fungsi untuk memuat data dengan caching untuk meningkatkan performa
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

train = pd.read_csv('https://media.githubusercontent.com/media/Firdatulayuni/Prediksi-Emisi/main/train.csv')

# Membuat salinan data asli untuk preprocessing
train_processed = train.copy()

# Mengatur opsi tampilan pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Judul utama
st.markdown("<h1 style='text-align: center;'>PREDIKSI EMISI CO2 DI RWANDA</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>KELOMPOK SEKTE ULTI NOLAN</h3>", unsafe_allow_html=True)

# Tabs untuk navigasi
deskripsi, eda, prediksi  = st.tabs(["Deskripsi Dataset", "EDA", "Prediksi Emisi CO2"])

# Tab Deskripsi
with deskripsi:
    desc = """
    Dataset CO2 Emission In Rwanda merupakan data emisi open source yang berasal dari pengamatan satelit Sentinel-5P untuk memprediksi emisi karbon. 
    Sekitar 497 lokasi unik dipilih dari berbagai wilayah di Rwanda, dengan distribusi di sekitar lahan pertanian, kota, dan pembangkit listrik. Tujuh 
    fitur utama diekstraksi setiap minggu dari Sentinel-5P dari Januari 2019 hingga November 2022. Setiap fitur (Sulfur Dioksida, Karbon Monoksida, dll) 
    mengandung sub fitur seperti column_number_density yang merupakan kerapatan kolom vertikal di permukaan tanah, yang dihitung dengan menggunakan teknik Spektroskopi Serapan Optik Diferensial (DOAS).
    Berikut merupakan penjelasan dari fitur-fitur utama yang ada pada dataset. 
    <br><b>1. Sulfur Dioksida (SO2)</b><br>
        SO2 memasuki atmosfer bumi melalui proses alami dan antropogenik. Ini berperan dalam bidang kimia pada skala lokal dan global, dampaknya berkisar dari
        polusi jangka pendek hingga dampak terhadap iklim. anya sekitar 30% emisi SO2 yang berasal dari sumber alami; mayoritas berasal dari antropogenik. 
        Emisi SO 2 berdampak buruk terhadap kesehatan manusia dan kualitas udara. SO 2 berdampak pada iklim melalui gaya radiasi, melalui pembentukan aerosol sulfat. 
        Emisi SO 2 vulkanik juga dapat menimbulkan ancaman bagi penerbangan, bersamaan dengan abu vulkanik.
    <br><b>2. Karbon Monoksida (CO)</b><br>
        Karbon monoksida (CO) adalah gas jejak atmosfer yang penting untuk memahami kimia troposfer. Di wilayah perkotaan tertentu, ini merupakan polutan atmosfer yang utama. 
        Sumber utama CO adalah pembakaran bahan bakar fosil, pembakaran biomassa, dan oksidasi metana dan hidrokarbon lainnya di atmosfer. Meskipun pembakaran bahan bakar fosil 
        merupakan sumber utama CO di garis lintang tengah utara, oksidasi isoprena dan pembakaran biomassa memainkan peranan penting di daerah tropis. 
        TROPOMI pada satelit Sentinel 5 Precursor (S5P) mengamati CO kelimpahan global memanfaatkan pengukuran pancaran bumi pada langit cerah dan langit mendung dalam rentang 
        spektral 2,3 m dari bagian inframerah gelombang pendek (SWIR) dari spektrum matahari. 
    <br><b>3. Nitrogen Dioksida (NO2)</b><br>
        Nitrogen Doksida (NO 2 dan NO) adalah gas penting di atmosfer bumi, yang terdapat di troposfer dan stratosfer. Mereka memasuki atmosfer sebagai akibat dari aktivitas antropogenik 
        (terutama pembakaran bahan bakar fosil dan pembakaran biomassa) dan proses alami (kebakaran hutan, petir, dan proses mikrobiologis di dalam tanah). Di sini, NO 2 digunakan untuk 
        mewakili konsentrasi kolektif nitrogen oksida karena pada siang hari, yaitu dengan adanya sinar matahari, siklus fotokimia yang melibatkan ozon (O 3 ) mengubah NO menjadi NO 2 
        dan sebaliknya dalam skala waktu menit.
    <br><b>4. Formaldehida (HCHO)</b><br>
        Formaldehida adalah gas perantara di hampir semua rantai oksidasi senyawa organik volatil non-metana (NMVOC), yang pada akhirnya menghasilkan CO 2 . Senyawa Organik Volatil Non-Metana (NMVOCs), 
        bersama dengan NOx, CO dan CH4, merupakan salah satu prekursor terpenting O3 troposfer . Sumber utama HCHO di atmosfer terpencil adalah oksidasi CH4 . Di seluruh benua, oksidasi NMVOC 
        yang lebih tinggi yang dipancarkan dari tumbuh-tumbuhan, kebakaran, lalu lintas, dan sumber-sumber industri menghasilkan peningkatan tingkat HCHO yang penting dan terlokalisasi. 
        Variasi distribusi formaldehida secara musiman dan antar-tahunan pada dasarnya tidak hanya berkaitan dengan perubahan suhu dan kejadian kebakaran, tetapi juga dengan perubahan aktivitas antropogenik.
    <br><b>5. UV Aerosol Index (UVAI)</b><br>
        Disebut juga dengan Absorbing Aerosol Index (AAI), AAI didasarkan pada perubahan hamburan Rayleigh yang bergantung pada panjang gelombang dalam rentang spektral UV untuk sepasang panjang gelombang. 
        Perbedaan antara hasil reflektansi yang diamati dan yang dimodelkan pada AAI. Jika AAI positif, hal ini menunjukkan adanya aerosol penyerap UV seperti debu dan asap. Hal ini berguna untuk melacak 
        evolusi gumpalan aerosol episodik dari semburan debu, abu vulkanik, dan pembakaran biomassa.
    <br><b>6. Ozon (O3)</b><br>
        Di stratosfer, lapisan ozon melindungi biosfer dari radiasi ultraviolet matahari yang berbahaya. Di troposfer, ia bertindak sebagai bahan pembersih yang efisien, namun pada konsentrasi tinggi ia juga berbahaya 
        bagi kesehatan manusia, hewan, dan tumbuh-tumbuhan. Ozon juga merupakan penyumbang gas rumah kaca yang penting terhadap perubahan iklim yang sedang berlangsung. Sejak penemuan lubang ozon Antartika pada tahun 1980an
        dan Protokol Montreal yang mengatur produksi zat perusak ozon yang mengandung klorin, ozon telah dipantau secara rutin dari dalam tanah dan dari luar angkasa.
    <br><b>7. Awan (Cloud)</b><br>
        Pengambilan properti cloud TROPOMI/S5P didasarkan pada algoritma OCRA dan ROCINN yang saat ini digunakan dalam operasional produk GOME dan GOME-2. OCRA mengambil fraksi awan menggunakan pengukuran di wilayah spektral UV/VIS dan ROCINN 
        mengambil tinggi awan (tekanan) dan ketebalan optik (albedo) menggunakan pengukuran di dalam dan sekitar pita oksigen A pada 760 nm. Algoritme versi 3.0 digunakan, yang didasarkan pada perlakuan yang lebih realistis terhadap awan sebagai 
        lapisan partikel penghambur cahaya yang seragam secara optik. Selain itu, parameter awan juga disediakan untuk model awan yang mengasumsikan awan tersebut merupakan batas refleksi Lambertian.
    Terdapat beberapa fitur tambahan pada data, berikut merupakan fitur yang kami gunakan besertea fitur tambahan untuk prediksi yaitu:
    <br><b>1. Latitude</b><br>
        Latitude adalah garis yang horizontal / mendatar. Titik 0 adalah sudut ekuator, tanda + menunjukan arah ke atas menuju kutub utara, sedangkan tanda minus di koordinat Latitude menuju ke kutub selatan.
        Titik yang dipakai dari 0 ke 90 derajat ke arah kutub utara, dan 0 ke -90 derajat ke kutub selatan
    <br><b>2. Longitude</b><br>
        Longitude adalah garis lintang . Angka dari sudut bundar bumi horizontal. Titik diawali dari 0 ke 180 derajat, dan 0 ke-180 ke arah sebaliknya.
    <br><b>3. Year</b><br>
        
    <br><b>4. Syclic Feature (Week Sin dan Week Cos)</b><br>
        Sinus dan cosinus, untuk mencerminkan siklus, misalnya tanggal 1 Januari mendekati tanggal 31 Desember.
    <br><b>6. Holidays</b><br>
        Emisi sangat bergantung pada tren musiman/liburan, oleh karena itu kami menambahkan fitur hokidays
    <br><b>7. Rotate Location (Rot_15_x, Rot_15_y, Rot_30_x, Rot_30_y)</b><br>
        Rotasi lokasi mengubah perspektif data dan menemukan pola atau hubungan baru yang tidak terlihat dalam data asli.
    <br><b>8. Distance_to_max_emmission</b><br>
        Menghitung jarak dari setiap lokasi ke lokasi dengan emisi tertinggi menggunakan fungsi haversine,
        yang merupakan cara untuk menghitung jarak antara dua titik di permukaan bumi
    """
    st.write(desc, unsafe_allow_html=True)

# Tab EDA
with eda:
    st.markdown("<h4 style='text-align: center;'>Data Train CO2 Emission</h4>", unsafe_allow_html=True)
    st.write(train.head())  # Menampilkan data asli

    train_eda = train.copy()

    # Menghitung missing values dan mengkonversinya ke dalam persentase
    missValTrain = train_eda.isnull().sum()
    missValTrain = missValTrain[missValTrain > 0].sort_values(ascending=False).head(20)
    missValTrain = missValTrain / len(train) * 100

    # Membuat visualisasi
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=missValTrain.values, y=missValTrain.index, palette='viridis', ax=ax)

    # Menambahkan label dan judul
    ax.set_xlabel('Persentase Missing Value (%)')
    ax.set_ylabel('Kolom')
    ax.set_title('Persentase Missing Value')

    # Menampilkan visualisasi di Streamlit
    st.markdown("<h3 style='text-align: center;'>Presentase Missing Value</h3>", unsafe_allow_html=True)
    st.pyplot(fig)

    # DISTRIBUSI EMISI
    # Menampilkan judul
    st.markdown("<h3 style='text-align: center;'>Distribusi Emisi CO2</h3>", unsafe_allow_html=True)

    # Membuat plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot histogram
    train_eda['emission'].plot(kind="hist", density=True, alpha=0.65, bins=50, color='skyblue', edgecolor='black', ax=ax)

    # Plot KDE
    train_eda['emission'].plot(kind="kde", color='darkblue', ax=ax)

    # Quantile lines
    quant_50, quant_95 = train['emission'].quantile(0.5), train['emission'].quantile(0.95)
    quants = [[quant_50, 1, 0.36], [quant_95, 0.6, 0.56]]
    for i in quants:
        ax.axvline(i[0], alpha=i[1], ymax=i[2], linestyle=":", color='red')

    # X
    ax.set_xlabel('Emission')
    ax.set_xlim((0, 3200))

    # Y
    ax.set_ylabel('Distribution of emissions')

    # Annotations
    ax.text(quant_50, .0037, "50th", size=11, alpha=.85)
    ax.text(quant_95, .0056, "95th Percentile", size=10, alpha=.8)

    # Title
    ax.set_title('Distribution of emissions', size=15, pad=10)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)
    

    #BOXPLOT DISTRIBUSI EMISI PER TAHUN
    st.markdown("<h3 style='text-align: center;'>Boxplot Emisi CO2 per Tahun</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x="year", y="emission", data=train_eda, palette="Set3", ax=ax)
    ax.set_title("Distribusi Emisi CO2 per Tahun")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Emisi CO2")
    st.pyplot(fig)

    #LATTITUDE VS EMISI
    st.markdown("<h3 style='text-align: center;'>Scatter Plot Latitude vs Emisi CO2</h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(x="latitude", y="emission", data=train_eda, palette="viridis", ax=ax)
    ax.set_title("Latitude vs Emisi CO2")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Emisi CO2")
    st.pyplot(fig)


    # plot train per week of year
    # Konversi kolom 'year' dan 'week_no' menjadi kolom 'date'
    st.sidebar.markdown("<h4 style='text-align: center;'>Emisi Per Minggu Dalam 3 Tahun</h4>", unsafe_allow_html=True)
    train_plot = train.copy(deep=True)
    train_plot['date'] = pd.to_datetime(train_plot['year'].astype(str) + '-' + train_plot['week_no'].astype(str) + '-1', format='%Y-%W-%w')

    # Slider untuk memilih rentang tahun dengan key unik
    start_year, end_year = st.sidebar.slider('Select Year Range for Week of Year', 2019, 2021, (2019, 2021), key='year_range_slider')

    # Filter berdasarkan tahun yang dipilih
    train_plot = train_plot[(train_plot['year'] >= start_year) & (train_plot['year'] <= end_year)]

    # Menyusun data untuk plotting
    avg_week = train_plot.groupby(['year', 'week_no'])['emission'].mean().reset_index()

    # Membuat line plot dengan warna berdasarkan tahun
    fig, ax = plt.subplots(figsize=(18, 10))
    palette = sns.color_palette('husl', n_colors=len(avg_week['year'].unique()))

    for i, year in enumerate(sorted(avg_week['year'].unique())):
        yearly_data = avg_week[avg_week['year'] == year]
        sns.lineplot(data=yearly_data, x='week_no', y='emission', label=str(year), color=palette[i], marker='o', ax=ax)

    # Menyesuaikan tampilan plot
    ax.set_title('Average emissions per week of year')
    ax.set_xlabel('Week of the year')
    ax.set_ylabel('Average emissions')
    ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    plt.tight_layout()
    st.markdown("<h3 style='text-align: center;'>Plot Emisi Per Minggu Dalam 3 Tahun</h3>", unsafe_allow_html=True)
    st.pyplot(fig)


    # Plot emisi untuk lokasi tertentu sepanjang tahun
    st.sidebar.markdown("<h4 style='text-align: center;'>Emisi CO2 untuk Lokasi Tertentu Sepanjang Tahun</h4>", unsafe_allow_html=True)
    # Membulatkan latitude dan longitude
    train_eda['latitude'] = round(train_eda['latitude'], 2)
    train_eda['longitude'] = round(train_eda['longitude'], 2)

    # Sampling lokasi tertentu
    sample_loc = train_processed[(train_eda['latitude'] == -0.51) & (train_eda['longitude'] == 29.29)]

    # Slider untuk memilih rentang tahun
    start_year, end_year = st.sidebar.slider('Pilih Rentang Tahun untuk Lokasi', 2019, 2021, (2021, 2021))

    # Filter berdasarkan tahun yang dipilih
    filtered_sample_loc = sample_loc[(sample_loc['year'] >= start_year) & (sample_loc['year'] <= end_year)]

    # Membuat line plot
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.suptitle(f'Emisi CO2 untuk lokasi lat -0.51 lon 29.29 dari {start_year} hingga {end_year}', y=1.02, fontsize=15)

    for year in range(start_year, end_year + 1):
        df = filtered_sample_loc[filtered_sample_loc['year'] == year]
        sns.lineplot(x='week_no', y='emission', data=df, ax=ax, label=str(year))

    ax.legend(title='Year')
    plt.tight_layout()
    st.markdown("<h3 style='text-align: center;'>Emisi CO2 untuk Lokasi Tertentu Sepanjang Tahun</h3>", unsafe_allow_html=True)
    st.pyplot(fig)

  
#Tab Prediksi
with prediksi:
    train_copy = train.copy()

    # Feature Engineering pada salinan data
    train_copy['date'] = pd.to_datetime(
        train_copy['year'].astype(str) + '-' + train_copy['week_no'].astype(str) + '-1',
        format='%Y-%W-%w'
    )

    drop_columns = ['ID_LAT_LON_YEAR_WEEK','emission', 'date']
    if 'ID' in train_copy.columns:
        drop_columns.append('ID')

    train_copy.drop(columns=['UvAerosolLayerHeight_aerosol_pressure', 'UvAerosolLayerHeight_solar_zenith_angle', 
                        'UvAerosolLayerHeight_aerosol_height', 'UvAerosolLayerHeight_aerosol_optical_depth', 
                        'UvAerosolLayerHeight_sensor_zenith_angle', 'UvAerosolLayerHeight_sensor_azimuth_angle', 
                        'UvAerosolLayerHeight_solar_azimuth_angle'], inplace=True, errors='ignore')

    avg_emission_non_virus = train_copy[train_copy['year'].isin([2019, 2021])].groupby('week_no')['emission'].mean()
    avg_emission_virus = train_copy[train_copy['year'] == 2020].groupby('week_no')['emission'].mean()
    ratios_for_weeks = avg_emission_non_virus / avg_emission_virus
    train_copy.loc[train_copy['year'] == 2020, 'emission'] *= train_copy['week_no'].map(ratios_for_weeks)
    train_copy.loc[(train_copy['week_no'] == 52) & (train_copy['year'] == 2020), 'emission'] = np.power(
        train_copy.loc[(train_copy['week_no'] == 52) & (train_copy['year'] == 2020), 'emission'], 
        1/1.5
    )
    train_copy['season'] = train_copy['date'].dt.month.apply(lambda x: 1 if 3 <= x <= 5 else 2 if 6 <= x <= 8 else 3 if 9 <= x <= 11 else 4)
    train_copy['holidays'] = train_copy['week_no'].isin([0, 51, 12, 30])
    train_copy['week_sin'] = np.sin(train_copy['week_no'] * (2 * np.pi / 52))
    train_copy['week_cos'] = np.cos(train_copy['week_no'] * (2 * np.pi / 52))

    def rotate_coordinates(df, angle):
        rot_matrix = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        r = rot_matrix(np.deg2rad(angle))
        return df.apply(lambda x: r.dot([x['longitude'], x['latitude']]), axis=1, result_type='expand')

    train_copy[['rot_15_x', 'rot_15_y']] = rotate_coordinates(train_copy, 15)
    train_copy[['rot_30_x', 'rot_30_y']] = rotate_coordinates(train_copy, 30)

    max_emission_location = train_copy.loc[train_copy['emission'].idxmax(), ['latitude', 'longitude']]
    train_copy['distance_to_max_emission'] = np.sqrt(
        (train_copy['latitude'] - max_emission_location['latitude']) ** 2 +
        (train_copy['longitude'] - max_emission_location['longitude']) ** 2
    )

    training_cols = ['latitude', 'longitude', 'year', 'week_sin', 'week_cos', 'holidays', 
                    'rot_15_x', 'rot_15_y', 'rot_30_x', 'rot_30_y', 'distance_to_max_emission']

    # Memuat model dari file
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Form untuk input data pengguna
    st.markdown("<h3 style='text-align: center;'>Prediksi Emisi CO2</h3>", unsafe_allow_html=True)

    def user_input_features():
        latitude = st.number_input("Latitude" "-0.51 s/d -3.299", min_value=-3.299, max_value=-0.51, value=-3.299)
        longitude = st.number_input("Longitude" " 29.29 s/d 30.301", min_value=29.29, max_value=30.301, value=29.29)
        year = st.selectbox("Year", [2019, 2020, 2021, 2022])
        week_no = st.number_input("Week Number", min_value=1, max_value=52, value=1)

        user_data = {
            'latitude': latitude,
            'longitude': longitude,
            'year': year,
            'week_no': week_no
        }

        user_data['week_sin'] = np.sin(user_data['week_no'] * (2 * np.pi / 52))
        user_data['week_cos'] = np.cos(user_data['week_no'] * (2 * np.pi / 52))
        user_data['holidays'] = user_data['week_no'] in [0, 51, 12, 30]

        return pd.DataFrame(user_data, index=[0])

    input_df = user_input_features()

    input_df[['rot_15_x', 'rot_15_y']] = rotate_coordinates(input_df, 15)
    input_df[['rot_30_x', 'rot_30_y']] = rotate_coordinates(input_df, 30)

    max_emission_location = train_copy.loc[train_copy['emission'].idxmax(), ['latitude', 'longitude']]
    input_df['distance_to_max_emission'] = np.sqrt(
        (input_df['latitude'] - max_emission_location['latitude']) ** 2 +
        (input_df['longitude'] - max_emission_location['longitude']) ** 2
    )

    input_features = input_df[training_cols]

    # Prediksi
    if st.button('Predict'):
        prediction = model.predict(input_features)
        st.write(f'Prediksi Emisi CO2: {prediction[0]:.2f}')
