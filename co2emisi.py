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

train = load_data('C:\\MBKM\\project\\train.csv')

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
    mengandung sub fitur seperti column_number_density yang merupakan kerapatan kolom vertikal di permukaan tanah, yang dihitung dengan menggunakan teknik DOAS.
    Fitur-fitur yang kami gunakan untuk prediksi yaitu:
    1. Latitude
    2. Longitude
    3. Year
    4. Week Sin
    5. Week Cos
    6. Holidays
    7. Rot_15_x
    8. Rot_15_y
    9. Rot_30_x
    10. Rot_30_y
    11. Distance_to_max_emmission
    """
    st.write(desc)

# Tab EDA
with eda:
    st.markdown("<h4 style='text-align: center;'>Data Train CO2 Emission</h4>", unsafe_allow_html=True)
    st.write(train.head())  # Menampilkan data asli

    train_eda = train.copy()

    # Tampilkan peta menggunakan Streamlit
    # Mengelompokkan data pelatihan berdasarkan 'latitude' dan 'longitude' serta menghitung jumlah 'emission' untuk setiap lokasi
    grouped = train_eda.groupby(['latitude', 'longitude'])['emission'].sum().reset_index()

    # Membuat peta dengan skala warna linier yang memetakan nilai emisi ke warna
    colormap = cm.LinearColormap(['green', 'red'], vmin=0, vmax=75000)  # emisi di atas 75.000 akan diberi warna hitam

    # Membuat peta yang berpusat pada rata-rata 'latitude' dan 'longitude' dari titik-titik data
    m = folium.Map(location=[grouped['latitude'].mean(), grouped['longitude'].mean()])

    # Menambahkan tanda lingkaran pada peta untuk setiap titik dalam dataframe 'grouped'
    for _, row in grouped.iterrows():
        rows_emission = row['emission']
        color = 'blue' if rows_emission == 0 else colormap(rows_emission) if rows_emission < 10**5 else 'black'
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=np.sqrt(row['emission'])*15,
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)

    # Menyesuaikan peta dengan batas tanda
    m.fit_bounds(m.get_bounds())

    # Menambahkan legenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius:6px; padding: 10px;">
        <b>Legenda Emisi</b><br>
        <i class="fa fa-circle" style="color:blue"></i>&nbsp; Emisi Nol<br>
        <i class="fa fa-circle" style="color:green"></i>&nbsp; Emisi Rendah<br>
        <i class="fa fa-circle" style="color:red"></i>&nbsp; Emisi Sedang<br>
        <i class="fa fa-circle" style="color:black"></i>&nbsp; Emisi Tinggi<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Menampilkan peta dengan Streamlit
    st.markdown("<h3 style='text-align: center;'>Peta Emisi</h3>", unsafe_allow_html=True)
    st_folium(m, width=700, height=500)

    # plot train per week of year
    # Konversi kolom 'year' dan 'week_no' menjadi kolom 'date'
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