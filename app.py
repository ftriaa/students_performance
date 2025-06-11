import streamlit as st
import joblib
import pandas as pd

# Load model dan tools preprocessing
model = joblib.load(open('model/model_xgb.pkl', 'rb'))
encoder = joblib.load(open('model/label_encoder.pkl', 'rb'))
scaler = joblib.load(open('model/robust_scaler.pkl', 'rb'))
feature_names = joblib.load(open('model/feature_names.pkl', 'rb'))

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="ftriaanggra23-dashboard/logo.png", 
    layout="wide")

# Custom CSS untuk warna latar belakang dan styling
st.markdown("""
<style>
/* Warna latar aplikasi */
body, .stApp {
    background-color: #e6f0ff;
    color: #333333;
}

/* Warna teks umum */
h1, h2, h3, h4, h5, h6, p, label, .stTextInput label, .stSelectbox label, 
.stNumberInput label, .stRadio label {
    color: #333333 !important;
}

/* Styling semua input (teks, angka) jadi putih */
input[type="text"], input[type="number"], textarea {
    background-color: white !important;
    color: black !important;
    border: none !important;
    border-radius: 5px !important;
    box-shadow: none !important;
    outline: none !important;
}                       

/* Styling untuk st.selectbox (dropdown) */
div[data-baseweb="select"] {
    background-color: white !important;
    color: black !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 5px !important;
}

/* Styling kotak dropdown ketika nilai sudah dipilih */
div[data-baseweb="select"] > div[role="combobox"] {
    background-color: white !important;
    color: black !important;
    border-radius: 5px !important;
}

/* Pastikan teks nilai yang dipilih juga hitam */
div[data-baseweb="select"] > div[role="combobox"] > div {
    color: black !important;
}

/* Hilangkan shadow atau border aneh saat terpilih */
div[data-baseweb="select"] > div[role="combobox"]:focus,
div[data-baseweb="select"] > div[role="combobox"]:hover {
    box-shadow: none !important;
    border: none !important;
}

/* Styling dropdown popup list item */
div[role="listbox"] {
    background-color: #222222 !important;
    color: white !important;
    border-radius: 5px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.4) !important;
    z-index: 1000;
}

/* Styling tiap item list dropdown */
div[role="option"] {
    color: white !important;
    background-color: #222222 !important;
    padding: 8px 12px !important;
    cursor: pointer;
}

/* Hover item dropdown */
div[role="option"]:hover {
    background-color: #555555 !important;
    color: white !important;
}

/* Styling tombol */
button[kind="primary"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #333333 !important;
    border-radius: 5px !important;
    box-shadow: none !important;
}

div.stButton > button {
    background-color: white !important;
    color: white !important;
    border-radius: 5px !important;
    border: 1px solid #333333 !important;
    padding: 8px 16px !important;
    font-weight: bold !important;
}

div.stButton > button:hover {
    background-color: #0056b3 !important;
    color: white !important;
}

/* Styling radio button text */
.css-1r6slb0 p {
    color: #333333 !important;
}

/* Hilangkan footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Show logo and title
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("ftriaanggra23-dashboard/logo.png", width=80)
with col2:
    st.markdown(
        "<h1 style='margin-top: 5px;'>Prediksi Dropout Mahasiswa</h1>",
        unsafe_allow_html=True
    )

st.markdown("""
**Aplikasi ini memprediksi status mahasiswa, apakah *Dropout*, *Graduate*, atau tetap *Enrolled*.**

Masukkan data mahasiswa di bawah ini, dan dapatkan prediksi statusnya.
""")

# Menu navigasi
tab1, tab2 = st.tabs(["📘 Latar Belakang", "📊 Prediksi Dropout"])

# === Bagian 1: Latar Belakang ===
with tab1:
    st.header("Latar Belakang")
    st.write("""
        Dropout siswa merupakan tantangan serius dalam dunia pendidikan. 
        Dengan memanfaatkan data siswa dan model machine learning, kita dapat memprediksi kemungkinan siswa akan dropout. 
        Hal ini memungkinkan pihak institusi mengambil tindakan preventif lebih awal untuk meningkatkan retensi siswa.
    """)
    st.subheader("Tujuan Aplikasi")
    st.markdown("""
    - Menganalisis faktor-faktor yang mempengaruhi siswa keluar (dropout).
    - Memberikan prediksi kemungkinan siswa akan dropout berdasarkan input data.
    - Membantu pengambilan keputusan oleh manajemen Jaya Jaya Institut.
    """)

# === Bagian 2: Prediksi Dropout ===
with tab2:
    st.header("Formulir Prediksi Dropout Siswa")

    student_name = st.text_input("Nama Mahasiswa")

    # Mapping label untuk fitur kategorikal
    categorical_map = {"Ya": 1, "Tidak": 0, "Laki-laki": 1, "Perempuan": 0}

    application_mode_map = {
        1: "1st phase - general contingent",
        2: "Ordinance No. 612/93",
        5: "1st phase - special contingent (Azores)",
        7: "Holders of other higher courses",
        10: "Holders of other courses",
        15: "1st phase - special contingent (Madeira)",
        16: "International student",
        17: "2nd phase - general contingent",
        18: "3rd phase - general contingent",
        26: "Ordinance No. 854-B/99",
        27: "Ordinance No. 533-A/99, item b2",
        39: "Over 23 years old",
        42: "Transfer",
        43: "Change of course",
        44: "Technological specialization diploma holders",
        51: "Change of institution/course",
        53: "Short cycle diploma holders",
        57: "Change of institution/course (International)"
    }

    mothers_occupation_map = {
        0: 'Student',
        1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
        2: 'Specialists in Intellectual and Scientific Activities', 
        3: 'Intermediate Level Technicians and Professions',
        4: 'Administrative staff',
        5: 'Personal Services, Security and Safety Workers and Sellers',
        6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
        7: 'Skilled Workers in Industry, Construction and Craftsmen',
        8: 'Installation and Machine Operators and Assembly Workers',
        9: 'Unskilled Workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: 'Blank',
        122: 'Health professionals',
        123: 'Teachers',
        125: 'Specialists in information and communication technologies (ICT)',
        131: 'Intermediate level science and engineering technicians and professions',
        132: 'Technicians and professionals, of intermediate level of health',
        134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
        141: 'Office workers, secretaries in general and data processing operators',
        143: 'Data, accounting, statistical, financial services and registry-related operators',
        144: 'Other administrative support staff',
        151: 'Personal service workers',
        152: 'Sellers',
        153: 'Personal care workers and the like',
        171: 'Skilled construction workers and the like, except electricians',
        173: 'Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
        175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
        191: 'Cleaning workers',
        192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
        193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
        194: 'Meal preparation assistants'
    }

    fathers_occupation_map = {
        0: 'Student',
        1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
        2: 'Specialists in Intellectual and Scientific Activities',
        3: 'Intermediate Level Technicians and Professions',
        4: 'Administrative staff',
        5: 'Personal Services, Security and Safety Workers and Sellers',
        6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
        7: 'Skilled Workers in Industry, Construction and Craftsmen',
        8: 'Installation and Machine Operators and Assembly Workers',
        9: 'Unskilled Workers',
        10: 'Armed Forces Professions',
        90: 'Other Situation',
        99: 'Blank',
        101: 'Armed Forces Officers',
        102: 'Armed Forces Sergeants',
        103: 'Other Armed Forces personnel',
        112: 'Directors of administrative and commercial services',
        114: 'Hotel, catering, trade and other services directors',
        121: 'Specialists in the physical sciences, mathematics, engineering and related techniques',
        122: 'Health professionals',
        123: 'Teachers',
        124: 'Specialists in finance, accounting, administrative organization, public and commercial relations',
        131: 'Intermediate level science and engineering technicians and professions',
        132: 'Technicians and professionals, of intermediate level of health',
        134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
        135: 'Information and communication technology technicians',
        141: 'Office workers, secretaries in general and data processing operators',
        143: 'Data, accounting, statistical, financial services and registry-related operators',
        144: 'Other administrative support staff',
        151: 'Personal service workers',
        152: 'Sellers',
        153: 'Personal care workers and the like',
        154: 'Protection and security services personnel',
        161: 'Market-oriented farmers and skilled agricultural and animal production workers',
        163: 'Farmers, livestock keepers, horticulturists, and similar'
    }  

    course_map = {
        33: "Biofuel Production Technologies",
        171: "Animation and Multimedia Design",
        8014: "Social Service (evening)",
        9003: "Agronomy",
        9070: "Communication Design",
        9085: "Veterinary Nursing",
        9119: "Informatics Engineering",
        9130: "Equinculture",
        9147: "Management",
        9238: "Social Service",
        9254: "Tourism",
        9500: "Nursing",
        9556: "Oral Hygiene",
        9670: "Advertising and Marketing Management",
        9773: "Journalism and Communication",
        9853: "Basic Education",
        9991: "Management (evening)"
    }

    qualification_map = {
        1: "Secondary education",
        2: "Bachelor's degree",
        3: "Degree",
        4: "Master's degree",
        5: "Doctorate",
        9: "11th year schooling",
        10: "12th year schooling not completed",
        38: "Tech specialization course",
        39: "Professional higher technical course",
        40: "Post-Bologna degree",
        41: "Pre-Bologna degree",
        42: "Basic education 3rd cycle",
        43: "Basic education 2nd cycle"
    }

    # Input user
    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    avg_units_approved = st.number_input("Rata-rata mata kuliah lulus (per semester)", min_value=0.0, step=0.1)
    avg_units_grade = st.number_input("Rata-rata nilai mata kuliah", min_value=0.0, max_value=200.0, step=0.1)
    avg_units_evaluated = st.number_input("Rata-rata mata kuliah dievaluasi", min_value=0.0, step=0.1)
    admission_grade = st.number_input("Nilai penerimaan (0-200)", min_value=0.0, max_value=200.0, step=0.1)
    tuition_fees_up_to_date = st.radio("Status pembayaran UKT", ["Ya", "Tidak"])
    age_at_enrollment = st.number_input("Usia saat mendaftar", min_value=16, max_value=80)
    previous_qualification_grade = st.number_input("Nilai kualifikasi sebelumnya (0-200)", min_value=0.0, max_value=200.0, step=0.1)

    course_label = st.selectbox("Program Studi", list(course_map.values()))
    course = [k for k, v in course_map.items() if v == course_label][0]

    avg_units_enrolled = st.number_input("Rata-rata mata kuliah diambil", min_value=0.0, step=0.1)

    fathers_occupation_label = st.selectbox("Pekerjaan Ayah", list(fathers_occupation_map.values()))
    fathers_occupation = [k for k, v in fathers_occupation_map.items() if v == fathers_occupation_label][0]

    mothers_occupation_label = st.selectbox("Pekerjaan Ibu", list(mothers_occupation_map.values()))
    mothers_occupation = [k for k, v in mothers_occupation_map.items() if v == mothers_occupation_label][0]

    gdp = st.number_input("GDP Negara asal (ribu Euro)", min_value=0.0, step=0.1)
    unemployment_rate = st.number_input("Tingkat pengangguran (%)", min_value=0.0, max_value=100.0, step=0.1)

    father_qual_label = st.selectbox("Kualifikasi Ayah", list(qualification_map.values()))
    fathers_qualification = [k for k, v in qualification_map.items() if v == father_qual_label][0]

    application_mode_label = st.selectbox("Mode Aplikasi", list(application_mode_map.values()))
    application_mode = [k for k, v in application_mode_map.items() if v == application_mode_label][0]

    mother_qual_label = st.selectbox("Kualifikasi Ibu", list(qualification_map.values()))
    mothers_qualification = [k for k, v in qualification_map.items() if v == mother_qual_label][0]

    inflation_rate = st.number_input("Tingkat inflasi (%)", min_value=0.0, max_value=100.0, step=0.1)
    scholarship_holder = st.radio("Penerima Beasiswa", ["Ya", "Tidak"])
    application_order = st.slider("Urutan pilihan program studi", 0, 9)

    # Buat DataFrame input sesuai fitur model
    data_input = pd.DataFrame({
        'Gender': [categorical_map[gender]],
        'Avg_units_approved': [avg_units_approved],
        'Avg_units_grade': [avg_units_grade],
        'Avg_units_evaluated': [avg_units_evaluated],
        'Admission_grade': [admission_grade],
        'Tuition_fees_up_to_date': [categorical_map[tuition_fees_up_to_date]],
        'Age_at_enrollment': [age_at_enrollment],
        'Previous_qualification_grade': [previous_qualification_grade],
        'Course': [course],
        'Avg_units_enrolled': [avg_units_enrolled],
        'Fathers_occupation': [fathers_occupation],
        'Mothers_occupation': [mothers_occupation],
        'GDP': [gdp],
        'Unemployment_rate': [unemployment_rate],
        'Fathers_qualification': [fathers_qualification],
        'Application_mode': [application_mode],
        'Mothers_qualification': [mothers_qualification],
        'Inflation_rate': [inflation_rate],
        'Scholarship_holder': [categorical_map[scholarship_holder]],
        'Application_order': [application_order]
    })

    # Urutkan kolom sesuai feature_names
    data_input = data_input.reindex(columns=feature_names)

    # Skala data
    scaled_input = scaler.transform(data_input)

    if st.button("Prediksi Status Mahasiswa"):
        if student_name.strip() == "":
            st.warning("Silakan masukkan nama mahasiswa terlebih dahulu.")
        else:
            try:
                # Transformasi data
                scaled_input = scaler.transform(data_input)

                # Prediksi label
                pred = model.predict(scaled_input)
                label = encoder.inverse_transform(pred)[0]

                # Probabilitas semua kelas
                proba = model.predict_proba(scaled_input)
                proba_dict = dict(zip(encoder.classes_, proba[0]))
                persen_label = proba_dict[label] * 100

                # Tampilkan hasil prediksi
                st.markdown("### 🎯 Hasil Prediksi")
                st.success(f"Status mahasiswa bernama **{student_name}** diprediksi sebagai: **{label}**")

                # Tampilkan hanya probabilitas label hasil prediksi
                st.markdown("### 📊 Keyakinan Model")
                if label == "Graduate":
                    st.info(f"Peluang mahasiswa untuk **lulus** adalah: **{persen_label:.2f}%**")
                elif label == "Dropout":
                    st.warning(f"Kemungkinan mahasiswa mengalami **dropout** adalah: **{persen_label:.2f}%**")
                elif label == "Enrolled":
                    st.info(f"Mahasiswa masih **aktif** dengan peluang: **{persen_label:.2f}%**")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

# Footer hak cipta
st.markdown(
    """
    <div class='footer'>
        ©2025 ftriaanggra23
    </div>
    """,
    unsafe_allow_html=True
)