from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = FastAPI(
    title="MindCheck Burnout Prediction API",
    description="API Produksi untuk mendeteksi tingkat burnout karyawan menggunakan Deep Learning",
    version="1.0"
)

class EmployeeInput(BaseModel):
    jenis_kelamin: str
    usia: int
    pendidikan_terakhir: str
    status_pernikahan: str
    departemen: str
    lama_bekerja_tahun: int
    tipe_perusahaan: str
    status_wfh: str
    jam_kerja_per_hari: int
    jam_lembur_per_hari: int
    jam_tidur_per_hari: int
    kualitas_tidur: int
    frekuensi_olahraga_per_minggu: int
    jam_layar_per_hari: int
    tingkat_stres: int
    kepuasan_kerja: int
    work_life_balance: int
    produktivitas_diri: int
    dukungan_atasan: int
    frekuensi_meeting_per_hari: int
    jumlah_deadline_per_minggu: int
    beban_kerja_persepsi: str
    status_merokok: str
    riwayat_kesehatan_mental: str
    keluhan_fisik_utama: str
    keamanan_pekerjaan: str
    frekuensi_konflik_kerja: int

@tf.keras.utils.register_keras_serializable()
class SmartFeatureAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SmartFeatureAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # PERBAIKAN: Ubah shape menjadi (input_shape[-1],) agar cocok menjadi bentuk (58,)
        self.w = self.add_weight(
            name='feature_weight',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super(SmartFeatureAttention, self).build(input_shape)
    
    def call(self, inputs):
        # Sesuaikan fungsi perkaliannya dengan format vektor 1 dimensi yang diactivate pakai sigmoid
        activated_weights = tf.nn.sigmoid(self.w)
        return inputs * activated_weights

def production_smooth_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to MindCheck Burnout Prediction API! Kunjungi /docs untuk dokumentasi otomatis."
    }

# 4. Endpoint Prediksi
@app.post("/predict")
def predict_burnout(input_data: EmployeeInput):
    try:
        required_files = ['onehot_encoder.pkl', 'scaler.pkl', 'le_target.pkl', 'feature_names.pkl', 'model_burnout_production.keras']
        for file in required_files:
            if not os.path.exists(file):
                raise HTTPException(status_code=500, detail=f"File aset penting '{file}' tidak ditemukan di server.")

        loaded_encoder = joblib.load('onehot_encoder.pkl')
        loaded_scaler = joblib.load('scaler.pkl')
        loaded_target_le = joblib.load('le_target.pkl')
        loaded_feature_names = joblib.load('feature_names.pkl')

        df_input = pd.DataFrame([input_data.dict()])

        cat_cols = ['jenis_kelamin', 'pendidikan_terakhir', 'status_pernikahan', 'departemen', 
                    'tipe_perusahaan', 'status_wfh', 'beban_kerja_persepsi', 'status_merokok', 
                    'riwayat_kesehatan_mental', 'keluhan_fisik_utama', 'keamanan_pekerjaan']
        num_cols = [col for col in df_input.columns if col not in cat_cols]

        input_cat_encoded = loaded_encoder.transform(df_input[cat_cols])
        input_cat_df = pd.DataFrame(input_cat_encoded, columns=loaded_encoder.get_feature_names_out(cat_cols))
        input_final_df = pd.concat([df_input[num_cols].reset_index(drop=True), input_cat_df], axis=1)

        input_final_df = input_final_df.reindex(columns=loaded_feature_names, fill_value=0)

        input_scaled = loaded_scaler.transform(input_final_df)

        # Load Model Keras dengan mendaftarkan Custom Objects
        loaded_model = tf.keras.models.load_model(
            'model_burnout_production.keras',
            custom_objects={
                'SmartFeatureAttention': SmartFeatureAttention,
                'production_smooth_loss': production_smooth_loss
            }
        )

        # Lakukan Prediksi Nilai Probabilitas
        prediction_prob = loaded_model.predict(input_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction_prob, axis=1)[0]
        confidence_score = np.max(prediction_prob) * 100

        # Kembalikan Label Teks Asli (Low / Medium / High)
        label_burnout = loaded_target_le.inverse_transform([predicted_class_idx])[0]

        # Return Hasil Output API JSON
        return {
            "status": "success",
            "results": {
                "prediksi_level": label_burnout,
                "kepastian_ai": f"{confidence_score:.2f}%",
                "probabilitas_detail": {
                    str(loaded_target_le.classes_[i]): f"{prob*100:.2f}%" for i, prob in enumerate(prediction_prob[0])
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")