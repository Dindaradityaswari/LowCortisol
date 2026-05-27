from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

app = FastAPI(
    title="MindCheck Burnout Prediction API",
    description="API Produksi untuk mendeteksi tingkat burnout karyawan",
    version="1.0"
)

# ... [Pertahankan class EmployeeInput persis seperti sebelumnya] ...
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
        self.w = self.add_weight(
            name='feature_weight',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        super(SmartFeatureAttention, self).build(input_shape)
    
    def call(self, inputs):
        activated_weights = tf.nn.sigmoid(self.w)
        return inputs * activated_weights

# =========================================================================
# LOAD ASET AI: REKONSTRUKSI MANUAL
# =========================================================================
print("Sedang merakit ulang arsitektur AI...")
try:
    loaded_encoder = joblib.load('onehot_encoder.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_target_le = joblib.load('le_target.pkl')
    loaded_feature_names = joblib.load('feature_names.pkl')

    # 1. Bangun arsitektur kosong secara manual
    # Pastikan shape inputnya sama dengan hasil preprocessingmu (58 fitur)
    input_layer = tf.keras.layers.Input(shape=(58,), name='input_features')
    x = SmartFeatureAttention(name='attention_layer')(input_layer)
    
    x = tf.keras.layers.Dense(64, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_6')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_6')(x)
    x = tf.keras.layers.Activation('relu', name='activation_6')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout_6')(x)
    
    x = tf.keras.layers.Dense(32, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001), name='dense_7')(x)
    x = tf.keras.layers.BatchNormalization(name='batch_normalization_7')(x)
    x = tf.keras.layers.Activation('relu', name='activation_7')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_7')(x)
    
    output_layer = tf.keras.layers.Dense(3, activation='softmax', name='output_layer')(x)

    loaded_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # 2. Suntikkan bobot yang sudah di-training
    loaded_model.load_weights("model_burnout_weights.weights.h5")
    print("Arsitektur dan bobot berhasil dimuat!")

except Exception as init_err:
    print(f"GAGAL MEMUAT ASET AI: {str(init_err)}")
    loaded_model = None
# =========================================================================

@app.get("/")
def read_root():
    return {"status": "online", "message": "API Aktif."}

@app.post("/predict")
def predict_burnout(input_data: EmployeeInput):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model gagal dirakit. Cek logs.")
        
    try:
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

        prediction_prob = loaded_model.predict(input_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction_prob, axis=1)[0]
        confidence_score = np.max(prediction_prob) * 100

        label_burnout = loaded_target_le.inverse_transform([predicted_class_idx])[0]

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
        raise HTTPException(status_code=500, detail=str(e))