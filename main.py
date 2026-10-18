import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

# Konsistensi hasil acak (Random Seed)
np.random.seed(42)
tf.random.set_seed(42)

print('=== PIPELINE DEEP LEARNING: KLASIFIKASI ANOMALI INFRASTRUKTUR ===')

# ================= ================= =================
# 1. GENERATE DATASET SIMULASI FITUR EKSTRAKSI CITRA/SENSOR
# ================= ================= =================
n_samples = 4000
n_features = 32  # Representasi fitur hasil ekstraksi dari citra/data spasial aset

X = np.random.randn(n_samples, n_features)
# Target biner: 0 = Kondisi Aset Normal/Baik, 1 = Butuh Perbaikan/Anomali (~20%)
y = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])

print(f'Total Sampel Data: {n_samples} baris')
print(f'Proporsi Kelas Anomali (Butuh Perbaikan): {y.mean() * 100:.2f}%\n')

# ================= ================= =================
# 2. DATA SPLIT & PREPROCESSING
# ================= ================= =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= ================= =================
# 3. MEMBANGUN ARSITEKTUR DEEP LEARNING (KERAS)
# ================= ================= =================
model = models.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Mencegah overfitting
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output biner (0 atau 1)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Menghitung Class Weight untuk mengatasi ketidakseimbangan kelas data anomali
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print('Melatih model Deep Learning...')
history = model.fit(
    X_train, y_train,
    epochs=12,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# ================= ================= =================
# 4. EVALUASI MODEL
# ================= ================= =================
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob >= 0.5).astype(int)

print('\n--- LAPORAN EVALUASI MODEL DEEP LEARNING ---')
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Anomali/Perbaikan (1)']))