import pandas as pd
from sklearn.datasets import load_iris
from statsmodels.datasets import get_rdataset

# ==============================
# 1. Dataset Forecasting (AirPassengers)
# ==============================

print("Mendownload dataset AirPassengers...")
air = get_rdataset('AirPassengers', package='datasets').data
air.columns = ['Bulan', 'Penumpang']
air['Bulan'] = pd.date_range(start='1949-01', periods=len(air), freq='M')
air.to_csv('data_airpassengers.csv', index=False)
print("✅ data_airpassengers.csv berhasil disimpan.\n")

# ==============================
# 2. Dataset Asosiasi (Transaksi Retail)
# ==============================

print("Membuat data transaksi manual...")
transactions = [
    ['susu', 'roti', 'mentega'],
    ['roti', 'bir', 'popok', 'telur'],
    ['susu', 'popok', 'bir', 'cola'],
    ['susu', 'roti', 'popok', 'bir'],
    ['roti', 'susu', 'popok'],
    ['susu', 'roti', 'keju'],
    ['roti', 'keju', 'bir'],
    ['roti', 'susu', 'keju', 'popok']
]

# Simpan transaksi ke file CSV (satu baris = satu transaksi)
with open('data_transaksi.csv', 'w') as f:
    f.write("Transaksi\n")
    for row in transactions:
        f.write(",".join(row) + "\n")
print("✅ data_transaksi.csv berhasil disimpan.\n")

# ==============================
# 3. Dataset Clustering (Iris)
# ==============================

print("Mendownload dataset Iris...")
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
df_iris.to_csv('data_iris.csv', index=False)
print("✅ data_iris.csv berhasil disimpan.\n")

print("SEMUA DATASET BERHASIL DISIMPAN ✅")
