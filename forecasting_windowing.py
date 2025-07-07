import pandas as pd
import matplotlib.pyplot as plt

# Load dataset dari statsmodels
from statsmodels.datasets import get_rdataset
air = get_rdataset('AirPassengers', package='datasets').data

# Rename kolom dan konversi waktu
df = air.copy()
df.columns = ['Bulan', 'Penumpang']
df['Bulan'] = pd.date_range(start='1949-01', periods=len(df), freq='ME')

# Moving average window = 12 (setahun)
df['MA_12'] = df['Penumpang'].rolling(window=12).mean()

# Visualisasi
plt.figure(figsize=(12, 5))
plt.plot(df['Bulan'], df['Penumpang'], label='Penumpang Asli')
plt.plot(df['Bulan'], df['MA_12'], label='Moving Average (12)', color='orange')
plt.title('Forecasting Penumpang Pesawat dengan Moving Average')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Penumpang')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
