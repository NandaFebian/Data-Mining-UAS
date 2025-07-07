import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os

# Matplotlib style
sns.set_theme(style="darkgrid")  # atau style lain

# =============================
# Function: Moving Average Forecast
# =============================
def run_moving_average():
    output_text.delete('1.0', tk.END)  # ✅ Hapus teks sebelumnya

    from statsmodels.datasets import get_rdataset
    df = get_rdataset('AirPassengers', package='datasets').data
    df.columns = ['Bulan', 'Penumpang']
    df['Bulan'] = pd.date_range(start='1949-01', periods=len(df), freq='ME')
    df['MA_12'] = df['Penumpang'].rolling(window=12).mean()

    show_text("\n[Forecasting] 5 data terakhir:\n")
    show_text(df.tail().to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df['Bulan'], df['Penumpang'], label='Penumpang Asli')
    ax.plot(df['Bulan'], df['MA_12'], label='Moving Average (12)', linestyle='--')
    ax.set_title('Moving Average Forecasting')
    ax.set_xlabel('Bulan')
    ax.set_ylabel('Jumlah Penumpang')
    ax.legend()
    show_plot(fig)

# =============================
# Function: Apriori Association
# =============================
def run_apriori():
    output_text.delete('1.0', tk.END)  # ✅ Hapus teks sebelumnya

    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    transactions = [
        ['susu', 'roti', 'mentega'],
        ['roti', 'bir', 'popok', 'telur'],
        ['susu', 'popok', 'bir', 'cola'],
        ['susu', 'roti', 'popok', 'bir'],
        ['roti', 'susu', 'popok'],
        ['susu', 'roti', 'keju'],
        ['roti', 'keju', 'bir'],
        ['roti', 'susu', 'keju', 'popok'],
        ['roti', 'keju', 'susu'],
        ['roti', 'susu', 'popok']
    ]

    te = TransactionEncoder()
    df = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    show_text("\n[Apriori] Frequent Itemsets:\n")
    show_text(frequent_itemsets.to_string(index=False))

    show_text("\n[Apriori] Association Rules:\n")
    if rules.empty:
        show_text("Tidak ditemukan aturan yang memenuhi.")
    else:
        show_text(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string(index=False))

# =============================
# Function: K-Means Clustering
# =============================
def run_kmeans():
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # fix error
    output_text.delete('1.0', tk.END)  # ✅ Hapus teks sebelumnya

    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    X['Cluster'] = labels

    show_text("\n[K-Means] Sampel hasil clustering:\n")
    show_text(X[['sepal length (cm)', 'sepal width (cm)', 'Cluster']].head().to_string(index=False))

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.iloc[:, :-1])

    fig, ax = plt.subplots(figsize=(6, 3))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    ax.set_title("K-Means Clustering (Iris PCA)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    show_plot(fig)

# =============================
# Helper: Show Output Text
# =============================
def show_text(msg):
    output_text.insert(tk.END, msg + '\n')
    output_text.see(tk.END)

# Helper: Show Plot in UI
def show_plot(fig):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# =============================
# UI Setup
# =============================
app = tk.Tk()
app.title("Mini Data Mining App")
app.geometry("900x600")

style = ttk.Style(app)
style.configure("TButton", font=("Segoe UI", 10), padding=6)

# Button Panel
button_frame = tk.Frame(app)
button_frame.pack(pady=10)

ttk.Button(button_frame, text="Forecasting (Moving Average)", command=run_moving_average).pack(side=tk.LEFT, padx=10)
ttk.Button(button_frame, text="Asosiasi (Apriori)", command=run_apriori).pack(side=tk.LEFT, padx=10)
ttk.Button(button_frame, text="Clustering (K-Means)", command=run_kmeans).pack(side=tk.LEFT, padx=10)

# Output Text
output_text = ScrolledText(app, height=15, font=("Consolas", 10))
output_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)

# Plot Area
plot_frame = tk.Frame(app)
plot_frame.pack(fill=tk.BOTH, expand=True)

app.mainloop()
