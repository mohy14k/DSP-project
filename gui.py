import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os

# Functions from the provided code

def get_values_from_line(line):
    values_str = line.split(sep="|")
    values = [float(values_str[i]) for i in range(len(values_str) - 1)]
    return values

def get_ECG_data(file_path):
    with open(file_path, 'r') as data_file:
        data = [get_values_from_line(line) for line in data_file.readlines()]
    return data

def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=360, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sp.butter(order, Wn=[low, high], btype='band')
    return sp.filtfilt(b, a, data)

def get_filtered_data(data, lowcut=0.5, highcut=40.0, fs=360, order=4):
    return [bandpass_filter(beat, lowcut, highcut, fs, order) for beat in data]

def Normalize_Signal(data, check):
    normalized_data = []
    for y in data:
        min_element = min(y)
        max_element = max(y)
        if check == 0:
            y_normalized = [(x - min_element) / (max_element - min_element) for x in y]
        else:
            y_normalized = [2 * (x - min_element) / (max_element - min_element) - 1 for x in y]
        normalized_data.append(y_normalized)
    return normalized_data

def wavelet_features(signal):
    coeffs = pywt.wavedec(signal, wavelet='db4', level=4)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff), np.sum(np.square(coeff)), np.max(coeff), np.min(coeff), np.max(coeff) - np.min(coeff)])
    return np.array(features)

def features_data(data):
    return np.array([wavelet_features(signal) for signal in data])

# Load pre-trained KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Training sample for demonstration (to be replaced with actual training data)
def train_knn():
    normal_train = np.random.randn(100, 360)  # Placeholder for Normal samples
    pvc_train = np.random.randn(100, 360)    # Placeholder for PVC samples

    normal_train_filtered_nor = Normalize_Signal(get_filtered_data(normal_train), check=1)
    pvc_train_filtered_nor = Normalize_Signal(get_filtered_data(pvc_train), check=1)

    train_data = np.concatenate((normal_train_filtered_nor, pvc_train_filtered_nor))
    train_labels = np.array([0] * len(normal_train_filtered_nor) + [1] * len(pvc_train_filtered_nor))
    train_data, train_labels = shuffle(train_data, train_labels, random_state=42)

    features_train = features_data(train_data)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)

    knn.fit(features_train, train_labels)
    return scaler

scaler = train_knn()

# GUI Functions
def upload_file():
    file_path = filedialog.askopenfilename(title="Data\\Normal&PVC\\Normal_Train.txt", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return

    try:
        # Process the ECG file
        ecg_data = get_ECG_data(file_path)
        filtered_data = get_filtered_data(ecg_data)
        normalized_data = Normalize_Signal(filtered_data, check=1)
        features = features_data(normalized_data)
        features = scaler.transform(features)

        # Predict with KNN
        predictions = knn.predict(features)
        result = "Normal" if np.mean(predictions) < 0.5 else "PVC (Disease)"

        messagebox.showinfo("Result", f"The patient is classified as: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")

def plot_sample():
    file_path = filedialog.askopenfilename(title="Data\\Normal&PVC\\Normal_Train.txt", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return

    try:
        # Display the first sample in the ECG file
        ecg_data = get_ECG_data(file_path)
        plt.plot(ecg_data[0])
        plt.title("ECG Signal (First Sample)")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while plotting: {str(e)}")

def load_signal():
    file_path = filedialog.askopenfilename(title="Load ECG Signal", filetypes=[("Text Files", "*.txt")])
    if not file_path:
        return

    try:
        # Load and plot the ECG signal
        ecg_data = get_ECG_data(file_path)
        plt.plot(ecg_data[0])
        plt.title("Loaded ECG Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the ECG signal: {str(e)}")

# Create GUI
root = tk.Tk()
root.title("ECG Classification")

# Set background color to black
root.configure(bg="#000000")

frame = tk.Frame(root, padx=20, pady=20, bg="#000000")
frame.pack(padx=10, pady=10)

# Title Label
label = tk.Label(frame, text="ECG Heartbeat Classification", font=("Arial", 20, "bold"), bg="#000000", fg="#00FF00")
label.grid(row=0, column=0, columnspan=2, pady=20)

# Upload ECG File Button
upload_button = tk.Button(frame, text="Upload ECG File", command=upload_file, width=20, height=2, relief="solid", bg="#00FF00", fg="black", font=("Arial", 12, "bold"), bd=2)
upload_button.grid(row=1, column=0, pady=10)

# Plot Sample Signal Button
plot_button = tk.Button(frame, text="Plot Sample Signal", command=plot_sample, width=20, height=2, relief="solid", bg="#00FF00", fg="black", font=("Arial", 12, "bold"), bd=2)
plot_button.grid(row=2, column=0, pady=10)

# Load Signal Button
load_button = tk.Button(frame, text="Load Signal", command=load_signal, width=20, height=2, relief="solid", bg="#00FF00", fg="black", font=("Arial", 12, "bold"), bd=2)
load_button.grid(row=3, column=0, pady=10)

# Exit Button
exit_button = tk.Button(frame, text="Exit", command=root.quit, width=20, height=2, relief="solid", bg="#FF0000", fg="white", font=("Arial", 12, "bold"), bd=2)
exit_button.grid(row=4, column=0, pady=10)

# Add rounded corners to buttons using a Canvas widget
def round_rectangle(x1, y1, x2, y2, r=25, **kwargs):
    """Function to draw a rounded rectangle with a specified corner radius."""
    points = [x1 + r, y1, x2 - r, y1, x2, y1 + r, x2, y2 - r, x2 - r, y2, x1 + r, y2, x1, y2 - r, x1, y1 + r]
    return root.create_polygon(points, **kwargs, smooth=True)

# Run the application
root.mainloop()
