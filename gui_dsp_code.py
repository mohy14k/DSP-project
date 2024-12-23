# %% [markdown]
# import libs

# %%
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

# %% [markdown]
# here we will read ECG. each line repesent a single heart beat. after each sample there is "|"

# %%
def get_values_from_line(line:str):
    values_str = line.split(sep="|")#return return list of float string values
    
    values =[]
    for i in range(len(values_str) - 1):#len(values_str) -1 because the end entry is "" or "\n"
        values.append(float(values_str[i]))

    return values
   

# %%
def get_ECG_data(filePath:str):
    dataFile = open(filePath,'r')
    data = [[]] #I do this to return data in right way

    for line in dataFile.readlines():
        values = get_values_from_line(line)#get samples of each heart beat 
        data.append(values)

    dataFile.close()
    return data[1:]# because first entery is [] we don't need it in the data

# %%
def bandpass_filter(data, lowcut = 0.5, highcut= 40.0, fs = 360, order = 4 ):
    # Design Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sp.butter(order, Wn=[low, high], btype='band')

    # Apply the filter
    filtered_data = sp.filtfilt(b, a, data)
    return filtered_data


# %%
def get_filtered_data(data, lowcut = 0.5, highcut= 40.0,fs = 360, order = 4):
    filtered_data = []
    for beat in data:
        filtered_data.append(bandpass_filter(data=beat, lowcut=lowcut, highcut=highcut, fs=fs,order=order))
    return filtered_data

# %%
def Normalize_Signal(data, check):
    """
    Normalize all sublists in the input data to the range [0, 1] or [-1, 1].

    :param data: A list of lists containing signal data.
    :param check: Integer flag. If 0, normalize to [0, 1]. If 1, normalize to [-1, 1].
    :return: A list of normalized sublists.
    """
    normalized_data = []

    for y in data:
        # Get min and max values in the sublist
        min_element = min(y)
        max_element = max(y)

        if min_element == max_element:
            raise ValueError("Normalization not possible: all values in a sublist are the same.")

        # Normalize based on the check parameter
        if check == 0:
            y_normalized = [(x - min_element) / (max_element - min_element) for x in y]
        else:
            y_normalized = [2 * (x - min_element) / (max_element - min_element) - 1 for x in y]

        normalized_data.append(y_normalized)

    return normalized_data

# %% [markdown]
# import our data from files

# %%
normal_train = get_ECG_data(r"Data\Normal&PVC\Normal_Train.txt")

print(len(normal_train))
print(normal_train[0])
print(normal_train[-1])
print(len(normal_train[40]))
print(len(normal_train[-1]))


# %%
normal_test = get_ECG_data(r"Data\Normal&PVC\Normal_Test.txt")

print(len(normal_test))
print(normal_test[0])
print(normal_test[-1])


# %%
pvc_test = get_ECG_data(r"Data\Normal&PVC\PVC_Test.txt")

print(len(pvc_test))
print(pvc_test[0])
print(pvc_test[-1])


# %%
pvc_train = get_ECG_data(r"Data\Normal&PVC\PVC_Train.txt")

print(len(pvc_train))
print(pvc_train[0])
print(pvc_train[-1])


# %% [markdown]
# plot data before the filter

# %%
plt.plot(normal_train[0])

# %%
plt.plot(normal_test[0])

# %%
plt.plot(pvc_train[0])

# %%
plt.plot(pvc_test[0])

# %% [markdown]
# apply bandpass filter on the data

# %%
normal_train_filtered = get_filtered_data(normal_train)
normal_test_filtered = get_filtered_data(normal_test)
pvc_train_filtered = get_filtered_data(pvc_train)
pvc_test_filtered = get_filtered_data(pvc_test)

# %% [markdown]
# plot after bandpass filter

# %%
plt.plot(normal_train_filtered[0])

# %%
plt.plot(normal_test_filtered[0])

# %%
plt.plot(pvc_train_filtered[0])

# %%
plt.plot(pvc_test_filtered[0])

# %% [markdown]
# apply normalize the data

# %%
normal_train_filtered_nor = Normalize_Signal(normal_train_filtered, check=1)  # Use check=1 for [-1, 1] normalization
normal_test_filtered_nor = Normalize_Signal(normal_test_filtered, check=1)  # Use check=1 for [-1, 1] normalization
pvc_train_filtered_nor = Normalize_Signal(pvc_train_filtered, check=1)  # Use check=1 for [-1, 1] normalization
pvc_test_filtered_nor = Normalize_Signal(pvc_test_filtered, check=1)  # Use check=1 for [-1, 1] normalization


# %%
plt.plot(normal_train_filtered_nor[0])

# %%
plt.plot(normal_test_filtered_nor[0])

# %%
plt.plot(pvc_train_filtered_nor[0])

# %%
plt.plot(pvc_test_filtered_nor[0])

# %%
def wavelet_features(signal):
    coeffs = pywt.wavedec(signal, wavelet='db4', level=4)
    wavelets = []
    for coeff in coeffs:
        wavelets.append(np.mean(coeff))
        wavelets.append(np.std(coeff))
        wavelets.append(np.sum(np.square(coeff)))
        wavelets.append(np.max(coeff))
        wavelets.append(np.min(coeff))
        wavelets.append(np.max(coeff) - np.min(coeff))
    return np.array(wavelets)

# %%
def features_data(data):
    features = []
    for signal in data:
        features.append(wavelet_features(signal))
    return np.array(features)

# %%
train_data = np.concatenate((normal_train_filtered_nor, pvc_train_filtered_nor))
train_labels = np.array([0] * len(normal_train_filtered_nor) + [1] * len(pvc_train_filtered_nor))

# %%
test_data = np.concatenate((normal_test_filtered_nor, pvc_test_filtered_nor))
test_labels = np.array([0] * len(normal_test_filtered_nor) + [1] * len(pvc_test_filtered_nor))

# %%
train_data, train_labels = shuffle(train_data, train_labels, random_state=46)
test_data, test_labels = shuffle(test_data, test_labels, random_state=46)

# %%
features_train = features_data(train_data)
features_test = features_data(test_data)

# %%
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# %%
k_range = range(1, 31)

cv_scores = []

for k in k_range:
    knn_test = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_test, features_train, train_labels, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(features_train, train_labels)

# %%
prediction = knn.predict(features_test)
print(accuracy_score(test_labels, prediction))

# %%
knn_scores = cross_val_score(knn, features_train, train_labels, cv=10)
print("Cross-validation scores:\n", knn_scores)
print("Average Cross-Validation score: {:.3f}".format(knn_scores.mean()))

# %%
confusion_mat= confusion_matrix(test_labels,prediction)
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Purples")
plt.xlabel("Predicted label")
plt.ylabel("Real label")
plt.title("Confusion Matrix")
plt.show()
#**************************************************************************************************************************************
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
