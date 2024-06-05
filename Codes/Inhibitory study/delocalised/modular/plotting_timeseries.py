""" Plotting time series from pickle files """

# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Parameters
num_neurons = 50
time_array = np.arange(0, 6000, 0.05)

# Directory
pickle_directory = "pickle_jar"

# Plotting
for file in os.listdir(pickle_directory):
    if file.startswith("state"):
        file_path = os.path.join(pickle_directory, file)
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        
        for neuron in range(num_neurons):
            plt.plot(time_array, state[:, neuron, 0])
            plt.xlabel("Time (ms)")
            plt.ylabel("V")
            plt.tight_layout()
            plt.savefig(f"img/{file[:-4]}_neuron_{neuron}.png")
            plt.clf()
        print(f"{file} done ...")


