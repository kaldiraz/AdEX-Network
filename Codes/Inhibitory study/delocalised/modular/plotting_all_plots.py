""" This code plots CV, OP, fraction of bursting neurons vs g_s for different networks and percentage inhibition"""

# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

plt.rc("font", size=23)

# Pickle directories
pickle_directory = "pickle_jar"

# Percentage inhibition
percentage_inhibition = float(sys.argv[1])

# Parameters
num_neurons = 50
time_array = np.arange(0, 6000, 0.05)
gs_array = np.arange(10, 40)

# Plotting parameters
color_list = ["black", "red", "blue"]
spread_color_list = ["black", "red", "blue"]
marker_color_list = ["black", "red", "blue"]
marker_shape = ["o", "^", "s"]
labels_list = []

# Plotting CV for gs:
def plotting_cv(percentage_inhibition):
    cv_for_all_networks = []
    for file in os.listdir(pickle_directory):
        if file.startswith("mega_cv") and f"percinh_{percentage_inhibition}" in file:
            print(file[22:-16])
            labels_list.append(f"r = {file[22:-16]}")
            with open(os.path.join(pickle_directory, file), "rb") as f:
                cv_array = pickle.load(f)
                cv_for_all_networks.append(cv_array)

    mean_cv_for_all_networks = np.mean(cv_for_all_networks, axis=1)

    plt.figure(figsize=(10, 8))
    for index, network in enumerate(range(3)):
        plt.plot(gs_array, np.mean(mean_cv_for_all_networks[network], axis=1), marker=marker_shape[index], color=color_list[index], mec="black", mfc=marker_color_list[index], label=labels_list[index])
        plt.fill_between(gs_array, np.mean(mean_cv_for_all_networks[network], axis=1) - np.std(mean_cv_for_all_networks[network], axis=1), np.mean(mean_cv_for_all_networks[network], axis=1) + np.std(mean_cv_for_all_networks[network], axis=1), color=spread_color_list[index], alpha=0.1)
#    plt.axvline(x=15, color=color_list[2], ls="-.")
#    plt.axvline(x=15.5, color=color_list[1], ls="-.")
#    plt.axvline(x=17, color=color_list[0], ls="-.")
    plt.axhline(y=0.5, ls=":")
    plt.xlabel("$g_s$")
    plt.ylabel("CV")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"img/cv_percinh_{percentage_inhibition}.png")
    print("CV plot saved ...")
    return mean_cv_for_all_networks

def plotting_number_bursting(percentage_inhibition, mean_cv_for_all_networks):

    with open(f"cv_crossing_data.dat", "w") as f:
        f.write("# 0 --> r = 0.01\n")
        f.write("# 1 --> r = 1\n")
        f.write("# 2 --> r = 100\n")
        f.write("# network\tgs\tneuron\tcv\n")
        for network in range(3):
            for gs_index, gs in enumerate(np.arange(12, 25, 0.5)):
                for neuron in range(50):
                    if (mean_cv_for_all_networks[network][gs_index][neuron]) > 0.5:
                        f.write(f"{network}\t{gs}\t{neuron}\t{mean_cv_for_all_networks[network][gs_index][neuron]}\n")

    bursting_list = [[[] for _ in range(len(gs_array))] for _ in range(3)]
    for network in range(3):
        bursting_neurons = []
        for gs_index, gs in enumerate(gs_array):
            for neuron in range(50):
                if (mean_cv_for_all_networks[network][gs_index][neuron]) > 0.5:
                    bursting_neurons.append(neuron)

            bursting_list[network][gs_index] = len(set(bursting_neurons))/num_neurons

    plt.figure(figsize=(10, 8))
    for network in range(3):
        plt.plot(gs_array, bursting_list[network], marker=marker_shape[network], color=color_list[network], label=labels_list[network], mec="black", mfc=marker_color_list[network])
    plt.xlabel("$g_s$")
    plt.ylabel("Fraction of bursting neurons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/number_bursting_percinh_{percentage_inhibition}.png")
    print("number of bursting saved ...")

def plotting_op(percentage_inhibition):
    op_for_all_networks = []
    for file in os.listdir(pickle_directory):
        if file.startswith("mega_op") and f"percinh_{percentage_inhibition}" in file:
            with open(os.path.join(pickle_directory, file), "rb") as f:
                op_array = pickle.load(f)
                op_for_all_networks.append(op_array)
               

    mean_op_for_all_network = np.mean(op_for_all_networks, axis=1)

    plt.figure(figsize=(10, 8))
    for index, network in enumerate(range(3)):
        plt.plot(gs_array, mean_op_for_all_network[network], marker=marker_shape[index], color=color_list[index], mec="black", mfc=marker_color_list[index], label=labels_list[index])
    plt.xlabel("$g_s$")
    plt.ylabel("R")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/op_percinh_{percentage_inhibition}.png")
    print("plotting op saved ...")

def firing_rate(percentage_inhibition):
    fir_for_all_networks = []
    for file in os.listdir(pickle_directory):
        if file.startswith("mega_fir") and f"percinh_{percentage_inhibition}" in file:
            with open(os.path.join(pickle_directory, file), "rb") as f:
                fir_array = pickle.load(f)
                fir_for_all_networks.append(fir_array)

    mean_fir_for_all_network = np.mean(fir_for_all_networks, axis=1)

    np.shape(mean_fir_for_all_network)

    plt.figure(figsize=(10, 8))
    for index, network in enumerate(range(3)):
        plt.plot(gs_array, np.mean(mean_fir_for_all_network[network], axis=1), marker=marker_shape[index], mec="black", mfc=marker_color_list[index], color=color_list[index], label=labels_list[index])
    plt.xlabel("$g_s$")
    plt.ylabel("Firing rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/fir_percinh_{percentage_inhibition}.png")
    print("plotting fir saved ...")


if __name__ == "__main__":
    mean_cv_for_all_networks = plotting_cv(percentage_inhibition)

    plotting_op(percentage_inhibition)
    plotting_number_bursting(percentage_inhibition, mean_cv_for_all_networks)
    firing_rate(percentage_inhibition)
