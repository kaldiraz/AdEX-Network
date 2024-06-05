"""
Contains the functions for simulating an AdeX neurons and couple it into a network of predetermined architecture

Author: Devanarayanan P
mail: prasanthdevanarayanan@gmail.com

"""
# ========== Importing packages ==========
import numpy as np
import pickle
import networkx as nx
import math
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# ========= Adaptation parameters ========
C = 200.0  # Membrane capacitance (pF)
g_l = 12.0  # Leak conductance (nS)
E_l = -70.0  # Leak reversal potential (mV)
V_t = -50.0  # Threshold potential (mV)
V_peak = 20.0  # Peak potential (mV)
delta_t = 2  # slope factor (mV)
tau_w = 300.0  # Adaptation time constant (ms)
a = 2.0  # Sub-threshold adaptation coupling (nS)
b = 50  # Spike-triggered adaptation increment (pA)
V_reset = E_l  # Reset current
current = 300  # pA
t_ref = 2  # Refractory time
V_rev = 0.0  # Reversal potential
gs = 3

# Network parameters
num_neurons = 50
degree = 20
prob = degree/(num_neurons - 1)

# simulation parameters
t_sim = 6000
dt = 0.05
t = np.arange(0, t_sim, dt)
iterations = int(t_sim/dt)

# =========== Loading network ============
def loading_network(file_path: str, user_choice: str, fraction_of_inhibitory_neurons: int):
    num_inhibitory_neurons = int(fraction_of_inhibitory_neurons * num_neurons)
    modular_network = nx.read_adjlist(file_path)
    module_size = [25, 25]

    choice_of_neurons = []
    if user_choice == "uniform":
        choice_of_neurons = np.random.choice(range(1, num_neurons + 1), num_inhibitory_neurons, replace=False)
    if user_choice == "from_module1":
        choice_of_neurons = np.random.choice(range(1, module_size[0] + 1), num_inhibitory_neurons, replace=False)
    if user_choice == "from_module2":
        choice_of_neurons = np.random.choice(range(module_size[0] + 1, num_neurons + 1), num_inhibitory_neurons, replace=False)

    for neuron in choice_of_neurons:
        for neighbor in modular_network.neighbors(str(neuron)):
            modular_network[str(neuron)][neighbor]["weight"] = -1
            modular_network[neighbor][str(neuron)]["weight"] = -1
    
    adj_matrix = nx.to_numpy_array(modular_network)
    return adj_matrix
    
# =========== Differntial equation =======
def adex_function(state, adj_matrix, neuron_index, iteration, spike, gs):

    v = state[neuron_index, 0]
    w = state[neuron_index, 1]

    dv_dt = (-g_l * (v - E_l) + g_l * delta_t * np.exp((v - V_t)/delta_t) - w + current)/C
    dw_dt = (a * (v - E_l) - w)/tau_w

    summation_term = 0
    for j in range(num_neurons):
        if len(spike[j]) != 0:
            if spike[j][-1] == t[iteration - 1]:
                summation_term += adj_matrix[neuron_index, j]
        else:
            continue

    coupling_term = gs * summation_term

    adex_output = np.array((dv_dt + coupling_term, dw_dt))

    return adex_output

# ============ euler ==================
def euler(initial_state, adj_matrix, gs):
    spike = [[] for _ in range(num_neurons)]

    state_final = np.zeros((iterations, num_neurons, 2))
    state_final[0, :, :] = initial_state

    for i in tqdm(range(1, iterations)):
        for j in range(num_neurons):
            adex_output = adex_function(state_final[i - 1, :, :], adj_matrix, j, i, spike, gs)

            state_final[i, j, :] = state_final[i - 1, j, :] + dt * adex_output

            if state_final[i, j, 0] > 0:
                state_final[i, j, 0] = V_reset
                state_final[i, j, 1] += b
                spike[j].append(t[i])

    return spike, state_final

# =========== initial conditions ============
state_ini_network = np.zeros((num_neurons, 2))

for i in range(len(state_ini_network)):
    state_ini_network[i] = np.array((np.random.uniform(-80, -70), np.random.uniform(0.1, 0.2)))

# ============ CV calculation ==============
def cv_network(spike):
    spike_interval = [[] for _ in range(num_neurons)]
    mean_isi = np.zeros(num_neurons)
    std_isi = np.zeros(num_neurons)
    coff_variation = np.zeros(num_neurons)

    for i in range(num_neurons):
        for j in range(1, len(spike[i])):
            spike_interval[i].append(spike[i][j] - spike[i][j - 1])

    for i in range(num_neurons):
        mean_isi[i] = np.mean(spike_interval[i])
        std_isi[i] = np.std(spike_interval[i])
        if math.isnan(mean_isi[i]):
            coff_variation[i] = 0
        else:
            coff_variation[i] = std_isi[i] / mean_isi[i]

    return coff_variation

# Calculating the adaptive index to show whether the system is behaving adaptively or not
def adaptation_index(spike):
    k = 4
    length_isi = 150

    spike_interval = [[] for _ in range(num_neurons)]
    a_value = [[] for _ in range(num_neurons)]

    for i in range(len(spike)):
        for j in range(1, len(spike[i])):
            spike_interval[i].append(spike[i][j] - spike[i][j - 1])

    spike_interval = [row[:length_isi] for row in spike_interval]

    for i in range(len(spike_interval)):
        summation = np.sum(
            [((spike_interval[i][j] - spike_interval[i][j - 1]) / (spike_interval[i][j] + spike_interval[i][j - 1])) for
             j in range(len(spike_interval[i][k:]))])
        a_value[i] = summation / (length_isi - k - 1)

    return a_value


# Calculating the order parameter of the network
def calculating_order_parameter(spike):
    # Recoring neuronal phase
    psi_j = [[] for _ in range(num_neurons)]

    # Calculating the phase
    for i in range(len(spike)):
        for m in range(0, len(spike[i]) - 1):
            t = spike[i][m]
            while t < spike[i][m + 1]:
                psi_j[i].append(2 * np.pi * m + 2 * np.pi * (t - spike[i][m]) / (spike[i][m + 1] - spike[i][m]))
                t += dt

    # Trimming the spikes so that the length of the arrays are the same
    minimum_length = min([len(psi_j[i]) for i in range(len(psi_j))])
    psi_j = [psi[:minimum_length] for psi in psi_j]

    exponential_phase = [[np.exp(1j * element) for element in psi] for psi in psi_j]

    # Calculating average phase and storing the phase values
    psi_data = pd.DataFrame(exponential_phase)
    psi_data = psi_data.T
    psi_data["average"] = 1 / num_neurons * abs(psi_data.sum(axis=1))
    # psi_data.to_csv("phase_of_neurons.csv")

    # Calculating the time average
    order_parameter = psi_data["average"].sum() / minimum_length

    return order_parameter


# Calculating the firing rate
def calculating_firing_rate(spike):
    firing_rate = np.zeros(len(spike))
    bin_size = 100
    time_bins = np.arange(200, t_sim + bin_size, bin_size)
    for i in range(len(spike)):
        fir, _ = np.histogram(spike[i], bins=time_bins)
        fir = fir / bin_size
        firing_rate[i] = np.mean(fir)
    return firing_rate
