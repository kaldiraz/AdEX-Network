"""
Contains the functions for simulating an AdeX neurons and couple it into a network of predetermined architecture

Author: Devanarayanan P
mail: prasanthdevanarayanan@gmail.com

"""

#============ necessary packages =================
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import math
from tqdm import tqdm
import networkx as nx

#=============== Parameter dictionary ============

adapting = {
        "C": 200.0,
        "g_l": 12.0,
        "E_l": -70.0,
        "V_t": -50.0,
        "delta_t": 2,
        "V_peak": 20.0,
        "tau_w": 300.0,
        "a": 2.0,
        "b": 50,
        "V_reset": -70.0,
        "current": 300,
        "t_ref": 2,
        "V_rev": 0.0,
        "gs": 3
        }

bursting = {
        "C": 200.0,
        "g_l": 10.0,
        "E_l": -70.0,
        "V_t": -50.0,
        "delta_t": 2,
        "V_peak": 20.0,
        "tau_w": 300.0,
        "a": 2.0,
        "b": 100,
        "V_reset": -46,
        "current": 400,
        "t_ref": 2,
        "V_rev": 0.0,
        "gs": 3
        }


# Network parameters
num_neurons = 50
degree = 25
prob = degree/(num_neurons - 1)

# simulation parameters
t_sim = 6000
dt = 0.05
t = np.arange(0, t_sim, dt)
iterations = int(t_sim/dt)

# =========== Loading network ============
def loading_network(file_path):
    modular_network = nx.read_adjlist(file_path)
    adj_matrix = nx.to_numpy_array(modular_network)

    return adj_matrix
    
# =========== Differntial equation =======
def adex_network(state, params, burst_choice, adj_matrix, neuron_index, iteration, spike, gs):

    V = state[neuron_index, 0]
    w = state[neuron_index, 1]
    adex_output = np.zeros_like(state[0])

    dV_dt = (params["g_l"] * (params["E_l"] - V) + params["g_l"] * params["delta_t"] * np.exp((V - params["V_t"]) / params["delta_t"]) - w + params["current"]) / params["C"]
    dw_dt = (params["a"] * (V - params["E_l"]) - w) / params["tau_w"]

    summation_term = 0
    for j in range(num_neurons):
        if len(spike[j]) != 0:
            if spike[j][-1] == t[iteration - 1]:
                summation_term += adj_matrix[neuron_index, j]
        else:
            continue
                
    coupling_term = gs * summation_term

    adex_output[0] = dV_dt + coupling_term
    adex_output[1] = dw_dt

    return adex_output

# ============ euler ==================
def euler(initial_state, bursting_choice, adj_matrix, gs):
    spike = [[] for _ in range(num_neurons)]
    state_final = np.zeros((iterations, num_neurons, 2))
    state_final[0] = initial_state

    for i in tqdm(range(1, iterations)):
        for j in range(num_neurons):
            
            if j in bursting_choice:
                params = bursting
            else:
                params = adapting

            adex_output = adex_network(state_final[i - 1, :, :], params, bursting_choice, adj_matrix, j, i, spike, gs)

            state_final[i, j, :] = state_final[i - 1, j, :] + dt * adex_output
            
            if state_final[i, j, 0] > 0:
                # state_final[i - 1, j, 0] = params["V_peak"]
                state_final[i, j, 0] = params["V_reset"]
                state_final[i, j, 1] += params["b"]
                
                if t[i] > 0:
                    spike[j].append(t[i])
    
    return spike, state_final

# =========== initial conditions ============
state_ini_network = np.zeros((num_neurons, 2))

for i in range(len(state_ini_network)):
    # state_ini_network[i] = np.array((np.random.uniform(-80, -70), np.random.uniform(0.1, 0.2)))
    state_ini_network[i] = np.array((np.random.uniform(-80, -70), 0)) 

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
