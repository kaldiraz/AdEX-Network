"""
Code to simulate a single AdEx neuron

Author: Devanarayanan P
mail: prasanthdevanarayanan@gmail.com

"""

# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Adex Parameters

irregular_spiking = False
transient_spiking = False
delayed_regular_spiking = False
delayed_accelerating = False
regular_bursting = False
initial_bursting = False
adaptation = True 
tonic_spiking = False

if tonic_spiking:
    C = 200.0  # Membrane capacitance (pF)
    g_l = 10.0  # Leak conductance (nS)
    E_l = -70.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 30.0  # Adaptation time constant (ms)
    a = 2.0  # Subthreshold adaptation coupling (nS)
    b = 0.0  # Spike-triggered adaptation increment (pA)
    V_reset = -58  # Reset current
    current = 500  # pA

if adaptation:
    C = 200.0  # Membrane capacitance (pF)
    g_l = 12.0  # Leak conductance (nS)
    E_l = -70.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 300.0  # Adaptation time constant (ms)
    a = 2.0  # Subthreshold adaptation coupling (nS)
    b = 60  # Spike-triggered adaptation increment (pA)
    V_reset = -58.0  # Reset current
    current = 500  # pA

if initial_bursting:
    C = 130.0  # Membrane capacitance (pF)
    g_l = 18.0  # Leak conductance (nS)
    E_l = -58.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 150.0  # Adaptation time constant (ms)
    a = 4.0  # Subthreshold adaptation coupling (nS)
    b = 120.0  # Spike-triggered adaptation increment (pA)
    V_reset = -50  # Reset current
    current = 400  # pA

if regular_bursting:
    C = 200.0  # Membrane capacitance (pF)
    g_l = 10.0  # Leak conductance (nS)
    E_l = -58.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 120.0  # Adaptation time constant (ms)
    a = 2.0  # Subthreshold adaptation coupling (nS)
    b = 100  # Spike-triggered adaptation increment (pA)
    V_reset = -46  # Reset current
    current = 210  # pA

if delayed_accelerating:
    C = 200.0  # Membrane capacitance (pF)
    g_l = 12.0  # Leak conductance (nS)
    E_l = -70.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 300.0  # Adaptation time constant (ms)
    a = -10.0  # Subthreshold adaptation coupling (nS)
    b = 0.0  # Spike-triggered adaptation increment (pA)
    V_reset = -58  # Reset current
    current = 300  # pA

if delayed_regular_spiking:
    C = 100.0  # Membrane capacitance (pF)
    g_l = 10.0  # Leak conductance (nS)
    E_l = -65.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 90.0  # Adaptation time constant (ms)
    a = -10.0  # Subthreshold adaptation coupling (nS)
    b = 30.0  # Spike-triggered adaptation increment (pA)
    V_reset = -47  # Reset current
    current = 110  # pA

if transient_spiking:
    C = 100.0  # Membrane capacitance (pF)
    g_l = 10.0  # Leak conductance (nS)
    E_l = -65.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 90.0  # Adaptation time constant (ms)
    a = 10.0  # Subthreshold adaptation coupling (nS)
    b = 100  # Spike-triggered adaptation increment (pA)
    V_reset = -47  # Reset current
    current = 180  # pA

if irregular_spiking:
    C = 100.0  # Membrane capacitance (pF)
    g_l = 12.0  # Leak conductance (nS)
    E_l = -60.0  # Leak reversal potential (mV)
    V_t = -50.0  # Threshold potential (mV)
    V_peak = 20.0  # Peak potential (mV)
    delta_t = 2  # slope factor (mV)
    tau_w = 130.0  # Adaptation time constant (ms)
    a = -11.0  # Subthreshold adaptation coupling (nS)
    b = 30  # Spike-triggered adaptation increment (pA)
    V_reset = -48  # Reset current
    current = 160  # pA

def adex_function(state, I):
    V, w = state
    adex_output = np.zeros_like(state)
    
    dV_dt = (-g_l * (V - E_l) + g_l * delta_t * np.exp((V - V_t)/delta_t) - w + current) / C
    dw_dt = (a * (V - E_l) - w)/tau_w

    adex_output[0] = dV_dt
    adex_output[1] = dw_dt

    return adex_output

t_sim = 6000
dt = 0.05
iterations = int(t_sim/dt)
t = np.arange(0, t_sim, dt)

def euler(initial_state, I):
    state_final = np.zeros((iterations, 2))

    state_final[0] = initial_state

    for i in tqdm(range(1, iterations)):
        adex_output = adex_function(state_final[i - 1], I[i])

        state_final[i] = state_final[i - 1] + dt * adex_output
        if state_final[i, 0] > 0:
            state_final[i - 1, 0] = V_peak
            state_final[i, 0] = V_reset
            state_final[i, 1] += b
    plt.rc("font", size=18)
    plt.figure(figsize=(10, 8))
    plt.plot(state_final[-11000:, 0], state_final[-11000:, 1], color="#988ed5")
    plt.xlabel("time (ms)")
    plt.ylabel("Adaptation Current (pA)")
    plt.savefig("adapting_phase_portrait.png")
    plt.tight_layout()
    plt.show()

input_current = np.full(iterations, current)
initial_state = [-70, 0]

euler(initial_state, input_current)
