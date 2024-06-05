import numpy as np
import pickle
import sys
from adex_functions import num_neurons, state_ini_network, t
from adex_functions import loading_network, euler, cv_network, adaptation_index, calculating_order_parameter, calculating_firing_rate

gs_range = np.arange(10, 40)
r_value = 0.01
bursting_number = float(sys.argv[1])
def removing_transient(spike, transient_time):
    new_spike = [[] for _ in range(len(spike))]
    for i in range(len(spike)):
        for j in spike[i]:
            if j > transient_time:
                new_spike[i].append(j)
    return new_spike

print(f"For r value: {r_value}")
adj_matrix_file = f"../adj/r_{r_value}.dat"

cv_array = []
ai_array = []
op_array = []
fir_array = []

adj_matrix = loading_network(adj_matrix_file)
bursting_choice = np.random.choice(range(num_neurons), int(bursting_number * num_neurons), replace=False)

print(bursting_choice)
for gs in gs_range:
    print(f"gs: {gs}")
    spike, state_final = euler(state_ini_network, bursting_choice, adj_matrix, gs)
    
    with open(f"pickle_jar/spike_gs_{gs}_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(spike, f)
    
    with open(f"pickle_jar/state_gs_{gs}_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(state_final, f)

    spike = removing_transient(spike, 2000)
    
    cv = cv_network(spike)
    cv_array.append(cv)
    print(cv)

    ai = adaptation_index(spike)
    ai_array.append(ai)
    print(ai)

    op = calculating_order_parameter(spike)
    op_array.append(op)
    print(op)

    fir = calculating_firing_rate(spike)
    fir_array.append(fir)
    print(fir)

    with open(f"pickle_jar/cv_mod_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(cv_array, f)

    with open(f"pickle_jar/ai_mod_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(ai_array, f)

    with open(f"pickle_jar/op_mod_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(op_array, f)

    with open(f"pickle_jar/fir_mod_{r_value}_burst_{bursting_number}.pkl", "wb") as f:
        pickle.dump(fir_array, f)
