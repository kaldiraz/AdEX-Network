import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import time

from adex_functions import euler, state_ini_network, loading_network, cv_network, t
from adex_functions import calculating_order_parameter, calculating_firing_rate, adaptation_index

num_neurons = 50
degree = 20
user_choice = "from_module1"
gs_range = np.arange(12, 25, 0.5)

percent_inhibitory_neurons = float(sys.argv[1])
print(f"percent_inhibitory_neurons: {percent_inhibitory_neurons}")
adj_matrix_dir = "../adj/"

plt.rc("font", size=18)
plt.figure(figsize=(10, 8))

def removing_transient(spike, transient_time):
    new_spike = [[] for _ in range(len(spike))]
    for i in range(len(spike)):
        for j in spike[i]:
            if j > transient_time:
                new_spike[i].append(j)
    return new_spike

if __name__=='__main__':
    start = time.time()
    for file in os.listdir(adj_matrix_dir):
        if file == "r_1.dat":
            file_path = os.path.join(adj_matrix_dir, file)
            print(file.center(80, "="))
            adj_matrix = loading_network(file_path, user_choice, percent_inhibitory_neurons)

            mega_cv_array = []
            mega_ai_array = []
            mega_op_array = []
            mega_fir_array = []

            for i in range(5):
                print(f"realisation: {i}".center(80))
                cv_array = []
                ai_array = []
                op_array = []
                fir_array = []

                for gs in gs_range:
                    print(f"gs: {gs}")
                    spike,state_final = euler(state_ini_network, adj_matrix, gs)

                    with open(f"pickle_jar/spike_gs_{gs}_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
                        pickle.dump(spike, f)

                    with open(f"pickle_jar/state_gs_{gs}_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
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

                mega_cv_array.append(cv_array)
                mega_ai_array.append(ai_array)
                mega_op_array.append(op_array)
                mega_fir_array.append(fir_array)

                with open(f"pickle_jar/mega_cv_modular_mixed_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
                    pickle.dump(mega_cv_array, f)

                with open(f"pickle_jar/mega_ai_modular_mixed_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
                    pickle.dump(mega_ai_array, f)

                with open(f"pickle_jar/mega_op_modular_mixed_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
                    pickle.dump(mega_op_array, f)

                with open(f"pickle_jar/mega_fir_modular_mixed_{file[2:-4]}_percinh_{percent_inhibitory_neurons}.pkl", "wb") as f:
                    pickle.dump(mega_fir_array, f)
                
                end = time.time()
                print(f"Total elapsed time: {end - start}")
