# AdEX-Network
 Simulating adaptive exponential integrate and Fire neurons

 This repository contains the code that I have made for simulating a network of adaptive exponential integrate and fire neurons for my MS Thesis. Different network are considered in this study, namely, Modular, Antimodular and Random networks.

## How to run the codes

1. `Bursting Study` contains the code to simulate a network of adaptive exponential integrate and fire neurons having both spiking and bursting parameters.

    - `adex_functions.py`: Contains all the functions used in the code for simulation.

    - `calculating_cv_$network_name.py`: Contains the code to calculate CV (Coefficient of variation) values for different networks, the percentage of bursting/inhibition is given as a terminal argument.

    - `cv.sh`: Script to run the all `calculating_cv_$network_name.py` files together.

    - `Plotting.py`: Contains the code to plot the data.

2. `Inhibitory Study` contains the code to simulate a network of adaptive exponential integrate and fire neurons having both inhibitory and excitatory parameters.

    - `delocalised` has the inhibitory nodes spread through out the network and `localised` has the inhibitory nodes localised in the modules of the networs.

        - `adj` folder contains all the adjacency matrix under consideration.

        - `modular` folders contain the codes to run.

3. `Single Neuron Study` contains the code to simulate a single adaptive exponential integrate and fire neuron and plotting its timeseries. 
