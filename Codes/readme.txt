How to Run the Codes

1. adj folder contains all the adjacency matrix under consideration
2. modular folders contain the codes to run which are distributed into Bursting study and Inhibitory study; and in inhibitory study there is a localised and delocalised case
3. Bursting study is regarding giving some neurons bursting parameters while others being given spiking parameters
4. Inhibitory study is regarding making some percentage of population inhibitory while others being excitatory
5. in each folder, adex_functions.py contains all the functions that are used in this study
6. calculating_cv_$network_name.py contains the code to calculate CV values for different networks, the percentage of bursting/inhibition is given as a terminal argument
7. the data is stored as pickle files in pickle_jar folder. The jupyter notebook files in each folder plots these pickle jar files and store the result in the img folder
8. Single neuron folder contains the code to generate plots in the case of single neurons.

For running the codes

1. navigate to the modular network for the case you are interested in and run the calculating_cv.....py file with a particular percentage of bursting/inhibition
2. After storing the data, the jupyter notebook files can be run in order to plot the data