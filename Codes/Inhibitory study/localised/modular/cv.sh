#! /bin/bash

for inh in 0 0.1 0.2 0.5 0.6;
do
	python3 calculating_cv.py $inh
done
wait
echo "All done"
