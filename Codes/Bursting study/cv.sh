#! /bin/bash

for bursting in 0.2 0.5 0.8;
do
    python3 calculating_cv.py $bursting
done

echo "All done"
