#!/bin/bash


python3.11 fit_power_law.py --name Czechia --N_run 14 --k 3 --size 6500
python3.11 fit_power_law.py --name Slovakia --N_run 20 --k 3 --size 2700
python3.11 fit_power_law.py --name Slovenia --N_run 10 --k 3 --size 2900

