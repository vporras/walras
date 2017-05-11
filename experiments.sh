#!/bin/bash

# constraints, mean reversion
echo cons_fixed05_mean
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion mean -c fixed -f 0.05 -e cons_fixed05_mean
echo cons_fixed10_mean
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion mean -c fixed -f 0.1 -e cons_fixed10_mean
echo cons_fixed20_mean
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion mean -c fixed -f 0.2 -e cons_fixed20_mean
echo cons_mean_mean
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion mean -c mean -e cons_mean_mean
echo cons_last_mean
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion mean -c last -e cons_last_mean

# constraints, total reversion
echo cons_fixed05_total
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion total -c fixed -f 0.05 -e cons_fixed05_total
echo cons_fixed10_total
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion total -c fixed -f 0.1 -e cons_fixed10_total
echo cons_fixed20_total
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion total -c fixed -f 0.2 -e cons_fixed20_total
echo cons_mean_total
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion total -c mean -e cons_mean_total
echo cons_last_total
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion total -c last -e cons_last_total

# constraints, random reversion
echo cons_fixed05_random
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion random -c fixed -f 0.05 -e cons_fixed05_random
echo cons_fixed10_random
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion random -c fixed -f 0.1 -e cons_fixed10_random
echo cons_fixed20_random
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion random -c fixed -f 0.2 -e cons_fixed20_random
echo cons_mean_random
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion random -c mean -e cons_mean_random
echo cons_last_random
python3 walras.py -r 500 -n 100 -m 0.0001 -d -s 0 -v 1 -p 0 -t 100 --reversion random -c last -e cons_last_random
