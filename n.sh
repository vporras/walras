#!/bin/bash

echo n_bt_10
python3 walras.py -r 500 -n 10   -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_10
echo n_bt_20
python3 walras.py -r 500 -n 20   -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_20
echo n_bt_40
python3 walras.py -r 500 -n 40   -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_40
echo n_bt_80
python3 walras.py -r 500 -n 80   -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_80
echo n_bt_160
python3 walras.py -r 500 -n 160  -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_160
echo n_bt_320
python3 walras.py -r 500 -n 320  -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_320
echo n_bt_640
python3 walras.py -r 500 -n 640  -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -e n_bt_640
