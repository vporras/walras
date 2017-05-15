#!/bin/bash

echo stable_conv_simple
python3 walras.py -r 500 -n 100 -m 0.0001 -s 0 -v 1 -p 0 -t 100 --trader-file traders_conv.txt -e stable_conv_simple
echo stable_div_simple
python3 walras.py -r 500 -n 100 -m 0.0001 -s 0 -v 1 -p 0 -t 100 --trader-file traders_div.txt -e stable_div_simple
echo stable_conv_backtrack
python3 walras.py -r 500 -n 100 -m 0.0001 -s 0 -v 1 -p 0 -t 100 --backtrack-prob 0.5 -b 5 25 100 -trader-file traders_conv.txt -e stable_conv_backtrack 
