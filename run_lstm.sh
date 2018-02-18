#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -l gpu=1 
#$ -o results_lstm.txt
#$ -e err.lstm

python speech_lstm.py
