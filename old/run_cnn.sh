#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -o results_cnn.txt
#$ -e err.cnn

python speech_cnn.py
