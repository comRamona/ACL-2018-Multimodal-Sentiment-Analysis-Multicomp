#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -l gpu=3 
#$ -o results_blstm.txt
#$ -e err.blstm

python speech_blstm.py
