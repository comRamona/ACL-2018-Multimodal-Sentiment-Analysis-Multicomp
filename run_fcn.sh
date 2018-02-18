#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -l gpu=1 
#$ -o results_fcn.txt
#$ -e err.fcn

python speech_fcn.py
