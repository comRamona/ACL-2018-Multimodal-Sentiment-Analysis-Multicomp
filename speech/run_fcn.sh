#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -l gpu=2
#$ -o results_fcn.txt
#$ -e err.fcn

python speech_fcn.py
