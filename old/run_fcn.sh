#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -o results_fcn.txt
#$ -e err.fcn

python speech_fcn.py
