#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -l gpu=1 

python speech_svm.py
