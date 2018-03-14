#!/bin/bash
#$ -N myjob
#$ -cwd
#$ -o cnn1.txt
#$ -e cnn1_err.txt


#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 15 --dropout_rate 0.1 --n_layers 1 --epochs 50
python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 20 --dropout_rate 0.1 --n_layers 1 --epochs 50
#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 25 --dropout_rate 0.1 --n_layers 1 --epochs 50


#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 15 --dropout_rate 0.2 --n_layers 1 --epochs 50
#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 20 --dropout_rate 0.2 --n_layers 1 --epochs 50
#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 25 --dropout_rate 0.2 --n_layers 1 --epochs 50


#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 15 --dropout_rate 0.35 --n_layers 1 --epochs 50
#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 20 --dropout_rate 0.35 --n_layers 1 --epochs 50
#python early_fusion_cnn.py --experiment_prefix cnn_early_fusion --max_len 25 --dropout_rate 0.35 --n_layers 1 --epochs 50




