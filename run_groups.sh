python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 15 --dropout_rate 0.1 --n_layers $2 >> pca/lstm_$1_$2_1.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 20 --dropout_rate 0.1 --n_layers $2 >> pca/lstm_$1_$2_2.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 25 --dropout_rate 0.1 --n_layers $2 >> pca/lstm_$1_$2_3.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 30 --dropout_rate 0.1 --n_layers $2 >> pca/lstm_$1_$2_4.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 15 --dropout_rate 0.2 --n_layers $2 >> pca/lstm_$1_$2_5.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 20 --dropout_rate 0.2 --n_layers $2 >> pca/lstm_$1_$2_6.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 25 --dropout_rate 0.2 --n_layers $2 >> pca/lstm_$1_$2_7.txt 2>&1 &
python pca/early_fusion_lstm.py --experiment_prefix lstm_early_fusion --mode $1 --max_len 30 --dropout_rate 0.2 --n_layers $2 >> pca/lstm_$1_$2_8.txt 2>&1 &

#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 15 --dropout_rate 0.1 --n_layers $2 >> pca/blstm_$1_$2_1.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 20 --dropout_rate 0.1 --n_layers $2 >> pca/blstm_$1_$2_2.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 25 --dropout_rate 0.1 --n_layers $2 >> pca/blstm_$1_$2_3.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 30 --dropout_rate 0.1 --n_layers $2 >> pca/blstm_$1_$2_4.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 15 --dropout_rate 0.2 --n_layers $2 >> pca/blstm_$1_$2_5.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 20 --dropout_rate 0.2 --n_layers $2 >> pca/blstm_$1_$2_6.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 25 --dropout_rate 0.2 --n_layers $2 >> pca/blstm_$1_$2_7.txt 2>&1 &
#python pca/early_fusion_blstm.py --experiment_prefix blstm_early_fusion --mode $1 --max_len 30 --dropout_rate 0.2 --n_layers $2 >> pca/blstm_$1_$2_8.txt 2>&1 &

#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 15 --dropout_rate 0.1 --n_layers $2 >> pca/cnn_$1_$2_1.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 20 --dropout_rate 0.1 --n_layers $2 >> pca/cnn_$1_$2_2.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 25 --dropout_rate 0.1 --n_layers $2 >> pca/cnn_$1_$2_3.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 30 --dropout_rate 0.1 --n_layers $2 >> pca/cnn_$1_$2_4.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 15 --dropout_rate 0.2 --n_layers $2 >> pca/cnn_$1_$2_5.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 20 --dropout_rate 0.2 --n_layers $2 >> pca/cnn_$1_$2_6.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 25 --dropout_rate 0.2 --n_layers $2 >> pca/cnn_$1_$2_7.txt 2>&1 &
#python pca/early_fusion_cnn.py --experiment_prefix cnn_early_fusion --mode $1 --max_len 30 --dropout_rate 0.2 --n_layers $2 >> pca/cnn_$1_$2_8.txt 2>&1 &

