log_path=savedir/scan_params/scan_hyperparams.txt
echo 'scan model hyperparams:' > $log_path
set -e

dpout_fc_vec=(0.0 0.2 0.5)
enc_lstm_dim_vec=(256 512 1024)
fc_dim_vec=(128 256 512)
gpu=0


for dpout_fc in ${dpout_fc_vec[@]}
do
    for enc_lstm_dim in ${enc_lstm_dim_vec[@]}
    do
        for fc_dim in ${fc_dim_vec[@]}
        do
            echo "dpout_fc: " $dpout_fc ", enc_lstm_dim: " $enc_lstm_dim ", fc_dim: " $fc_dim >> $log_path
            save_path="savedir/scan_params/BLSTM_dpc${dpout_fc}_ld${enc_lstm_dim}_fd${fc_dim}.pkl"
            echo $save_path
            python train_mojing.py --dpout_fc $dpout_fc --enc_lstm_dim $enc_lstm_dim \
              --fc_dim $fc_dim --save_path $save_path --gpu $gpu --batch_size 128 --nonlinear_fc 1
            python evaluate.py --modelpath $save_path --gpu $gpu >> $log_path
        done
    done
done
