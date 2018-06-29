echo 'scan model hyperparams:' > scan_hyperparams.txt
set -e

dpout_model_vec=(0.0 0.1 0.2)
dpout_fc_vec=(0.0 0.2 0.5)
enc_lstm_dim_vec=(256 512 1024)
fc_dim_vec=(128 256 512)
gpu=0


for dpout_model in ${dpout_model_vec[@]}
do
    for dpout_fc in ${dpout_fc_vec[@]}
    do
        for enc_lstm_dim in ${enc_lstm_dim_vec[@]}
        do
            for fc_dim in ${fc_dim_vec[@]}
            do
                echo "dpout_model:" $dpout_model ", dpout_fc: " $dpout_fc ", enc_lstm_dim: " $enc_lstm_dim ", fc_dim: " $fc_dim >> scan_hyperparams.txt
                save_path="savedir/scan_params/BLSTM_dpm${dpout_model}_dpc${dpout_fc}_ld${enc_lstm_dim}_fd${fc_dim}.pkl"
                echo $save_path
                python train.py --dpout_model $dpout_model --dpout_fc $dpout_fc --enc_lstm_dim $enc_lstm_dim \
                 --fc_dim $fc_dim --save_path $save_path --gpu $gpu
                python evaluate.py --modelpath $save_path >> scan_hyperparams.txt --gpu $gpu
            done
        done
    done
done