# !/bin/bash

for input_dim in 10 100 500
do
    for output_dim in 10 50 
    do
        for noise_scale in 0.1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5 5e-6
        do
            for num_meta_params in 2 5 10
            do
                for len_trajec in 50
                do
                    bsub -n 4 -o cluster_logs.out -gpu "num=1" -J "noisy" -q gpu_rtx "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -num_meta_params $num_meta_params -noise_scale $noise_scale\
                    -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file noisy_exps"
                done
            done
        done
    done
done