# !/bin/bash

for input_dim in 100 500
do
    for output_dim in 5 
    do
        for l1_lmbda in 1e-4 5e-5 1e-5 5e-6 1e-6 5e-7 1e-7 0
        do
            for num_meta_params in 10 27
            do
                for len_trajec in 50
                do
                    bsub -n 12 -o cluster_logs.out -gpu "num=1" -J "exps" -q gpu_tesla "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -num_meta_params $num_meta_params -l1_lmbda $l1_lmbda\
                    -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 1000 -output_file regularizer_exps_V2"
                done
            done
        done
    done
done
