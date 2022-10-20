# !/bin/bash

for input_dim in 100 500
do
    for output_dim in 10 50 
    do
        for sparsity in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.01
        do
            for num_meta_params in 10
            do
                for len_trajec in 50
                do
                    bsub -n 12 -o cluster_logs.out -gpu "num=1" -J "sparsity" -q gpu_rtx "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -num_meta_params $num_meta_params -sparsity $sparsity\
                    -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file sparsity_exps"
                done
            done
        done
    done
done
