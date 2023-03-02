# !/bin/bash

for input_dim in 5 50 500
do
    for output_dim in 5 50 500
    do
        for len_trajec in 2 5 10
        do
            for sparsity in 1 0.9 0.8 0.7 0.6 0.5
            do
                for jobid in 0 1 2 3 4
                do
                    bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "exps" -q gpu_rtx "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -upto_ith_order 3 -sparsity $sparsity\
                    -jobid $jobid -noise_scale 0 -len_trajec $len_trajec -log_expdata True -num_trajec 200 -meta_epochs 50 -output_file sparsity_exps"
                done
            done
        done
    done
done