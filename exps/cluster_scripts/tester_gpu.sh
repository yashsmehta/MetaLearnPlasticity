# !/bin/bash

for input_dim in 50 
do
    for output_dim in 5
    do
        for len_trajec in 5 10
        do
            for l1_lmbda in 0 1e-7
            do
                for jobid in 0
                do
                    bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "regexps" -q gpu_rtx "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -upto_ith_order 3 -sparsity 1 -l1_lmbda $l1_lmbda \
                    -jobid $jobid -noise_scale 0 -len_trajec $len_trajec -log_expdata True -num_trajec 20 -meta_epochs 2 -output_file regularizer_exps"
                done
            done
        done
    done
done