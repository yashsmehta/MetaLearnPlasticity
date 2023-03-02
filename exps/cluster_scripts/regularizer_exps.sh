# !/bin/bash

for input_dim in 50 500
do
    for output_dim in 5 50
    do
        for len_trajec in 2 5 10
        do
            for l1_lmbda in 0 1e-7 1e-8 1e-9 5e-10 1e-10 5e-11 1e-11 5e-12 1e-12
            do
                for jobid in 0 1 2 3 4
                do
                    bsub -n 4 -o cluster_logs.out -J "regexps" -q local "python exps/scalability/main.py \
                    -input_dim $input_dim -output_dim $output_dim -upto_ith_order 3 -sparsity 1 -l1_lmbda $l1_lmbda \
                    -jobid $jobid -noise_scale 0 -len_trajec $len_trajec -log_expdata True -num_trajec 200 -meta_epochs 100 -output_file regularizer_exps"
                done
            done
        done
    done
done