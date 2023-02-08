# !/bin/bash

for input_dim in 50 500
do
    for output_dim in 1 5 50
    do
        for upto_ith_order in 3 5
        do
            for len_trajec in 2 10 25 
            do
                for sparsity in 1
                do
                    for l1_lmbda in 1e-8 1e-9 1e-10 0
                    do
                        for jobid in 0 1 2
                        do
                            bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "xw diff" -q gpu_rtx "python exps/scalability/main.py \
                            -input_dim $input_dim -output_dim $output_dim -upto_ith_order $upto_ith_order -sparsity $sparsity\
                            -jobid $jobid -noise_scale 0 -l1_lmbda $l1_lmbda -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file diff_init_diff_x"
                        done
                    done
                done
            done
        done
    done
done