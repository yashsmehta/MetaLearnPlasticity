# !/bin/bash

for input_dim in 500
do
    for output_dim in 50
    do
        for upto_ith_order in 2 3 4 5 6
        do
            for noise_scale in 0
            do
                for sparsity in 1 0.05
                do
                    for l1_lmbda in 1e-5 1e-6 1e-7 1e-8 0
                    do
                        for jobid in 0 1 2 3 4
                        do
                            bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "ith" -q gpu_rtx "python exps/scalability/main.py \
                            -input_dim $input_dim -output_dim $output_dim -upto_ith_order $upto_ith_order -sparsity $sparsity\
                            -jobid $jobid -noise_scale $noise_scale -l1_lmbda $l1_lmbda -len_trajec 5 -log_expdata True -num_trajec 100 -meta_epochs 20 -output_file upto_ith_exps"
                        done
                    done
                done
            done
        done
    done
done
