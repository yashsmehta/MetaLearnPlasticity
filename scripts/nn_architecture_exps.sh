# !/bin/bash

for input_dim in 10 
do
    for output_dim in 1 2 5 10 50 5000
    do
        for upto_ith_order in 3
        do
            for noise_scale in 0
            do
                for sparsity in 1
                do
                    for jobid in 0 1 2 3 4
                    do
                        bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "arch" -q gpu_rtx "python exps/scalability/main.py \
                        -input_dim $input_dim -output_dim $output_dim -upto_ith_order $upto_ith_order -sparsity $sparsity\
                        -jobid $jobid -noise_scale $noise_scale -len_trajec 10 -log_expdata True -num_trajec 100 -meta_epochs 50 -output_file output_dims"
                    done
                done
            done
        done
    done
done