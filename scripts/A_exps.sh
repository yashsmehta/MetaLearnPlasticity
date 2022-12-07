# !/bin/bash

for input_dim in 50 200
do
    for output_dim in 50 200
    do
        for upto_ith_order in 3
        do
            for noise_scale in 0 0.2 
            do
                for sparsity in 1 0.9 0.8
                do
                    for jobid in 0 1
                    do
                        bsub -n 6 -o cluster_logs.out -J "As" -q local "python exps/scalability/main.py \
                        -input_dim $input_dim -output_dim $output_dim -upto_ith_order $upto_ith_order -sparsity $sparsity\
                        -jobid $jobid -noise_scale $noise_scale -len_trajec 50 -log_expdata True -num_trajec 250 -meta_epochs 200 -output_file A_exps"
                    done
                done
            done
        done
    done
done