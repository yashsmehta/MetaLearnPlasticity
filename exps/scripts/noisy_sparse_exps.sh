# !/bin/bash

for input_dim in 50
do
    for output_dim in 50
    do
        for upto_ith_order in 3
        do
            for noise_scale in 0 0.2 0.4 0.6 0.8 1
            do
                for sparsity in 1 0.9 0.8 0.7 0.6 0.5
                do
                    for jobid in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
                    do
                        bsub -n 4 -o cluster_logs.out -J "noispa" -q local "python exps/scalability/main.py \
                        -input_dim $input_dim -output_dim $output_dim -upto_ith_order $upto_ith_order -sparsity $sparsity\
                        -jobid $jobid -noise_scale $noise_scale -len_trajec 10 -log_expdata True -num_trajec 200 -meta_epochs 50 -output_file sparse-noise"
                    done
                done
            done
        done
    done
done