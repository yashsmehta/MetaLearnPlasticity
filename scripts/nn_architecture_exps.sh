# !/bin/bash

for input_dim in 1 112 223 334 445 556 667 778 889 1000 
do
    for output_dim in 50
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
                        -jobid $jobid -noise_scale $noise_scale -len_trajec 50 -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file nn_architecture_exps"
                    done
                done
            done
        done
    done
done


for input_dim in 500 
do
    for output_dim in 1 112 223 334 445 556 667 778 889 1000 
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
                        -jobid $jobid -noise_scale $noise_scale -len_trajec 50 -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file nn_architecture_exps"
                    done
                done
            done
        done
    done
done