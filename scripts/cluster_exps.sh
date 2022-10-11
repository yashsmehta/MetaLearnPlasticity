# !/bin/bash

for input_dim in 2 10 100
do
    for output_dim in 1 10
    do
        for non_linear in True False
        do
            for type in weight activity
            do
                for num_meta_params in 2 5 10 27
                do
                    for len_trajec in 2 10 50
                    do
                        bsub -n 5 -o cluster_logs.out -gpu "num=1" -J "scalability" -q gpu_rtx "python exps/scalability/main.py \
                        -input_dim $input_dim -output_dim $output_dim -non_linear $non_linear -num_meta_params $num_meta_params \
                        -type $type -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 250 -output_file cluster_exps"
                    done
                done
            done
        done
    done
done
