# !/bin/bash

for input_dim in 10 
do
    for output_dim in 1
    do
        for non_linear in True 
        do
            for num_meta_params in 2
            do
                for type in activity
                do
                    for len_trajec in 3 
                    do
                        bsub -n 8 -o cluster_logs.out -J "scalability" -q local "python exps/scalability/main.py \
                        -input_dim $input_dim -output_dim $output_dim -non_linear $non_linear -num_meta_params $num_meta_params \
                        -type $type -len_trajec $len_trajec -log_expdata True -num_trajec 20 -meta_epochs 2 -output_file cluster_exps"
                    done
                done
            done
        done
    done
done