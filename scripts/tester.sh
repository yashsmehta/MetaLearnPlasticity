# !/bin/bash

for input_dim in 10 
do
    for output_dim in 1
    do
        for non_linear in True 
        do
            for plasticity_rule in hebbian
            do
                for type in activity weight
                do
                    for len_trajec in 10
                    do
                        bsub -n 4 -o out -J "Benchmark" -q local "python exps/scalability_meta_learning_plasticity/main.py \
                        -input_dim $input_dim -output_dim $output_dim -non_linear $non_linear \
                        -plasticity_rule $plasticity_rule -type $type -len_trajec $len_trajec -log_expdata True -num_trajec 50 -meta_epochs 3" 
                    done
                done
            done
        done
    done
done
