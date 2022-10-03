# !/bin/bash

for input_dim in 10 50 100
do
    for output_dim in 1 10 100
    do
        for non_linear in True False
        do
            for plasticity_rule in hebbian oja random
            do
                for type in activity weight
                do
                    for len_trajec in 2 10 50 100
                    do
                        bsub -n 4 -o out -J "Benchmark" -q local "python exps/scalability_meta_learning_plasticity/main.py \
                        -input_dim $input_dim -output_dim $output_dim -non_linear $non_linear \
                        -plasticity_rule $plasticity_rule -type $type -len_trajec $len_trajec -log_expdata True -num_trajec 250 -meta_epochs 250"
                    done
                done
            done
        done
    done
done
