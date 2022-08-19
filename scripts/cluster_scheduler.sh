# !/bin/bash

# benchmarking on CPU
for m in 10 1257 2505 3752 5000 
do
    for time_steps in 10 2507 5005 7502 10000 
    do
        for density in 0.1 0.25 0.5 0.75 1.0
        do
            bsub -n 8 -o out -J "Benchmark" -q local "python playground/benchmarking.py \
            --output_nodes $m --time_steps $time_steps --density $density \
             > playground/out/cpu$m.$timesteps.$density" 
        done
    done
done


# benchmarking on GPU
for m in 10 1257 2505 3752 5000 
do
    for time_steps in 10 2507 5005 7502 10000 
    do
        for density in 0.1 0.25 0.5 0.75 1.0
        do
            bsub -n 12 -o out -J "Benchmark" -q 'gpu_tesla' -gpu "num=1" "python playground/benchmarking.py \
                --output_nodes $m --time_steps $time_steps --density $density \
             > playground/out/gpu$m.$timesteps.$density" 
        done
    done
done