#!/bin/bash
# job submission loop 

# fixing gcc problem (install clang)
#conda install clang_osx-64 clangxx_osx-64 -c anaconda

#run sims for bandit combined - 1 chain only
for sim_num in {0..30}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 100 -R 32 python power_bandit4arm_combined.py pt $sim_num 70 300
fsl_sub -T 100 -R 32 python power_bandit4arm_combined.py hc $sim_num 70 300
done
# fsl_sub -T 300 -R 64 -s openmp,2 python power_bandit4arm_combined.py pt $sim_num 90 300

#run sims for bandit
for sim_num in {0..30}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 300 -R 64 python power_bandit4arm_lapse.py pt $sim_num 70 300
fsl_sub -T 300 -R 64 python power_bandit4arm_lapse.py hc $sim_num 70 300
done

# run sims for generalise
for sim_num in {0..30}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 250 -R 64 python power_generalise_gs.py pt $sim_num 70 300
fsl_sub -T 250 -R 64 python power_generalise_gs.py hc $sim_num 70 300
done

# run sims for motor adapt
for sim_num in {0..50}
do
echo "submitted job simulation with seed $sim_num "
fsl_sub -T 10 -R 64 python power_motoradapt_single.py hc $sim_num 70 15
fsl_sub -T 10 -R 64 python power_motoradapt_single.py pt $sim_num 70 15
done

# hack conda environment
cp -r ../hBayesDM/commons/* /home/fs0/syzhang/.conda/envs/pystan/lib/python3.9/site-packages/hbayesdm/common/