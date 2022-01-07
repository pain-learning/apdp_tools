# Quantitative Cognitive Tests (QCT) tools

This repository is the code used for data simulation and analysis for pain (ADPD consortium project).

## Requirements

Please clone this repo to your machine. Create a new conda environment `apdp_tools` with the given requirements:

```setup
conda env create -f environment.yml
```

Activate conda environment before running the code:

```setup
conda activate apdp_tools
```

If pystan3 throws a compiler error, install the latest GCC and G++ compilers from conda within the activated conda environment using the following (tested on Linux):

```setup
conda install -c conda-forge gcc_linux-64
conda install -c conda-forge gxx_linux-64
```
   
## Simulations

Data simulation is important to verify model assumptions. To simulate data and fit for bandit task, run the main script:

```train
python simulations/sim_bandit4arm_combined.py pt 0 3 100
```

The example `sim_bandit4arm_combined` above runs the data simulation of a 4-arm bandit task (Seymour et al 2012 JNS) with given input parameters, fitted the sumulated data hierarchically using Stan and produce the parameter distribution in output sample traces. It has the following changable parameters:

* pt - simulate patients (or use hc for controls). Change parameters inside main script.
* seed number 0 (for power calcualtion, change seed to simulate multiple times)
* simulate 3 participants (or more if you like)
* each participant to complete 100 trials (or a different number of your choice)

Checking the fitted model parameters against the input model parameters (pt and hc dictionaries in the simulation code) can give an idea of how well the model fitting works. In addition, there are a list of other tasks:
* `sim_generalise_gs.py` for generalisation instrumental avoidance task (Norbury et al. 2018 eLife)
* `sim_motoradapt_single.py` for motor adaptation task (Takiyama 2016)
* `sim_motorcircle_basic.py` for motor decision task (Kurniawan 2010)
* `sim_bandit4arm_lapse.py` for the 4-arm bandit task as in the example, but with a simplified model.

For power calculation, the simulation can be run n times with different random seeds (i.e. run the study n times) to estimate significance in group differences. For example, the loop below runs a simulation 50 times, each with 70 patients and controls completing 300 trials.

```
for sim_num in {0..50}
do
echo "submitted job simulation with seed $sim_num "
python simulations/sim_bandit4arm_combined.py pt $sim_num 70 300
python simulations/sim_bandit4arm_combined.py hc $sim_num 70 300
done
```

Please note Stan can take several hours to run for a large number of subjects/trials on a cluster. And evaluation requires at least simulation from at least 30 different random seeds (per task).

## Evaluation

For power calculation, to evaluate effect size and fitted parameter distribution, run:

```eval
python compare_hdi.py bandit
```

* model name bandit (or use generalise/motoradapt for other tasks)
  
Output plots and statistics are in `./figs`.

## License

This project is licensed under [MIT](https://opensource.org/licenses/MIT) license.