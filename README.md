# Quantitative Cognitive Tests (QCT) tools

This repository is the code used for data simulation and analysis for pain (ADPD consortium project).

## Requirements

Please clone this repo to your machine. Create a new conda environment `apdp_tools` with the given requirements (nobuild and explicit also available):

```setup
conda env create -f environment.yml
```

Activate conda environment before running the code:

```setup
conda activate apdp_tools
```

If pystan3 throws a compiler error, install the latest GCC and G++ compilers from conda within the activated conda environment using the following (tested on Linux CentOS 7):

```setup
conda install -c conda-forge gcc_linux-64
conda install -c conda-forge gxx_linux-64
```
   
## Simulations

Data simulation is important to verify model assumptions. To simulate data and fit for bandit task, run the main script:

```train
python simulations/sim_bandit3arm_combined.py pt 0 3 100
```

The example `sim_bandit3arm_combined` above runs the data simulation of a 3-arm bandit task (Seymour et al 2012 JNS) with given input parameters, fitted the sumulated data hierarchically using Stan and produce the parameter distribution in output sample traces. It has the following changable parameters:

* pt - simulate patients (or use hc for controls). Change parameters inside main script.
* seed number 0 (for power calcualtion, change seed to simulate multiple times)
* simulate 3 participants (or more if you like)
* each participant to complete 100 trials (or a different number of your choice)

Checking the fitted model parameters against the input model parameters (pt and hc dictionaries in the simulation code) can give an idea of how well the model fitting works. In addition, there are a list of other tasks:
* `sim_generalise_gs.py` for generalisation instrumental avoidance task (Norbury et al. 2018 eLife)
* `sim_motoradapt_single.py` for motor adaptation task (Takiyama 2016)
* `sim_motorcircle_basic.py` for motor decision task (Kurniawan 2010)

For power calculation, the simulation can be run n times with different random seeds (i.e. run the study n times) to estimate significance in group differences. For example, the loop below runs a simulation 50 times, each with 70 patients and controls completing 240 trials.

```
for sim_num in {0..50}
do
echo "submitted job simulation with seed $sim_num "
python simulations/sim_bandit3arm_combined.py pt $sim_num 70 240
python simulations/sim_bandit3arm_combined.py hc $sim_num 70 240
done
```

Please note Stan can take several hours to run for a large number of subjects/trials on a cluster. And evaluation requires at least simulation from at least 30 different random seeds (per task).

## Visualisation

Once the simulations with different study design arguments are finished, you can proceed with power calculation, to evaluate effect size and fitted parameter distribution, run:

```eval
python visualisation/hdi_compare.py 160 3 sim_bandit3arm_combined 2 50
```
Where the first argument is the number of trials used in the simulation (160), the second argument is the number of subjects in the simulation (3), the third argument corresponds to model name (sim_bandit3arm_combined), the fourth specifies the number of simulations minus 1, and the last number of permutations.

It's recommended to run at least 10 simulations (`draw_idx=10` / 4th argument above) and permutate 100 times (`n_perm=100` / last argument above) to ensure a good estimation of HDI stats.
  
Output plots and statistics are in `./figs`.

Other models names available are: 'motorcircle', 'motoradapt', 'bandit3arm_combined', 'generalise'.

## Using your own data

It's also possible to collect your own data and use that for analysis using this toolbox. Simply follow these steps:

* Fork the study repository on [Pavlovia](https://pavlovia.org) to your account
* Run the online study with your subject pool (requires Pavlovia license, see their website for details)
* Clone the forked repository locally in the same directory as `apdp_tools` 
* Run code in `data_transform` to convert Pavlovia data into model-compatible structure. For example, for the generalisation task, the stan compatible txt file can be found in `transformed_data`

```eval
python data_transform/convert_data.py generalise ../generalisation_py 190
```

Currently, available task names include `generalise`, `bandit3arm`, `circlemotor`, which is specified in argument 1 above. The 2nd argument is the relative path of your Pavlovia task directory (forked from our source, cloned to your local machine). Alternatively, you can write your own data conversion function to match any changes you've made to the task. Finally, the last optional argument specifies how many trials to expect in the dataset for each participant to reject files which are not complete, if not specified, incomplete files will be included.

* Fit your data to models using scripts in `data_fit` (importing existing functions in `Simulation` above, but modified data input path), and visualise the results following `Visualisation`

## License

This project is licensed under [MIT](https://opensource.org/licenses/MIT) license.
