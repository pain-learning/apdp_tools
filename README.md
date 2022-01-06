# Quantitative Cognitive Tests (QCT) tools

This repository is the code used for data simulation and analysis for pain (ADPD consortium project).

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

Or install [hBayesDM](https://hbayesdm.readthedocs.io/en/v1.0.1/) to get Stan and pyStan setup:

```
pip install hbayesdm
```
   
## Usage

To simulate data and fit  for bandit task, run the main script:

```train
python power_bandit4arm_combined.py pt 0 70 300
```

* pt - simulate patients (or use hc for controls). Change parameters inside main script.
* seed number 0
* simulate 70 participants
* each participant to complete 300 trials

Run `power_generalise_gs.py` for generalisation task, or `power_motoradapt_single.py` for motor adaptation task. 

Please note Stan can take several hours to run for a large number of subjects/trials on a cluster. And evaluation requires at least simulation from at least 30 different random seeds (per task).

## Evaluation

To evaluate effect size and fitted parameter distribution, run:

```eval
python compare_hdi.py bandit
```

* model name bandit (or use generalise/motoradapt for other tasks)
  
Output plots and statistics are in `./figs`.

## License

This project is licensed under [MIT](https://opensource.org/licenses/MIT) license.