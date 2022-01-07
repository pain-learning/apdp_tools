"""
calculate effect size from params
"""
import numpy as np

# bandit
param_dict_hc = {
    'Arew': 0.569,  # reward learning rate
    'Apun': 0.016,  # punishment learning rate
    'R':    9.417,  # reward sensitivity
    'P':    5.778,  # punishment sensitivity
    'xi':   0.026,  # lapse
    'd':    0.469   # decay
}

# assumed patient parameters 
param_dict_pt = {
    'Arew': 0.585,  # reward learning rate
    'Apun': 0.106,  # punishment learning rate
    'R':    12.750,  # reward sensitivity
    'P':    9.953,  # punishment sensitivity
    'xi':   0.105,   # lapse
    'd':    0.603   # decay
}

# control sd
sd_dict_hc= {
    'Arew': 0.178,  # reward learning rate
    'Apun': 0.018,  # punishment learning rate
    'R':    5.031,  # reward sensitivity
    # 'P':    1.715,  # punishment sensitivity
    'P':    3.715,  # punishment sensitivity
    'xi':   0.003,   # lapse
    'd':    0.261   # decay
}

# patient sd
sd_dict_pt = {
    'Arew': 0.300,  # reward learning rate
    'Apun': 0.010,  # punishment learning rate
    'R':    5.911,  # reward sensitivity
    # 'P':    1.577,  # punishment sensitivity
    'P':    2.577,  # punishment sensitivity
    'xi':   0.043,   # lapse
    'd':    0.311   # decay
}

eff_bandit_p = (param_dict_hc['P']-param_dict_pt['P'])/np.mean([sd_dict_hc['P'], sd_dict_pt['P']])
eff_bandit_xi = (param_dict_hc['xi']-param_dict_pt['xi'])/np.mean([sd_dict_hc['xi'], sd_dict_pt['xi']])

print(f'bandit punishment sensitivity effect={eff_bandit_p:.3f}')
print(f'bandit lapse effect={eff_bandit_xi:.3f}')

#generalse
param_dict_hc = {
    'sigma_a': 0.45,  # generalisation param for shock
    'sigma_n': 0.06,  # generalisation param for no shock
    'eta':    0.17,     # p_h dynamic learning rate
    'kappa':  0.75,    # p_h dynamic learning rate
    'beta': 9.5,       # softmean beta
    'bias': 0.3      # softmean bias
}
# hc sd
sd_dict_hc = {
    'sigma_a': 0.05,  # generalisation param for shock
    'sigma_n': 0.01,  # generalisation param for no shock
    'eta':    0.1,     # p_h dynamic learning rate
    'kappa':  0.2,    # p_h dynamic learning rate
    'beta': 2,       # softmean beta
    'bias': 0.1      # softmean bias
}
# patient params
param_dict_pt = {
    'sigma_a': 0.85,  # generalisation param for shock
    'sigma_n': 0.03,  # generalisation param for no shock
    'eta':    0.18,     # p_h dynamic learning rate
    'kappa':  0.76,    # p_h dynamic learning rate
    'beta': 4.3,       # softmean beta
    'bias': 0.3      # softmean bias
}
# patient sd
sd_dict_pt = {
    'sigma_a': 0.05,  # generalisation param for shock
    'sigma_n': 0.01,  # generalisation param for no shock
    'eta':    0.10,     # p_h dynamic learning rate
    'kappa':  0.2,    # p_h dynamic learning rate
    'beta': 2,       # softmean beta
    'bias': 0.1      # softmean bias
}

eff_gen_sigma = (param_dict_hc['sigma_a']-param_dict_pt['sigma_a'])/np.mean([sd_dict_hc['sigma_a'], sd_dict_pt['sigma_a']])
eff_gen_beta = (param_dict_hc['beta']-param_dict_pt['beta'])/np.mean([sd_dict_hc['beta'], sd_dict_pt['beta']])

print(f'gen sigma a effect={eff_gen_sigma:.3f}')
print(f'gen beta effect={eff_gen_beta:.3f}')

# motor
param_dict_hc = {
    'A_retention': 2.7,  # retention rate 0.92
    'B_learning': -1.3,  # learning rate 0.33
    'norm_sig': .6 # sd of individual trajectory 1.65
}
# patient parameters (made up based on Takiyama 2016)
param_dict_pt = {
    'A_retention': 1.1,  # retention rate 0.81
    'B_learning': -0.1,  # learning rate 0.47
    'norm_sig': .9 # sd of individual trajectory 1
}
# healthy control sd
sd_dict_hc = {
    'A_retention': 0.8,  # retention rate
    'B_learning': 0.6,  # learning rate
    'norm_sig': 1.8 # sd of individual trajectory 
}
# patient sd
sd_dict_pt = {
    'A_retention': 0.8,  # retention rate
    'B_learning': 0.5,  # learning rate
    'norm_sig': 1.6 # sd of individual trajectory 
}

eff_mt_A = (param_dict_hc['A_retention']-param_dict_pt['A_retention'])/np.mean([sd_dict_hc['A_retention'], sd_dict_pt['A_retention']])
eff_mt_B = (param_dict_hc['B_learning']-param_dict_pt['B_learning'])/np.mean([sd_dict_hc['B_learning'], sd_dict_pt['B_learning']])

print(f'mt A effect={eff_mt_A:.3f}')
print(f'mt B effect={eff_mt_B:.3f}')

# all mean
eff_all = [eff_bandit_p, eff_bandit_xi, eff_gen_beta, eff_gen_sigma, eff_mt_A, eff_mt_B]
print(f'mean of all effects={np.mean(np.abs(eff_all)):.3f}')
print(f'median of all effects={np.median(np.abs(eff_all)):.3f}')
