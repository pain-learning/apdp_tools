"""
plot motor circle demo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from power_motorcircle_basic import model_motorcircle

def plot_circles(task_params):
    """plotting target and penalty circles"""
    # plot circles
    fig, ax = plt.subplots()
    c_penalty = plt.Circle((task_params['x_penalty'], task_params['y_penalty']), task_params['radius'], color='g', fill=True)
    c_target = plt.Circle((task_params['x_target'], task_params['y_target']), task_params['radius'], color='y', fill=False, linewidth=8)
    plt.gca().add_patch(c_penalty)
    plt.gca().add_patch(c_target)
    # plot centres
    mult = 1.2
    # plt.vlines(task_params['x_target'], ymax=task_params['y_target']+task_params['radius']*mult, ymin=task_params['y_target']-task_params['radius']*mult, colors='k')
    # plt.vlines(task_params['x_penalty'], ymax=task_params['y_penalty']+task_params['radius']*mult, ymin=task_params['y_penalty']-task_params['radius']*mult, colors='k')
    # write penalty value
    # plt.text(task_params['x_penalty'], task_params['y_penalty'], s=task_params['penalty_val'], size=40, ha='center', va='center')
    return fig, ax

def plot_simulation(task_params, df_hc, df_pt, legend=True):
    """plot simulated trajectories"""
    # plotting circles
    fig, ax = plot_circles(task_params)
    # fig, ax = plt.subplots()
    # plot sim
    plt.scatter(df_hc['x'], df_hc['y'], s=30, marker='>', alpha=0.4, c='k', zorder=10, edgecolors='None')
    plt.scatter(df_pt['x'], df_pt['y'], s=30, marker='o', alpha=0.4, c='b', zorder=12, edgecolors='None')
    plt.scatter(np.mean(df_hc['x']), np.mean(df_hc['y']), s=80, marker='>', edgecolors='k', facecolors='k', label='Control', zorder=13)
    plt.scatter(np.mean(df_pt['x']), np.mean(df_pt['y']), s=80, marker='o', edgecolors='k', facecolors='b', label='Pain', zorder=15)
    if legend:
        plt.legend(fontsize=12, loc='lower left')
    plt.xlim(-2.5, 1.5)
    ax.set_aspect('equal')
    plt.axis('off')

def plot_violin(df_hc, df_pt):
    """plot violin distribution"""
    df_hc['group'] = 'Control'
    df_pt['group'] = 'Pain'
    df = pd.concat([df_hc, df_pt])
    # plot
    fig, ax = plt.subplots()
    # g= sns.violinplot(data=df, y='x', x="group", split=True, inner="quart", linewidth=1,palette={"Pain": "b", "Control": ".85"}, ax=ax)
    g= sns.kdeplot(data=df, x='x', hue="group", fill=True, alpha=0.5, ax=ax, linewidth=0, palette={"Pain": "b", "Control": "k"})
    sns.despine(left=True)
    g.set(ylabel=None)
    ax.get_legend().remove()
    ax.set(yticklabels=[], xticklabels=[])
    ax.tick_params(left=False)
    # ax.set_ylim([0,1])
    # ax[n].tick_params(axis='y', labelsize=8) 
    # if model_name=='motoradapt' and n==2:
    #     g.set(yticklabels=[])
    g.set(xlabel=None)



if __name__ == "__main__":
    ########### plotting
    n = 100
    task_near = {
        'x_target': 0, # target circle x
        'y_target': 0, # target circle y
        'radius': 1, # radius
        'x_penalty': -0.7, # penalty circle x
        'y_penalty': 0, # penalty circle y
        'penalty_val': -500 # penalty value
    }
    task_far = {
        'x_target': 0, # target circle x
        'y_target': 0, # target circle y
        'radius': 1, # radius
        'x_penalty': -1.2, # penalty circle x
        'y_penalty': 0, # penalty circle y
        'penalty_val': -500 # penalty value
    }
    params_hc = {
        'loss_sens': 0.1, # sensitivity to loss
        'perturb': 0.1, # cov 
    }
    params_pt = {
        'loss_sens': 0.12, # sensitivity to loss
        'perturb': 0.13, # cov 
    }
    # simulation
    df_hc = model_motorcircle(task_near, params_hc, subjID=0, num_trial=n)
    df_pt = model_motorcircle(task_near, params_pt, subjID=0, num_trial=n)
    # plotting trajectories
    plot_simulation(task_near, df_hc, df_pt, legend=False)
    plt.savefig('./figs/task_near.png', bbox_inches='tight', transparent=True)
    # plot combined distrib
    plot_violin(df_hc, df_pt)
    plt.savefig('./figs/task_near_dist.png', bbox_inches='tight', transparent=True)

    # simulation
    df_hc = model_motorcircle(task_far, params_hc, subjID=0, num_trial=n)
    df_pt = model_motorcircle(task_far, params_pt, subjID=0, num_trial=n)
    # plotting trajectories
    plot_simulation(task_far, df_hc, df_pt)
    plt.savefig('./figs/task_far.png', bbox_inches='tight', transparent=True)

    # plot combined distrib
    plot_violin(df_hc, df_pt)
    plt.savefig('./figs/task_far_dist.png', bbox_inches='tight', transparent=True)