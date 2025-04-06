import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.append('../utils')

from utils.generate_charts import *
from utils.marcenko_pastur import *

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import load_model
from tensorflow.random import normal
from scipy.stats import skew, kurtosis, wasserstein_distance, kstest


from utils.gan import GAN
from utils.tcn import make_TCN, receptive_field_size
from utils.preprocessing import *
from utils.quant_strats import *
from utils.utils import *

from scipy.stats import t

import datetime 
import shutil

plt.rcParams['axes.labelsize'] = 8  # Default fontsize for x and y labels
plt.rcParams['xtick.labelsize'] = 7  # Default fontsize for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 7  # Default fontsize for y-axis tick labels
plt.rcParams['legend.fontsize'] = 7 # Default fontsize for legend
plt.rcParams['axes.titlesize'] = 8  # Default fontsize for plot title

gaussian_color = 'green'
original_color = 'blue'
synthetic_original_color = 'dodgerblue'
synthetic_original_color_line = 'red'
### params

# seed
seed = 42

# directory for importing the associated generator
gan_dir = './GAN_training/GAN_benchmark'

## generate synthetic data of long length len_syn_data
len_syn_data = 30200
window_len = 63
n_assets_to_analyze = 5
n_market_gen = 100

np.random.seed(seed)
tf.random.set_seed(seed)


chosen_pc_returns_df = pd.read_csv(gan_dir+'/chosen_pc_returns_df.csv', index_col=0)
residuals_df = pd.read_csv(gan_dir+'/residuals_df.csv', index_col=0)
returns_df = pd.read_csv(gan_dir+'/returns_df.csv', index_col=0)
factor_based_returns_df = pd.read_csv(gan_dir+'/factor_based_returns_df.csv', index_col=0)
standardized_returns_df = pd.read_csv(gan_dir+'/standardized_returns_df.csv', index_col=0)
eigenvalues = np.load(gan_dir+'/eigenvalues.npy')
eigenvectors = np.load(gan_dir+'/eigenvectors.npy')
with open(gan_dir+'/residuals_params.pkl', 'rb') as f:
    residual_params = pickle.load(f)

eval_dir = gan_dir + '/evaluation'

# Delete the folder if it exists
if os.path.exists(eval_dir):
    shutil.rmtree(eval_dir)


cluster_count = len([name for name in os.listdir(gan_dir) if os.path.isdir(os.path.join(gan_dir, name))])

os.makedirs(eval_dir)

generators = []
clusters = []

for c in range(cluster_count):
    print('===== Cluster_'+str(c+1)+' =====')
    actual_cluster_label = 'Cluster_'+str(c+1)
    cluster = pd.read_csv(gan_dir+'/Cluster_'+str(c+1)+'.csv', index_col=0, header=[0,1])
    generator = load_model(gan_dir+'/cluster_'+str(c+1)+'/trained_generator')
    
    generators.append(generator)
    clusters.append(cluster)

    n_pc_in_cluster = cluster.shape[1]
    noise = normal([1, 1, len_syn_data + 63 - 1, 3])
    syn_cluster = pd.DataFrame(generator(noise).numpy().squeeze().T)
    syn_cluster = (syn_cluster-syn_cluster.mean())/syn_cluster.std()


    ## autocorrelations
    real_linear_ac = []
    real_abs_ac = []
    real_lev_ac = []

    for i in range(n_pc_in_cluster):
        real_linear_ac.append(my_acf(cluster.iloc[:,i].values, window_len))
        real_abs_ac.append(my_acf(abs(cluster.iloc[:,i].values)**2, window_len))
        real_lev_ac.append(my_acf(cluster.iloc[:,i].values, window_len, lev=True))

    real_linear_ac = np.array(real_linear_ac)[:,1:] # omit first point
    real_abs_ac = np.array(real_abs_ac)[:,1:]
    real_lev_ac = np.array(real_lev_ac)[:,1:]

    syn_linear_ac = my_acf(syn_cluster.iloc[:,0].values, window_len)[1:]
    syn_abs_ac = my_acf(abs(syn_cluster.iloc[:,0].values)**2, window_len)[1:]
    syn_lev_ac = my_acf(syn_cluster.iloc[:,0].values, window_len, lev=True)[1:]

    fig = plt.figure(layout="tight", figsize=(18,7.5))

    gs = GridSpec(6, 3, figure=fig)

    ax1 = fig.add_subplot(gs[:2, 2])
    ax2 = fig.add_subplot(gs[2:4, 2])
    ax3 = fig.add_subplot(gs[4:, 2])
    ax4 = fig.add_subplot(gs[:3, 0])
    ax5 = fig.add_subplot(gs[3:, 0])
    ax6 = fig.add_subplot(gs[:3, 1])
    ax7 = fig.add_subplot(gs[3:, 1])

    ax1.plot(real_linear_ac.T, alpha=.2, color='red', label='_nolegend_')
    ax1.plot(syn_linear_ac, alpha=1, color='grey', label='Synthetic');
    ax1.set_title('Autocorrelations of returns')
    ax1.set_xlabel('$\\tau$ (days)')

    ax2.plot(real_abs_ac.T, alpha=.2, color='red')
    ax2.plot(syn_abs_ac, alpha=1, color='grey');
    ax2.set_title('Autocorrelations of squared returns')
    # ax[1].set_ylim(-.5,7);
    ax2.set_xlabel('$\\tau$ (days)')

    ax3.plot(real_lev_ac.T, alpha=.2, color='red')
    ax3.plot(syn_lev_ac, alpha=1, color='grey')
    ax3.set_title('Cross-correlation between returns and squared returns')
    ax3.set_xlabel('$\\tau$ (days)')

    ax4.hist(cluster.values.flatten(), bins=100, density=True, color='red', alpha=.5, label='Historical');
    ax4.hist(syn_cluster, bins=100, density=True, color='grey', alpha=.5, label='Syntetic');
    ax4.legend()
    ax4.set_title('Histogram of cluster returns')
    ax4.set_ylabel('Density')

    for i in range(n_pc_in_cluster):
        x_real = np.sort(cluster.iloc[:,i])
        y_real = np.linspace(0,1, len(x_real))
        ax5.plot(x_real,y_real, color='red', alpha=.2)

    x_syn = np.sort(syn_cluster.squeeze().values)
    y_syn = np.linspace(0,1, len(x_syn))
    ax5.plot(x_syn,y_syn, color='grey', alpha=1)
    ax5.set_ylabel('Cumulative density')
    ax5.set_title('Cumulative density of cluster returns')
    ax5.set_xlim(np.quantile(x_real,.01).item(),np.quantile(x_real,.99).item())

    for i in range(n_pc_in_cluster):

        ### Hill estimator
        n_sample = cluster.shape[0]
        # Set the range of k values
        k_values = np.arange(int(n_sample*.004), int(n_sample*.05), 1)

        # Initialize arrays to store Hill estimates and confidence intervals
        hill_estimates_loss_real = np.zeros_like(k_values, dtype=float)
        conf_intervals_lower_loss_real = np.zeros_like(k_values, dtype=float)
        conf_intervals_upper_loss_real = np.zeros_like(k_values, dtype=float)

        x_real = cluster.iloc[:,i].squeeze().values

        for i, k in enumerate(k_values):
            hill_estimate_loss_s, standard_error_loss_s = hill_estimator(x_real, k)

            hill_estimates_loss_real[i] = hill_estimate_loss_s
            conf_intervals_lower_loss_real[i] = hill_estimate_loss_s - 1.96 * standard_error_loss_s  # 1.96 is the z-value for a 95% confidence interval
            conf_intervals_upper_loss_real[i] = hill_estimate_loss_s + 1.96 * standard_error_loss_s

        # Plot Hill estimates with confidence intervals
        ax6.plot(k_values, hill_estimates_loss_real, label='_nolegend_', color='red', alpha=.2)
        ax6.set_title('Hill Tail Index')
        ax6.set_xlabel('k')

    x_syn = syn_cluster.squeeze().values

    hill_estimates_loss_syn = np.zeros_like(k_values, dtype=float)
    conf_intervals_lower_loss_syn = np.zeros_like(k_values, dtype=float)
    conf_intervals_upper_loss_syn = np.zeros_like(k_values, dtype=float)

    for i, k in enumerate(k_values):
        hill_estimate_loss_r, standard_error_loss_r = hill_estimator(x_syn, k)

        hill_estimates_loss_syn[i] = hill_estimate_loss_r
        conf_intervals_lower_loss_syn[i] = hill_estimate_loss_r - 1.96 * standard_error_loss_r  # 1.96 is the z-value for a 95% confidence interval
        conf_intervals_upper_loss_syn[i] = hill_estimate_loss_r + 1.96 * standard_error_loss_r

    ax6.plot(k_values, hill_estimates_loss_syn, color='grey')

    x_normal = np.random.standard_normal(n_sample)

    hill_estimates_loss_normal = np.zeros_like(k_values, dtype=float)
    conf_intervals_lower_loss_normal = np.zeros_like(k_values, dtype=float)
    conf_intervals_upper_loss_normal = np.zeros_like(k_values, dtype=float)

    for i, k in enumerate(k_values):
        hill_estimate_loss_r, standard_error_loss_r = hill_estimator(x_normal, k)

        hill_estimates_loss_normal[i] = hill_estimate_loss_r
        conf_intervals_lower_loss_normal[i] = hill_estimate_loss_r - 1.96 * standard_error_loss_r  # 1.96 is the z-value for a 95% confidence interval
        conf_intervals_upper_loss_normal[i] = hill_estimate_loss_r + 1.96 * standard_error_loss_r

    ax6.plot(k_values, hill_estimates_loss_normal, label='Gaussian', color='green', ls='-.', alpha=0.5)
    ax6.legend()


    disc = np.linspace(0.001,0.999, 1000)
    ax7.scatter(np.quantile(cluster.values.flatten(), disc), np.quantile(syn_cluster.values.flatten(), disc), s=2, color='red', alpha=0.5)
    ax7.plot(np.linspace(ax7.get_xlim()[0],ax7.get_xlim()[1], 10), np.linspace(ax7.get_xlim()[0],ax7.get_xlim()[1], 10), alpha=0.5, color='grey')
    ax7.set_xlabel('Historical quantiles')
    ax7.set_ylabel('Synthetic quantiles')
    ax7.set_title('Q-Q plot')

    fig.suptitle(actual_cluster_label+': Evaluation of the synthetic data', fontsize=10)

    plt.tight_layout()

    plt.savefig(eval_dir+'/evaluation_cluster_'+str(c+1)+'.png')

    plt.close()

    df_real = cluster.describe().T
    df_real['kurtosis'] = cluster.kurt().values
    df_real['skew'] = cluster.skew().values
    df_real.drop('count', axis=1, inplace=True)

    df_syn = syn_cluster.describe().T
    df_syn['kurtosis'] = syn_cluster.kurt().values
    df_syn['skew'] = syn_cluster.skew().values  
    df_syn.drop('count', axis=1, inplace=True)

    df_real.index = df_real.index.droplevel(1)
    df_syn.index = ['Synthetic']

    df_final = pd.concat([df_syn, df_real], axis=0).round(2).T

    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.table(cellText=df_final.values, colLabels=df_final.columns, rowLabels=df_final.index, loc='center', cellLoc='center')
    plt.title(actual_cluster_label+': Descriptive statistics of the synthetic data')

    plt.savefig(eval_dir+'/stats_cluster_'+str(c+1)+'.png')

    plt.close()


asset_dir = eval_dir + '/assets'

# Delete the folder if it exists
if os.path.exists(asset_dir):
    shutil.rmtree(asset_dir)

os.makedirs(asset_dir)

syn_asset_returns, _, _, _  = market_generator(generators = generators,
                                                clusters = clusters,
                                                residual_params = residual_params,
                                                len_syn = len_syn_data,
                                                eigenvalues = eigenvalues,
                                                eigenvectors = eigenvectors,
                                                returns_df = returns_df,
                                                random_state = None)  

asset_list = np.random.choice(syn_asset_returns.columns, n_assets_to_analyze, replace=False)

for asset in tqdm(asset_list):
    
    returns_syn = syn_asset_returns[asset].values
    returns_real = returns_df[asset].values

    ## autocorrelations
    real_linear_ac = []
    real_abs_ac = []
    real_lev_ac = []


    real_linear_ac.append(my_acf(returns_real, window_len))
    real_abs_ac.append(my_acf(returns_real**2, window_len))
    real_lev_ac.append(my_acf(returns_real, window_len, lev=True))

    real_linear_ac = np.array(real_linear_ac)[:,1:] # omit first point
    real_abs_ac = np.array(real_abs_ac)[:,1:]
    real_lev_ac = np.array(real_lev_ac)[:,1:]

    syn_linear_ac = my_acf(returns_syn, window_len)[1:]
    syn_abs_ac = my_acf(returns_syn**2, window_len)[1:]
    syn_lev_ac = my_acf(returns_syn, window_len, lev=True)[1:]

    fig = plt.figure(layout="tight", figsize=(12,5))

    gs = GridSpec(6, 3, figure=fig)

    ax0 = fig.add_subplot(gs[:3, 0])
    ax1 = fig.add_subplot(gs[:2, 2])
    ax2 = fig.add_subplot(gs[2:4, 2])
    ax3 = fig.add_subplot(gs[4:, 2])
    ax4 = fig.add_subplot(gs[:2, 1])
    ax5 = fig.add_subplot(gs[2:4, 1])
    ax6 = fig.add_subplot(gs[4:, 1])
    ax7 = fig.add_subplot(gs[3:, 0])

    returns_syn_windowed = np.array([returns_syn[i*n_sample:i*n_sample+n_sample] for i in range(int(len_syn_data//n_sample))]).T
    cum_returns_syn = np.cumprod(returns_syn_windowed+1, axis=0)
    cum_returns_syn = cum_returns_syn/cum_returns_syn[0,:] * 100

    cum_returns_real = np.cumprod(returns_real+1)
    cum_returns_real = cum_returns_real/cum_returns_real[0] * 100

    ax0.plot(cum_returns_real, color=original_color, label='Original', alpha=1, lw=1, ls='dotted', zorder=100)
    ax0.plot(cum_returns_syn, color=synthetic_original_color, label='Synthetic', alpha=.5, lw=.8)
    ax0.set_xlabel('$\\tau$ (days)')
    ax0.set_ylabel('Cumulative performance')
    ax0.set_yscale('log')
    handles, labels = ax0.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax0.legend(by_label.values(), by_label.keys())


    ax1.plot(real_linear_ac.T, alpha=1, color=original_color, label='_nolegend_', lw=1, ls='dotted', zorder=100)
    ax1.plot(syn_linear_ac, alpha=1, color=synthetic_original_color, label='Synthetic', lw=1, zorder=100);
    ax1.set_title('Autocorrelations of returns')
    ax1.set_xlabel('$\\tau$ (days)')

    ax2.plot(real_abs_ac.T, alpha=1, color=original_color, label='_nolegend_', lw=1, ls='dotted', zorder=100)
    ax2.plot(syn_abs_ac, alpha=1, color=synthetic_original_color, lw=1);
    ax2.set_title('Autocorrelations of squared returns')
    # ax[1].set_ylim(-.5,7);
    ax2.set_xlabel('$\\tau$ (days)')

    ax3.plot(real_lev_ac.T, alpha=1, color=original_color, label='_nolegend_', lw=1, ls='dotted', zorder=100)
    ax3.plot(syn_lev_ac, alpha=1, color=synthetic_original_color, lw=1);
    ax3.set_title('Cross-correlation between returns and squared returns')
    ax3.set_xlabel('$\\tau$ (days)')

    ax4.hist(returns_real, bins=100, density=True, color=original_color, alpha=.5, label='Historical');
    ax4.hist(returns_syn, bins=100, density=True, color=synthetic_original_color, alpha=.5, label='Syntetic');

    ax4.set_title('Histogram of asset returns')
    ax4.set_ylabel('Density')

    x_real = np.sort(returns_real)
    y_real = np.linspace(0,1, len(x_real))
    ax5.plot(x_real,y_real, color=original_color, alpha=.5, lw=.8)

    x_syn = np.sort(returns_syn)
    y_syn = np.linspace(0,1, len(x_syn))
    ax5.plot(x_syn,y_syn, color=synthetic_original_color, alpha=.5, lw=.8)
    ax5.set_title('Cumulative density of asset returns')
    ax5.set_xlim(np.quantile(x_real,.01).item(),np.quantile(x_real,.99).item())

    ### Hill estimator
    n_sample = len(returns_real)
    # Set the range of k values
    k_values = np.arange(int(n_sample*.004), int(n_sample*.05), 1)

    # Initialize arrays to store Hill estimates and confidence intervals
    hill_estimates_loss_real = np.zeros_like(k_values, dtype=float)
    conf_intervals_lower_loss_real = np.zeros_like(k_values, dtype=float)
    conf_intervals_upper_loss_real = np.zeros_like(k_values, dtype=float)

    x_real = returns_real

    for i, k in enumerate(k_values):
        hill_estimate_loss_s, standard_error_loss_s = hill_estimator(x_real, k)
        # hill_estimate_loss_r, standard_error_loss_r = hill_estimator(-x_real[x_real<0], k)

        hill_estimates_loss_real[i] = hill_estimate_loss_s
        conf_intervals_lower_loss_real[i] = hill_estimate_loss_s - 1.96 * standard_error_loss_s  # 1.96 is the z-value for a 95% confidence interval
        conf_intervals_upper_loss_real[i] = hill_estimate_loss_s + 1.96 * standard_error_loss_s

    # Plot Hill estimates with confidence intervals
    ax6.plot(k_values, hill_estimates_loss_real, label='_nolegend_', alpha=1, color=original_color, lw=1, ls='dotted', zorder=100)
    ax6.set_title('Hill Tail Index')
    ax6.set_xlabel('k')

    for i in range(returns_syn_windowed.shape[1]):

        x_syn = returns_syn_windowed[:,i]

        hill_estimates_loss_syn = np.zeros_like(k_values, dtype=float)
        conf_intervals_lower_loss_syn = np.zeros_like(k_values, dtype=float)
        conf_intervals_upper_loss_syn = np.zeros_like(k_values, dtype=float)

        for i, k in enumerate(k_values):
            hill_estimate_loss_r, standard_error_loss_r = hill_estimator(x_syn, k)

            hill_estimates_loss_syn[i] = hill_estimate_loss_r
            conf_intervals_lower_loss_syn[i] = hill_estimate_loss_r - 1.96 * standard_error_loss_r  # 1.96 is the z-value for a 95% confidence interval
            conf_intervals_upper_loss_syn[i] = hill_estimate_loss_r + 1.96 * standard_error_loss_r

        ax6.plot(k_values, hill_estimates_loss_syn, color=synthetic_original_color, alpha=.5, lw=.8)

    x_normal = np.random.standard_normal(n_sample)

    hill_estimates_loss_normal = np.zeros_like(k_values, dtype=float)
    conf_intervals_lower_loss_normal = np.zeros_like(k_values, dtype=float)
    conf_intervals_upper_loss_normal = np.zeros_like(k_values, dtype=float)

    for i, k in enumerate(k_values):
        hill_estimate_loss_r, standard_error_loss_r = hill_estimator(x_normal, k)

        hill_estimates_loss_normal[i] = hill_estimate_loss_r
        conf_intervals_lower_loss_normal[i] = hill_estimate_loss_r - 1.96 * standard_error_loss_r  # 1.96 is the z-value for a 95% confidence interval
        conf_intervals_upper_loss_normal[i] = hill_estimate_loss_r + 1.96 * standard_error_loss_r

    ax6.plot(k_values, hill_estimates_loss_normal, label='Gaussian', color='green', ls='-.', alpha=.5, lw=.8)
    ax6.legend()


    disc = np.linspace(0.001,0.999, 1000)
    ax7.scatter(np.quantile(returns_real, disc), np.quantile(returns_syn, disc), s=2, color=original_color, alpha=.5)
    ax7.plot(np.linspace(ax7.get_xlim()[0],ax7.get_xlim()[1], 10), np.linspace(ax7.get_xlim()[0],ax7.get_xlim()[1], 10), alpha=.5, color='grey', lw=.8)
    ax7.set_xlabel('Historical quantiles')
    ax7.set_ylabel('Synthetic quantiles')
    ax7.set_title('Q-Q plot')

    fig.suptitle(asset + ': Evaluation of the synthetic stock data', fontsize=10)
    plt.tight_layout()

    plt.savefig(asset_dir+'/evaluation_'+asset+'.png')
    plt.close()

est_linear_ac = []
est_abs_ac = []
est_lev_ac = []

real_linear_ac = []
real_abs_ac = []
real_lev_ac = []

scaled_linear_ac = []
scaled_abs_ac = []
scaled_lev_ac = []

for i in range(returns_df.shape[1]):
    est_linear_ac.append(my_acf(syn_asset_returns.iloc[:,i].values, window_len))
    est_abs_ac.append(my_acf(abs(syn_asset_returns.iloc[:,i].values)**2, window_len))
    est_lev_ac.append(my_acf(syn_asset_returns.iloc[:,i].values, window_len, lev=True))

    real_linear_ac.append(my_acf(returns_df.iloc[:,i].values, window_len))
    real_abs_ac.append(my_acf(abs(returns_df.iloc[:,i].values)**2, window_len))
    real_lev_ac.append(my_acf(returns_df.iloc[:,i].values, window_len, lev=True))

    # scaled_linear_ac.append(my_acf(syn_scaled_asset_returns.iloc[:,i].values, window_len))
    # scaled_abs_ac.append(my_acf(abs(syn_scaled_asset_returns.iloc[:,i].values)**2, window_len))
    # scaled_lev_ac.append(my_acf(syn_scaled_asset_returns.iloc[:,i].values, window_len, lev=True))

est_linear_ac = np.array(est_linear_ac)[:,1:]
est_abs_ac = np.array(est_abs_ac)[:,1:]
est_lev_ac = np.array(est_lev_ac)[:,1:]

real_linear_ac = np.array(real_linear_ac)[:,1:]
real_abs_ac = np.array(real_abs_ac)[:,1:]
real_lev_ac = np.array(real_lev_ac)[:,1:]

fig, ax = plt.subplots(1,3, figsize=(12,3))
ci = 1.96
line_w=0.8

mean = np.mean(est_linear_ac, axis=0)
se_mean = np.std(est_linear_ac, axis=0)

ax[0].plot(range(1,window_len+1), mean, color=synthetic_original_color, label='Synthetic', lw=line_w)
ax[0].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=synthetic_original_color, alpha=.05)


mean = np.mean(real_linear_ac, axis=0)
se_mean = np.std(real_linear_ac, axis=0)

ax[0].plot(range(1,window_len+1), mean, color=original_color, label='Original', ls='dotted', lw=line_w)
ax[0].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=original_color, alpha=.05)
ax[0].legend()
ax[0].set_ylim(-.3,.3)
ax[0].set_title('Autocorrelations of returns')
ax[0].set_xlabel('$\\tau$ (days)')

mean = np.mean(est_abs_ac, axis=0)
se_mean = np.std(est_abs_ac, axis=0)

ax[1].plot(range(1,window_len+1), mean, color=synthetic_original_color, label='Synthetic', lw=line_w)
ax[1].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=synthetic_original_color, alpha=.05)

mean = np.mean(real_abs_ac, axis=0)
se_mean = np.std(real_abs_ac, axis=0)

ax[1].plot(range(1,window_len+1), mean, color=original_color, label='Original', ls='dotted', lw=line_w)
ax[1].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=original_color, alpha=.05)
ax[1].set_title('Autocorrelations of squared returns')
ax[1].set_xlabel('$\\tau$ (days)')

mean = np.mean(est_lev_ac, axis=0)
se_mean = np.std(est_lev_ac, axis=0)

ax[2].plot(range(1,window_len+1), mean, color=synthetic_original_color, label='Synthetic', lw=line_w)
ax[2].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=synthetic_original_color, alpha=.05)

mean = np.mean(real_lev_ac, axis=0)
se_mean = np.std(real_lev_ac, axis=0)

ax[2].plot(range(1,window_len+1), mean, color=original_color, label='Original', ls='dotted', lw=line_w)
ax[2].fill_between(range(1,window_len+1), (mean-se_mean*ci), (mean+se_mean*ci), color=original_color, alpha=.05)
ax[2].set_title('Cross-correlation between returns and squared returns')
ax[2].set_xlabel('$\\tau$ (days)')

plt.tight_layout()

plt.savefig(eval_dir+'/evaluation_ac.png')

plt.close()

results_dict = {}

for k in tqdm(range(n_market_gen)):
    
    results_dict[k] = {}

    syn_asset_returns, syn_factor_based_returns, syn_residual_returns, syn_pc_returns  = market_generator(  generators = generators,
                                                                                                            clusters = clusters,
                                                                                                            residual_params = residual_params,
                                                                                                            len_syn = len_syn_data,
                                                                                                            eigenvalues = eigenvalues,
                                                                                                            eigenvectors = eigenvectors,
                                                                                                            returns_df = returns_df,
                                                                                                            random_state = None)
    
    syn_asset_returns = syn_asset_returns[:n_sample]
    syn_factor_based_returns = syn_factor_based_returns[:n_sample]
    syn_residual_returns = syn_residual_returns[:n_sample]
    
    results_dict[k]['syn_asset_returns'] = {}
    results_dict[k]['syn_factor_based_returns'] = {}
    results_dict[k]['syn_residual_returns'] = {}

    for i in zip([returns_df, factor_based_returns_df, residuals_df], ['syn_asset_returns', 'syn_factor_based_returns', 'syn_residual_returns']):
        
        key = i[1]
        orig = i[0].copy()

        results_dict[k][key]['wasserstein_distances'] = []
        results_dict[k][key]['ks_stats'] = []
        results_dict[k][key]['ks_pvalues'] = []
        results_dict[k][key]['linear_autocorr_scores'] = []
        results_dict[k][key]['vol_clustering_scores'] = []
        results_dict[k][key]['lev_effect_scores'] = []
        
        X = np.arange(window_len+1).reshape(window_len+1, 1)
        for i,col in enumerate(syn_asset_returns.columns):

            my_syn_returns = eval(key)[col].values
            my_real_returns = orig[col].values
            
            results_dict[k][key]['wasserstein_distances'].append(wasserstein_distance(my_syn_returns, my_real_returns)) 
            ks_test = kstest(my_syn_returns, my_real_returns)
            results_dict[k][key]['ks_stats'].append(ks_test.statistic)
            results_dict[k][key]['ks_pvalues'].append(ks_test.pvalue)
            results_dict[k][key]['linear_autocorr_scores'].append(sum(my_acf((my_syn_returns), lag_len=window_len)[1:]**2))
            results_dict[k][key]['vol_clustering_scores'].append(sum(my_acf((my_syn_returns**2), lag_len=window_len)[1:]**2))
            lev_corrs = my_acf((my_syn_returns), lag_len=window_len, lev=True)
            results_dict[k][key]['lev_effect_scores'].append(sum(lev_corrs**2))
            # reg = LinearRegression().fit(X, lev_corrs)
            # lev_effect_scores.append(np.sign(reg.coef_.item())*sum(lev_corrs**2
            
with open(eval_dir+'/results_dict.pkl','wb') as f:
    pickle.dump(results_dict,f)

# with open(eval_dir+'/results_dict.pkl', 'rb') as f:
#     results_dict = pickle.load(f)

for i in zip([returns_df, factor_based_returns_df, residuals_df], ['syn_asset_returns', 'syn_factor_based_returns', 'syn_residual_returns']):

    orig = i[0]
    key = i[1]
    
    returns_df = orig.copy()

    df_synthetic_w1 = pd.DataFrame(index=returns_df.columns)
    df_synthetic_ks_stats = pd.DataFrame(index=returns_df.columns)
    df_synthetic_ks_pvalues = pd.DataFrame(index=returns_df.columns)
    df_synthetic_linear_autocorr_scores = pd.DataFrame(index=returns_df.columns)
    df_synthetic_vol_clustering_scores = pd.DataFrame(index=returns_df.columns)
    df_synthetic_lev_effect_scores = pd.DataFrame(index=returns_df.columns)
    df_synthetic_skew = pd.DataFrame(index=returns_df.columns)
    df_synthetic_kurt = pd.DataFrame(index=returns_df.columns)
    df_synthetic_VaR = pd.DataFrame(index=returns_df.columns)
    df_synthetic_ES = pd.DataFrame(index=returns_df.columns)
    df_synthetic_VaR_high = pd.DataFrame(index=returns_df.columns)
    df_synthetic_ES_high = pd.DataFrame(index=returns_df.columns)

    for k in range(n_market_gen):

        df_synthetic_w1['Wasserstein distance_'+str(k)] = results_dict[k][key]['wasserstein_distances']
        df_synthetic_ks_stats['KS stats_'+str(k)] = results_dict[k][key]['ks_stats']
        df_synthetic_ks_pvalues['KS pvalues_'+str(k)] = results_dict[k][key]['ks_pvalues']
        df_synthetic_linear_autocorr_scores['Linear autocorrelation_'+str(k)] = results_dict[k][key]['linear_autocorr_scores']
        df_synthetic_vol_clustering_scores['Volatility clustering_'+str(k)] = results_dict[k][key]['vol_clustering_scores']
        df_synthetic_lev_effect_scores['Leverage effect_'+str(k)] = results_dict[k][key]['lev_effect_scores']
        df_synthetic_skew['Skewness_'+str(k)] = orig.skew()
        df_synthetic_kurt['Kurtosis_'+str(k)] = orig.kurt()
        df_synthetic_VaR['VaR_95_'+str(k)] = -orig.quantile(.05)
        df_synthetic_ES['ES_95_'+str(k)] = -orig[orig<orig.quantile(.05)].mean()
        df_synthetic_VaR_high['VaR_99_'+str(k)] = -orig.quantile(.01)
        df_synthetic_ES_high['ES_99_'+str(k)] = -orig[orig<orig.quantile(.01)].mean()


    ## some stats from historical returns

    linear_autocorr_scores = []
    vol_clustering_scores = []
    lev_effect_scores = []

    X = np.arange(window_len+1).reshape(window_len+1, 1)
    for i,col in enumerate(returns_df.columns):
        linear_autocorr_scores.append(sum(my_acf((returns_df[col].values), lag_len=window_len)[1:]**2))
        vol_clustering_scores.append(sum(my_acf((returns_df[col].values**2), lag_len=window_len)[1:]**2))
        lev_corrs = my_acf((returns_df[col].values), lag_len=window_len, lev=True)
        # reg = LinearRegression().fit(X, lev_corrs)
        # lev_effect_scores.append(np.sign(reg.coef_.item())*sum(lev_corrs**2))
        lev_effect_scores.append(sum(lev_corrs**2))

    df_scores = pd.DataFrame(index=returns_df.columns)
    df_scores['Linear autocorrelation'] = linear_autocorr_scores
    df_scores['Volatility clustering'] = vol_clustering_scores
    df_scores['Leverage effect'] = lev_effect_scores
    df_scores['Skewness'] = returns_df.skew()
    df_scores['Kurtosis'] = returns_df.kurt()
    df_scores['VaR_95'] = -returns_df.quantile(.05)
    df_scores['ES_95'] = -returns_df[returns_df<returns_df.quantile(.05)].mean()
    df_scores['VaR_99'] = -returns_df.quantile(.01)
    df_scores['ES_99'] = -returns_df[returns_df<returns_df.quantile(.01)].mean()

    wasserstein_distances_gaussian = []
    ks_stats_gaussian = []
    ks_pvalues_gaussian = []

    for i,col in enumerate(returns_df.columns):
        mu = np.mean(returns_df[col].values)
        sigma =  np.std(returns_df[col].values)
        my_normal_returns = np.random.normal(mu, sigma, size=n_sample)
        my_real_returns = returns_df[col].values
        ks_test = kstest(my_normal_returns, my_real_returns)
        wasserstein_distances_gaussian.append(wasserstein_distance(my_normal_returns, my_real_returns)) 
        ks_stats_gaussian.append(ks_test.statistic)
        ks_pvalues_gaussian.append(ks_test.pvalue)    

    df_syn = df_synthetic_w1.copy()
    fig, ax = plt.subplots(figsize=(20, 4))
    df_syn.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn.index, wasserstein_distances_gaussian, color='green', label='Gaussian', alpha=.5)
    ax.plot(df_syn.index, df_syn.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    ax.set_ylabel('Wasserstein distance')
    ax.set_title('Wasserstein distance', fontsize=12)

    ax.set_xlabel('Stock')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_wasserstein_distance.png')
    plt.close()

    df_syn = df_synthetic_ks_stats.copy()
    fig, ax = plt.subplots(figsize=(20, 4))
    df_syn.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn.index, ks_stats_gaussian, color='green', label='Gaussian', alpha=.5)
    ax.plot(df_syn.index, df_syn.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    ax.set_ylabel('KS statistics')
    ax.set_xlabel('Stock')
    ax.set_title('KS statistics', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_ks_stats.png')
    plt.close()

    df_syn = df_synthetic_ks_pvalues.copy()
    fig, ax = plt.subplots(figsize=(20, 4))
    df_syn.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    # ax.plot(df_syn.index, ks_pvalues_gaussian, color='green', label='Gaussian', alpha=.5)
    # ax.plot(df_syn.index, df_syn.mean(axis=1), color=synthetic_original_color, label='Synthetic')
    ax.hlines(.05, df_syn.index[0], df_syn.index[-1], color='red', linestyles='dashed', lw=1)
    ax.set_ylabel('KS p-values')
    ax.set_xlabel('Stock')
    ax.set_title('KS p-values', fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_ks_pvalues.png')
    plt.close()

    df_syn = df_synthetic_vol_clustering_scores.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['Volatility clustering'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')
    ax.set_ylabel('Volatility clustering score')
    ax.set_xlabel('Stock')
    ax.set_title('Volatility clustering score', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_vol_clustering_score.png')
    plt.close()

    df_syn = df_synthetic_lev_effect_scores.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['Leverage effect'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')
    ax.set_ylabel('Leverage effect score')
    ax.set_xlabel('Stock')
    ax.set_title('Leverage effect score', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_lev_effect_score.png')
    plt.close()

    df_syn = df_synthetic_skew.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['Skewness'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')
    ax.set_ylabel('Skewness')
    ax.set_xlabel('Stock')
    ax.set_title('Skewness', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    ax.set_ylim(-2, 7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_skew.png')
    plt.close()


    df_syn = df_synthetic_kurt.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['Kurtosis'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')
    ax.set_ylabel('Kurtosis')
    ax.set_xlabel('Stock')
    ax.set_title('Kurtosis', fontsize=12)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    # ax.set_ylim(-2, 7)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_kurt.png')
    plt.close()

    df_syn = df_synthetic_VaR.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['VaR_95'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')

    ax.legend()

    df_syn = df_synthetic_ES.copy()

    df_real_sorted = df_scores['ES_95'].loc[df_real_sorted.index]
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic', ls='--')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original', ls='--')

    ax.set_ylabel('$(VaR_{95\%},ES_{95\%})$')
    ax.set_xlabel('Stock')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 

    ax.set_title('$(VaR_{95\%},ES_{95\%})$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_VaR_ES_95.png')
    plt.close()

    # VaR 99
    df_syn = df_synthetic_VaR_high.copy()
    fig, ax = plt.subplots(figsize=(20, 4))

    df_real_sorted = df_scores['VaR_99'].sort_values()
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original')

    ax.legend()

    df_syn = df_synthetic_ES_high.copy()

    df_real_sorted = df_scores['ES_99'].loc[df_real_sorted.index]
    df_syn_sorted = df_syn.loc[df_real_sorted.index]

    df_syn_sorted.stack().reset_index().plot.scatter(x='level_0', y=0, s=.06, color=synthetic_original_color, ax=ax)
    ax.plot(df_syn_sorted.index, df_syn_sorted.mean(axis=1), color=synthetic_original_color_line, label='Synthetic', ls='--')
    df_real_sorted.plot(ax=ax, color=original_color, label='Original', ls='--')

    ax.set_ylabel('$(VaR_{99\%},ES_{99\%})$')
    ax.set_xlabel('Stock')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10)) 
    # ax.set_ylim(-2, 7)

    ax.set_title('$(VaR_{99\%},ES_{99\%})$', fontsize=12)

    plt.tight_layout()
    plt.savefig(eval_dir+'/'+key+'_VaR_ES_99.png')
    plt.close()

