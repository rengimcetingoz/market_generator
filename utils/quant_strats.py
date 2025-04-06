import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.random import normal
from scipy.stats import t
import sys
sys.path.append('./utils')
from preprocessing import *

def mean_reversion_strategy(returns_df, lookback_window, upper_pct, lower_pct, rebal_freq, implementation_lag, sector_neutral=False):

    rebal_dates = pd.date_range(start=returns_df.index[lookback_window-1], end=returns_df.index[-1], freq=rebal_freq)

    # compute n-day returns at each day
    # log_returns_df = np.log(returns_df+1)
    # cum_returns_df = log_returns_df.rolling(lookback_window).sum()#/log_returns_df.ewm(span=260).std()

    log_returns_df = (returns_df+1).cumprod()
    cum_returns_df = log_returns_df.pct_change(lookback_window)

    if sector_neutral:
         # rank n-day returns at each rebal date within each sector
        pct_df = cum_returns_df.reindex(rebal_dates).dropna().groupby(axis=1, level=1).rank(pct=True, axis=1)
    else:
        # rank n-day returns at each rebal date
        pct_df = cum_returns_df.reindex(rebal_dates).dropna().rank(pct=True, axis=1) 

    # assign the associated weights, ~1/(d/2) and ~-1/(d/2), to low-performing quartile and high-performing quartile respectively
    n_lower = pct_df[pct_df<lower_pct].count(axis=1)[0]
    n_upper = pct_df[pct_df>upper_pct].count(axis=1)[0]
    pct_df[pct_df<lower_pct] = 1/n_lower
    pct_df[(pct_df>=lower_pct) & (pct_df<=upper_pct)] = 0
    pct_df[pct_df>upper_pct] = -1/n_upper

    #forward fill the weights computed for each rebal
    weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
    weights_df.loc[pct_df.index] = pct_df
    weights_df.fillna(method='ffill', inplace=True)

    # compute the strategy returns
    strategy_returns = (returns_df * weights_df.shift(implementation_lag+1)).dropna().sum(axis=1)
    
    return strategy_returns, weights_df


def block_bootstrap(returns_df, block_length=63, overlapping=True):
    n_sample = len(returns_df)
    n_blocks = n_sample//block_length
    if overlapping:
        my_indices = [np.arange(i,i+block_length, dtype=int) for i in np.random.choice(range(n_sample-block_length), size=n_blocks)]
        n_remaining = n_sample - len(np.concatenate(my_indices)) 
        i_last = np.random.choice(range(n_sample-n_remaining))
        my_indices+= [np.arange(i_last,i_last+n_remaining, dtype=int)]
        my_indices = np.concatenate(my_indices)
    else:
        my_indices = [np.arange(i*block_length,(i+1)*block_length, dtype=int) for i in np.random.choice(range(n_blocks), size=n_blocks)]
        n_remaining = n_sample - len(np.concatenate(my_indices)) 
        i_last = np.random.choice(range(n_sample-n_remaining))
        my_indices+= [np.arange(i_last,i_last+n_remaining, dtype=int)]
        my_indices = np.concatenate(my_indices)

    bootstrap_sample = returns_df.iloc[my_indices]

    return bootstrap_sample

def bootstrap_results(inputs):

    seed = inputs[0]
    returns_df = inputs[1]
    upper_pct = inputs[2]
    lower_pct = inputs[3]
    rebal_freq = inputs[4]
    implementation_lag = inputs[5]
    sector_neutral = inputs[6]
    lookback_windows = inputs[7]
    n_bootstrap_sample = inputs[8]
    block_length = inputs[9]
    overlapping = inputs[10]

    np.random.seed(seed)

    results_dict = {}
    results_dict['Name'] = 'bootstrap_seed_' + str(seed)
    results_dict['Sharpe'] = {}
    
    for h in lookback_windows: results_dict['Sharpe'][h] = []

    for b in tqdm(range(n_bootstrap_sample)):
        bootstrap_sample = block_bootstrap(returns_df, block_length=block_length, overlapping=overlapping)
        bootstrap_sample.index = returns_df.index
        for h in lookback_windows:
            strategy_returns, _ = mean_reversion_strategy(returns_df=bootstrap_sample, lookback_window=h, upper_pct=upper_pct, lower_pct=lower_pct, rebal_freq=rebal_freq, implementation_lag=max(1,int(implementation_lag*h)), sector_neutral=sector_neutral)
            mu = strategy_returns.mean()*260
            sigma = strategy_returns.std()*260**.5
            results_dict['Sharpe'][h].append(mu/sigma)

    return results_dict

def market_generator(generators, clusters, residual_params, len_syn, eigenvalues, eigenvectors, returns_df, random_state):

    if not random_state==None:
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    # create a dataframe of scaled PC returns
    pc_returns_df = pd.concat(clusters, axis=1)

    # get total number of PCs
    n_chosen_pc = pc_returns_df.shape[1]

    syn_scaled_pc_returns = np.array([])

    # generate synthetic scaled PC returns respecting the number of PCs (n_s) in the associated cluster
    for gen,cluster in zip(generators, clusters):
        n_s = cluster.shape[1]
        noise = normal([n_s, 1, len_syn + 63 - 1, 3])
        if n_s==1:
            syn_returns_pc = np.expand_dims(gen(noise).numpy().squeeze(), axis=1)
        else:
            syn_returns_pc = gen(noise).numpy().squeeze().T

        # center and scale the synthetic data to remove the sampling error, optional
        syn_returns_pc = (syn_returns_pc - np.mean(syn_returns_pc, axis=0))/np.std(syn_returns_pc, axis=0)

        # concatenate generated clusters/PC returns
        if len(syn_scaled_pc_returns)==0:
            syn_scaled_pc_returns = syn_returns_pc
        else:
            syn_scaled_pc_returns = np.concatenate((syn_scaled_pc_returns, syn_returns_pc), axis=1)

    # re-ordering
    syn_scaled_pc_returns = pd.DataFrame(syn_scaled_pc_returns, columns=pc_returns_df.columns)
    syn_scaled_pc_returns = syn_scaled_pc_returns.sort_index(axis=1, level=0)

    # re-scale scaled synthetic PC returns using stored eigenvalues
    syn_pc_returns = syn_scaled_pc_returns*eigenvalues[:n_chosen_pc]**.5

    # map/inverse transform to original space using stored eigenvectors
    syn_factor_based_returns = syn_pc_returns.values@eigenvectors[:,:n_chosen_pc].T
    syn_factor_based_returns = pd.DataFrame(syn_factor_based_returns, columns=returns_df.columns)

    # create empty DataFrame to store synthetic asset returns
    scaled_asset_returns = pd.DataFrame(index=syn_factor_based_returns.index, columns=returns_df.columns)

    # add synthetic residuals
    for i in syn_factor_based_returns.columns:
        syn_res = t.rvs(*residual_params[i], size=len_syn)
        scaled_asset_returns[i] = syn_factor_based_returns[i] + syn_res

    syn_residual_returns = scaled_asset_returns - syn_factor_based_returns
    
    # correct the mean and the volatility
    scaled_asset_returns = (scaled_asset_returns - scaled_asset_returns.mean())/scaled_asset_returns.std()
    syn_asset_returns = scaled_asset_returns*returns_df.std()+returns_df.mean()

    return syn_asset_returns, syn_factor_based_returns, syn_residual_returns, syn_pc_returns


def gan_results(inputs):

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.random import normal

    ## generators
    generator_cluster_1 = load_model('./data/generators/2024_3_9__13_21_training_cluster_1/trained_generator')
    generator_cluster_2 = load_model('./data/generators/2024_3_11__16_30_training_cluster_2/trained_generator')
    generator_cluster_3 = load_model('./data/generators/2024_3_9__14_50_training_cluster_3/trained_generator')

    generators = [generator_cluster_1, generator_cluster_2, generator_cluster_3]

    seed = inputs[0]
    returns_df = inputs[1]
    upper_pct = inputs[2]
    lower_pct = inputs[3]
    rebal_freq = inputs[4]
    implementation_lag = inputs[5]
    sector_neutral = inputs[6]
    lookback_windows = inputs[7]
    n_bootstrap_sample = inputs[8]
    # generators = inputs[9]
    clusters = inputs[9]
    residual_params = inputs[10]
    eigenvalues = inputs[11]
    eigenvectors = inputs[12]

    np.random.seed(seed)
    tf.random.set_seed(seed)

    results_dict = {}
    results_dict['Name'] = 'GAN_seed_' + str(seed)
    results_dict['Sharpe'] = {}

    for h in lookback_windows: results_dict['Sharpe'][h] = []

    n_sample, d = returns_df.shape
    syn_asset_returns, _, _, _  = market_generator(generators = generators,
                                                   clusters = clusters,
                                                   residual_params = residual_params,
                                                   len_syn = n_sample*n_bootstrap_sample,
                                                   eigenvalues = eigenvalues,
                                                   eigenvectors = eigenvectors,
                                                   returns_df = returns_df,
                                                   random_state = None)    
    
    syn_asset_returns = np.array([syn_asset_returns[i*n_sample:i*n_sample+n_sample] for i in range(n_bootstrap_sample)])
    for b in tqdm(range(n_bootstrap_sample)):        
        gan_sample = pd.DataFrame(syn_asset_returns[b], columns=returns_df.columns, index=returns_df.index)
        gan_sample.index = returns_df.index
        for h in lookback_windows:
                strategy_returns, _ = mean_reversion_strategy(returns_df=gan_sample, lookback_window=h, upper_pct=upper_pct, lower_pct=lower_pct, rebal_freq=rebal_freq, implementation_lag=max(1,int(implementation_lag*h)), sector_neutral=sector_neutral)
                mu = strategy_returns.mean()*260
                sigma = strategy_returns.std()*260**.5
                results_dict['Sharpe'][h].append(mu/sigma)

    return results_dict        


def pretraining_transformation_unit_for_lambda(x):

    dims = x.shape
    x = x.reshape((dims[0]*dims[1], 1))

    # standardize log returns
    scaler1 = StandardScaler()
    x = scaler1.fit_transform(x)

    # apply inverse lambert w to go from heavy tailed to approx. normal
    inv_lambert_w = Gaussianize()
    x = inv_lambert_w.fit_transform(x)

    # standardize
    scaler2 = StandardScaler()
    x = scaler2.fit_transform(x)

    x = x.reshape(dims)

    return x, (scaler1, inv_lambert_w, scaler2)