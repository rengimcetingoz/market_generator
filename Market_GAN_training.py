import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import json
sys.path.append('../utils')

from utils.generate_charts import *
from utils.marcenko_pastur import *

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from utils.gan import GAN
from utils.tcn import make_TCN, receptive_field_size
from utils.preprocessing import *
from utils.quant_strats import *
from utils.utils import *

from scipy.stats import t

import datetime 

# create dedicated folder for storing purposes
ct = datetime.datetime.now()
cur_date = str(ct.year) + '_' + str(ct.month) + '_' + str(ct.day) + '__' + str(ct.hour) + '_' + str(ct.minute)
my_file_name = 'GAN_training_'+cur_date
my_loc = './GAN_training/'+my_file_name

os.mkdir(my_loc)


### training data
returns_df = pd.read_csv('./data/training_data.csv', index_col=0)
returns_df.index = pd.to_datetime(returns_df.index)
n_sample, d = returns_df.shape

### params
np.random.seed(0)
tf.random.set_seed(0)

n_chosen_pcs = 16
n_cluster = 3

gan_params = {'log_returns':False,
              'window':63,
              'block_size':2,
              'latent_dim':3,
              'n_dilations': 5,
              'n_filters': 100,
              'noise_dist':'normal',
              'disc_lambda_layer':False,
              'batch_size':128,
              'n_batches':5000,
              'additional_d_steps':0,
              'lr_d':1e-5,
              'lr_g':5e-6,
              'store_stats':True,
              'store_freq':2000}

with open(my_loc+'/gan_params.txt', 'w') as file:
     file.write(json.dumps(gan_params)) 

window_len = gan_params['window']

### extracting factors
print('====== EXTRACTING FACTORS ======')
corr_matrix = returns_df.corr()
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
indices = eigenvalues.argsort()[::-1]
eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:,indices]

standardized_returns_df = (returns_df - returns_df.mean())/returns_df.std()
pc_returns_df = standardized_returns_df @ eigenvectors
scaled_pc_returns_df = pc_returns_df/eigenvalues**.5
chosen_pc_returns_df = pc_returns_df.iloc[:,:n_chosen_pcs]
chosen_scaled_pc_returns_df = scaled_pc_returns_df.iloc[:,:n_chosen_pcs]
factor_based_returns_df = chosen_pc_returns_df @ eigenvectors[:,:n_chosen_pcs].T
factor_based_returns_df.columns = returns_df.columns
residuals_df = standardized_returns_df - factor_based_returns_df

# save extracted data
chosen_pc_returns_df.to_csv(my_loc+'/chosen_pc_returns_df.csv')
residuals_df.to_csv(my_loc+'/residuals_df.csv')
returns_df.to_csv(my_loc+'/returns_df.csv')
factor_based_returns_df.to_csv(my_loc+'/factor_based_returns_df.csv')
standardized_returns_df.to_csv(my_loc+'/standardized_returns_df.csv')
with open(my_loc+'/eigenvalues.npy', 'wb') as f:
    np.save(f, eigenvalues)
with open(my_loc+'/eigenvectors.npy', 'wb') as f:
    np.save(f, eigenvectors)

print('Number of chosen PCs: ', n_chosen_pcs)
print('Variance explained by chosen PCs: '+ "{:.2%}".format(sum(eigenvalues[:n_chosen_pcs])/sum(eigenvalues)))

### clustering PCs
print('====== PC CLUSTERING ======')
print('Number of clusters: ', n_cluster)

chosen_eigenvalues = eigenvalues[:n_chosen_pcs]
scaled_chosen_pc_returns_df = chosen_pc_returns_df/chosen_eigenvalues**.5

# compute scores
x = np.arange(window_len+1).reshape(window_len+1, 1)
score_skew = scaled_chosen_pc_returns_df.skew()
score_kurt = scaled_chosen_pc_returns_df.kurt()
score_left_tail_index = (scaled_chosen_pc_returns_df.quantile(1-.99) - scaled_chosen_pc_returns_df.quantile(.5))/(scaled_chosen_pc_returns_df.quantile(1-.75) - scaled_chosen_pc_returns_df.quantile(.5))/3.45
score_right_tail_index = (scaled_chosen_pc_returns_df.quantile(.99) - scaled_chosen_pc_returns_df.quantile(.5))/(scaled_chosen_pc_returns_df.quantile(.75) - scaled_chosen_pc_returns_df.quantile(.5))/3.45
linear_autocorr_scores = []
vol_clustering_scores = []
lev_effect_scores = []
for i,col in enumerate(scaled_chosen_pc_returns_df.columns):
    linear_autocorr_scores.append(sum(my_acf((scaled_chosen_pc_returns_df[col].values), lag_len=window_len)[1:]**2))
    vol_clustering_scores.append(sum(my_acf((scaled_chosen_pc_returns_df[col].values**2), lag_len=window_len)[1:]**2))
    lev_corrs = my_acf((scaled_chosen_pc_returns_df[col].values), lag_len=window_len, lev=True)
    reg = LinearRegression().fit(x, lev_corrs)
    lev_effect_scores.append(np.sign(reg.coef_.item())*sum(lev_corrs**2))

df = pd.DataFrame()
df['Skew'] = score_skew
df['Kurtosis'] = score_kurt
# df['Left-tail index'] = score_left_tail_index
# df['Right-tail index'] = score_right_tail_index
# df['Linear autocorr. score'] = linear_autocorr_scores
df['Vol. clustering score'] = vol_clustering_scores
df['Lev. effect score'] = lev_effect_scores
df['Variance'] = chosen_eigenvalues

df_standardized = (df - df.mean())/df.std()

# clustering algo
clustering = AgglomerativeClustering(n_cluster).fit(df_standardized.values)
pc_labels = clustering.labels_

clustered_scaled_chosen_pc_returns_df = scaled_chosen_pc_returns_df.copy()
clustered_scaled_chosen_pc_returns_df.columns = pd.MultiIndex.from_tuples(zip(clustered_scaled_chosen_pc_returns_df.columns, pc_labels))
for i in range(n_cluster):
    my_class_df = clustered_scaled_chosen_pc_returns_df.reindex([i], axis=1, level=1)
    my_class_df.to_csv(my_loc+'/Cluster_'+str(i+1)+'.csv')

print('PC cluster labels: ', pc_labels)

# train n_cluster GANs 
print('====== GAN TRAINING ======')
for i in range(n_cluster):
    print('**** CLUSTER ' +str(i+1)+' ****')
    my_loc_cluster = my_loc + '/cluster_'+str(i+1)
    my_loc_cluster_monitoring = my_loc_cluster+'/training_monitoring'
    
    os.mkdir(my_loc_cluster)
    os.mkdir(my_loc_cluster_monitoring)
    os.mkdir(my_loc_cluster_monitoring+'/stylized_facts_charts')
    os.mkdir(my_loc_cluster_monitoring+'/cum_return_charts')

    # get ith cluster returns
    data = clustered_scaled_chosen_pc_returns_df.reindex([i], axis=1, level=1)
    returns = data.values

    # pre-process data (scaling, gaussianize if necessary)
    scaled_returns, scalers = pretraining_transformation_unit_for_lambda(returns)
    scaler1, inv_lambert_w, scaler2 = scalers
    # scalers_ = (scaler1, inv_lambert_w, scaler2)
    scalers_ = [scaler1]
    def pre_process(x, scalers=scalers_):
        n_rows, n_cols = x.shape
        x = scalers[0].transform(x.reshape((n_rows*n_cols,1))).reshape((n_rows, n_cols))
        # x = scalers[1].transform(x.reshape((n_rows*n_cols,1))).reshape((n_rows, n_cols))
        # x = scalers[2].transform(x)
        return x
    preprocessed_returns = pre_process(returns, scalers=scalers_)

    # reshape data to make it trainable 
    returns_rolled_concat = []
    for i in range(preprocessed_returns.shape[1]):
        returns_rolled = rolling_window(preprocessed_returns[:,[i]], window_len)
        returns_rolled_concat.append(returns_rolled)
    returns_rolled_concat = np.concatenate(returns_rolled_concat, axis=1)
    training_data = np.expand_dims(np.moveaxis(returns_rolled_concat, 0,1), 1).astype('float32')
    # print('Number of samples: ', training_data.shape[0])
    # print('Length of each sample: ', training_data.shape[2])

    # parameters
    n_dilations = gan_params['n_dilations']
    dilations = 2**(np.arange(n_dilations))
    block_size = gan_params['block_size']
    rfs = receptive_field_size(dilations, block_size)
    n_filters = gan_params['n_filters']
    latent_dim = gan_params['latent_dim']


    # discriminator
    discriminator = make_TCN(dilations=dilations,
                            fixed_filters=n_filters,
                            moving_filters=0,
                            use_batchNorm=False,
                            one_series_output=False,
                            sigmoid=False,
                            input_dim=[1, rfs, 1],
                            block_size=block_size,
                            lambda_layer=(gan_params['disc_lambda_layer'],(inv_lambert_w,scaler2)))

    # generator
    generator = make_TCN(dilations=dilations,
                        fixed_filters=n_filters,
                        moving_filters=0,
                        use_batchNorm=True,
                        one_series_output=False,
                        sigmoid=False,
                        input_dim=[1, None, latent_dim],
                        block_size=block_size)


    def post_process(x, scalers=scalers_):
        n_rows, n_cols = x.shape
        x = scalers[-1].inverse_transform(x)
        # x = scalers[-2].inverse_transform(x.reshape((n_rows*n_cols,1))).reshape((n_rows, n_cols))
        # x = scalers[-3].inverse_transform(x)
        return x

    gan = GAN(discriminator, generator, 2*rfs - 1,
              noise_dist=gan_params['noise_dist'],
              real_returns=np.expand_dims(returns[:,0], 1),
              preprocessed_returns=np.expand_dims(preprocessed_returns[:,0], 1),
              lr_d=gan_params['lr_d'],
              lr_g=gan_params['lr_g'],
              store_stats=gan_params['store_stats'],
              store_freq=gan_params['store_freq'],
              post_process_func=post_process,
              file_loc=my_loc_cluster_monitoring)
    
    batch_size = gan_params['batch_size']
    n_batches = gan_params['n_batches']
    additional_d_steps = gan_params['additional_d_steps']

    gan.train(training_data, batch_size, n_batches, additional_d_steps)

    generator.save(my_loc_cluster+'/trained_generator')
    discriminator.save(my_loc_cluster+'/trained_discriminator')


### residual fitting
print('====== RESIDUAL FIT ======')
residual_params = {}
for i in tqdm(residuals_df.columns):
    t_params = t.fit(residuals_df[i].values)
    # if d.o.f is less than 2, replace it by 2 to have variance<inf
    if t_params[0]<2:
        t_params = (2, t_params[1], t_params[2])
    residual_params[i] = t_params

with open(my_loc+'/residuals_params.pkl','wb') as f:
    pickle.dump(residual_params,f)



