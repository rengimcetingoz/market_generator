import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import skew, kurtosis, wasserstein_distance
from scipy.cluster.hierarchy import dendrogram

def fig2img(fig):
	"""Convert a Matplotlib figure to a PIL Image and return it"""
	from PIL import Image
	import io
	buf = io.BytesIO()
	fig.savefig(buf)
	buf.seek(0)
	img = Image.open(buf)
	return img

def Images_pad_horizontal(Images: list, v_center: bool):
	widths, heights = zip(*(i.size for i in Images))
	total_width = sum(widths)
	max_height = max(heights)
	new_im = Image.new('RGB', (total_width, max_height), "white")
	x_offset = 0
	y_offset = 0
	for im in Images:
		if v_center: y_offset = int((max_height - im.size[1])/2)
		new_im.paste(im, (x_offset, y_offset))
		x_offset += im.size[0]
	return new_im


def Images_pad_vertical(Images: list, h_center: bool):
	from PIL import Image
	widths, heights = zip(*(i.size for i in Images))
	max_width = max(widths)
	total_height = sum(heights)
	new_im = Image.new('RGB', (max_width, total_height), "white")
	x_offset = 0
	y_offset = 0
	for im in Images:
		if h_center: x_offset = int((max_width - im.size[0])/2)
		new_im.paste(im, (x_offset, y_offset))
		y_offset += im.size[1]
	return new_im




# def my_fig2img(fig, method: str):

#     if method=='direct':
#         img = fig2img_direct(fig)

#     elif method=='indirect':
#         from PIL import Image
#         import os
#         fig.savefig('image', bbox_inches='tight')

#         with Image.open('image.png') as temp_image:
#             img = temp_image.copy()
#             # temp_image.close() # python 3.9
#             os.remove('image.png')
	
#     return img

def my_acf(my_arr, lag_len, lev=False):
	x = my_arr
	x = x - x.mean()
	acorr = np.empty(lag_len + 1)
	if lev:
		x_squared = my_arr**2
		x_squared = x_squared - x_squared.mean()
		acorr[0] = x_squared.dot(x)/np.sqrt(x_squared.dot(x_squared)*x.dot(x))
		for i in range(lag_len):
			acorr[i + 1] = x_squared[i + 1 :].dot(x[: -(i + 1)])/np.sqrt(x_squared[i + 1 :].dot(x_squared[i + 1 :])*x[: -(i + 1)].dot(x[: -(i + 1)]))
	else:
		acorr[0] = 1
		for i in range(lag_len):
			acorr[i + 1] = x[i + 1 :].dot(x[: -(i + 1)])/np.sqrt(x[i + 1 :].dot(x[i + 1 :])*x[: -(i + 1)].dot(x[: -(i + 1)]))
	return acorr
  
def stats(real_returns, syn_returns, alpha=.95, acf_window=200):
	wassersteins = []
	for arr in syn_returns:
		wassersteins.append(wasserstein_distance(arr,real_returns.T[0]))
	means = syn_returns.mean(axis=1)
	vols = syn_returns.std(axis=1)
	skews = skew(syn_returns, axis=1)
	kurts = kurtosis(syn_returns, axis=1)
	vars = (-np.quantile(syn_returns, 1-alpha, axis=1))
	es = [-np.mean(i[i<-var]) for i, var in zip(syn_returns, vars)]
	acfs_linear = [np.linalg.norm(my_acf(i, acf_window, lev=False)-my_acf(real_returns.T[0], acf_window, lev=False))/acf_window for i in syn_returns]
	acfs_vol_cluster = [np.linalg.norm(my_acf(i**2, acf_window, lev=False)-my_acf(real_returns.T[0]**2, acf_window, lev=False))/acf_window for i in syn_returns]
	acfs_lev_eff = [np.linalg.norm(my_acf(i, acf_window, lev=True)-my_acf(real_returns.T[0], acf_window, lev=True))/acf_window for i in syn_returns]
	return (np.mean(wassersteins),np.std(wassersteins)), (means.mean(), means.std()), (vols.mean(), vols.std()), (skews.mean(), skews.std()), (kurts.mean(), kurts.std()), (vars.mean(), vars.std()), (np.mean(es), np.std(es)), (np.mean(acfs_linear), np.std(acfs_linear)), (np.mean(acfs_vol_cluster), np.std(acfs_vol_cluster)), (np.mean(acfs_lev_eff), np.std(acfs_lev_eff))

def plot_stylized_facts(real_returns, syn_returns, window_acf=200, title='', save_bool=False, save_name='', color='red'):

	import matplotlib
	matplotlib.rc('xtick', labelsize=8)
	matplotlib.rc('ytick', labelsize=8)
	
	fig, ax = plt.subplots(1,4, figsize=(20,3))
	
	ax[0].hist(real_returns, color=color, alpha=.5, density=True, bins=100, label='Real')
	ax[0].hist(syn_returns[0], color='grey', alpha=.5, density=True, bins=100, label='Synthetic')
	# plt.hist(syn_returns[30], color='red', alpha=.5, density=True, bins=100, label='Synthetic');
	ax[0].set_xlim(real_returns.mean() - 10*real_returns.std(),real_returns.mean() + 10*real_returns.std())
	ax[0].legend(fontsize=8)
	ax[0].set_title('Distribution of returns', fontsize=8)
	
	# ax[1].plot(np.cumsum(syn_returns, axis=1).T, color='grey', lw=.2, alpha=.5)
	# ax[1].plot(np.cumsum(real_returns.T), color='red', label='Real', lw=.5, alpha=.5)
	# # plt.hist(syn_returns[30], color='red', alpha=.5, density=True, bins=100, label='Synthetic');
	# ax[1].set_ylim(-250,250)
	# ax[1].legend(fontsize=8)
	# ax[1].set_title('Cumulative log returns', fontsize=8)
	conf_int = 2
	acfs = []
	for s in syn_returns:
		acfs.append(my_acf((s), lag_len=window_acf))

	mean_acf = np.array(acfs).mean(axis=0)
	std_acf = np.array(acfs).std(axis=0)
	up_acf = mean_acf + conf_int*std_acf
	down_acf = mean_acf - conf_int*std_acf

	ax[1].plot(my_acf((real_returns.T[0]), lag_len=window_acf), color=color, alpha=.5, label='Real')
	ax[1].plot(mean_acf, color='grey', alpha=.5, label='Synthetic')
	ax[1].plot(up_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[1].plot(down_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[1].legend(fontsize=8)
	ax[1].set_title('Autocorrelation - returns', fontsize=8);
	ax[1].set_ylim(-1,1)

	acfs = []   
	for s in syn_returns:
		acfs.append(my_acf((s)**2, lag_len=window_acf))

	mean_acf = np.array(acfs).mean(axis=0)
	std_acf = np.array(acfs).std(axis=0)
	up_acf = mean_acf + conf_int*std_acf
	down_acf = mean_acf - conf_int*std_acf

	ax[2].plot(my_acf((real_returns.T[0])**2, lag_len=window_acf), color=color, alpha=.5, label='Real')
	ax[2].plot(mean_acf, color='grey', alpha=.5, label='Synthetic')
	ax[2].plot(up_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[2].plot(down_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[2].legend(fontsize=8)
	ax[2].set_title('Autocorrelation - squared returns', fontsize=8);
	ax[2].set_ylim(-.2,1)
# wassersteins = []
# for arr in syn_returns:
#     wassersteins.append(wasserstein_distance(arr,real_returns.T[0]))

# ax[2].hist(wassersteins,bins=20, density=False, color='grey', alpha=.5);
# ymin, ymax = ax[2].get_ylim()
# ax[2].vlines(np.mean(wassersteins), ymin, ymax, color='red', linestyles='dashed', label='mean')
# ax[2].set_title('Distribution of $W_1(real,syn)$', fontsize=8)
# ax[2].legend(fontsize=8)
# ax[2].set_xlim(0,.5);

	acfs = []
	for s in syn_returns:
		acfs.append(my_acf((s), lag_len=window_acf, lev=True))

	mean_acf = np.array(acfs).mean(axis=0)
	std_acf = np.array(acfs).std(axis=0)
	up_acf = mean_acf + conf_int*std_acf
	down_acf = mean_acf - conf_int*std_acf

	ax[3].plot(my_acf((real_returns.T[0]), lag_len=window_acf, lev=True), color=color, alpha=.5, label='Real')
	ax[3].plot(mean_acf, color='grey', alpha=.5, label='Synthetic')
	ax[3].plot(up_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[3].plot(down_acf, linestyle='dashed', color='grey', alpha=.2)
	ax[3].legend(fontsize=8)
	ax[3].set_title('Autocorrelation - (returns,squared returns)', fontsize=8);
	ax[3].set_ylim(-1,1);
	
	fig.suptitle(title, fontsize=10)
# acfs = []
# for s in syn_returns:
#     acfs.append(acf(abs(s), nlags=100))

# mean_acf = np.array(acfs).mean(axis=0)
# std_acf = np.array(acfs).std(axis=0)
# up_acf = mean_acf + 2*std_acf
# down_acf = mean_acf - 2*std_acf

# ax[3].plot(acf(abs(real_returns.T[0]), nlags=100), color='red', alpha=.5, label='Real')
# ax[3].plot(mean_acf, color='grey', alpha=.5, label='Synthetic')
# ax[3].plot(up_acf, linestyle='dashed', color='grey', alpha=.2)
# ax[3].plot(down_acf, linestyle='dashed', color='grey', alpha=.2)
# ax[3].legend(fontsize=8)
# ax[3].set_title('Autocorrelation - absolute returns', fontsize=8);
	if save_bool:
		plt.savefig(save_name+'.png')
	plt.close()
	
	return fig
	
	
	
def save_stats_charts(wasses,
					  means_,
					  vols,
					  skews,
					  kurts,
					  value_at_risks,
					  expected_shortfalls,
					  acfs_linear,
					  acfs_vol_cluster,
					  acfs_leverage_eff,
					  real_returns, 
					  save=True,
					  name='Pre',
					  loc='',
					  up_xlim=1000):

	import matplotlib
	matplotlib.rc('xtick', labelsize=8)
	matplotlib.rc('ytick', labelsize=8)

	fig, ax = plt.subplots(4,3, figsize=(20,12))
	
	conf_int = 1.96
	
	means = np.array(wasses)[:,0]
	stds = np.array(wasses)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[0][0].plot(means, color='grey', alpha=.5)
	ax[0][0].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[0][0].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	xmin, xmax = 0,up_xlim
	ax[0][0].hlines(0, xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[0][0].set_title('Wasserstein distance', fontsize=8);
	# ax[0][0].set_ylim(0,.8);

	means = np.array(means_)[:,0]
	stds = np.array(means_)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[0][1].plot(means, color='grey', alpha=.5)
	ax[0][1].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[0][1].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	# xmin, xmax = ax[0][1].get_xlim()
	ax[0][1].hlines(real_returns.mean(), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[0][1].set_title('Mean', fontsize=8);
	# ax[0][1].set_ylim(-.5,.25);

	means = np.array(vols)[:,0]
	stds = np.array(vols)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[0][2].plot(means, color='grey', alpha=.5)
	ax[0][2].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[0][2].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	# xmin, xmax = ax[0][2].get_xlim()
	ax[0][2].hlines(real_returns.std(), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[0][2].set_title('Volatility', fontsize=8);
	# ax[0][2].set_ylim(.5,1.5);


	means = np.array(skews)[:,0]
	stds = np.array(skews)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[1][0].plot(means, color='grey', alpha=.5)
	ax[1][0].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[1][0].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	# xmin, xmax = ax[1][0].get_xlim()
	ax[1][0].hlines(skew(real_returns), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[1][0].set_title('Skewness', fontsize=8);
	# ax[1][0].set_ylim(-1,1);



	means = np.array(value_at_risks)[:,0]
	stds = np.array(value_at_risks)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[1][1].plot(means, color='grey', alpha=.5)
	ax[1][1].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[1][1].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	# xmin, xmax = ax[1][1].get_xlim()
	alpha = .95
	ax[1][1].hlines(-np.quantile(real_returns, 1-alpha, axis=0), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[1][1].set_title('VaR', fontsize=8);

	means = np.array(expected_shortfalls)[:,0]
	stds = np.array(expected_shortfalls)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[1][2].plot(means, color='grey', alpha=.5)
	ax[1][2].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[1][2].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	# xmin, xmax = ax[1][2].get_xlim()
	alpha = .95
	ax[1][2].hlines(-np.mean(real_returns[real_returns<np.quantile(real_returns, 1-alpha, axis=0)]), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[1][2].set_title('Expected Shortfall', fontsize=8);

	means = np.array(acfs_linear)[:,0]
	stds = np.array(acfs_linear)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[2][0].plot(means, color='grey', alpha=.5)
	ax[2][0].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][0].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][0].set_title('Linear ACF Similarity', fontsize=8);
	# xmin, xmax = ax[2][0].get_xlim()
	ax[2][0].hlines(0, xmin, xmax, linestyle='dashed', color='red', alpha=.5);

	means = np.array(acfs_vol_cluster)[:,0]
	stds = np.array(acfs_vol_cluster)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[2][1].plot(means, color='grey', alpha=.5)
	ax[2][1].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][1].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][1].set_title('Nonlinear ACF Similarity (squared returns)', fontsize=8);
	# xmin, xmax = ax[2][1].get_xlim()
	ax[2][1].hlines(0, xmin, xmax, linestyle='dashed', color='red', alpha=.5);


	means = np.array(acfs_leverage_eff)[:,0]
	stds = np.array(acfs_leverage_eff)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[2][2].plot(means, color='grey', alpha=.5)
	ax[2][2].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][2].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	ax[2][2].set_title('Nonlinear ACF Similarity (return, squared returns)', fontsize=8);
	# xmin, xmax = ax[2][2].get_xlim()
	ax[2][2].hlines(0, xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	
	means = np.array(kurts)[:,0]
	stds = np.array(kurts)[:,1]
	up_means = means + stds*conf_int
	down_means = means - stds*conf_int

	ax[3][0].plot(means, color='grey', alpha=.5)
	ax[3][0].plot(up_means, linestyle='dashed', color='grey', alpha=.2)
	ax[3][0].plot(down_means, linestyle='dashed', color='grey', alpha=.2)
	xmin, xmax = ax[1][1].get_xlim()
	ax[3][0].hlines(kurtosis(real_returns), xmin, xmax, linestyle='dashed', color='red', alpha=.5);
	ax[3][0].set_title('Kurtosis', fontsize=8);
	# ax[1][1].set_ylim(0,10);

	# ax[1][2].set_ylim(0,10);
	if save:
		plt.savefig(loc+'/stats_during_training_'+name+'.png')
		
	plt.close()
	
def plot_cum_perfs(synthetic_data, preprocessed_returns, post_processed_synthetic_data, returns, save=True, loc='', name=''):
	
	import matplotlib
	matplotlib.rc('xtick', labelsize=8)
	matplotlib.rc('ytick', labelsize=8)
	
	fig, ax = plt.subplots(1,2, figsize=(12,3))
	
	ax[0].plot(np.cumsum(synthetic_data, axis=1).T, color='grey', lw=.3, alpha=.5)
	ax[0].plot(np.cumsum(preprocessed_returns.T[0]), color='red', label='Real', lw=.5, alpha=.5)
	ax[0].set_ylabel('Cumulative log returns', fontsize=8)
	ax[0].set_title('Pre-processed', fontsize=10)
	ax[0].legend();
	
	ax[1].plot(np.cumsum(post_processed_synthetic_data, axis=1).T, color='grey', lw=.3, alpha=.5)
	ax[1].plot(np.cumsum(returns.T[0]), color='blue', label='Real', lw=.5, alpha=.5)
	ax[1].set_title('Post-processed', fontsize=10)
	ax[1].legend();
	
	if save:
		fig.savefig(loc+'/cum_returns'+name+'.png')
		
	plt.close()
	
	
def hill_estimator(data, k):
	order_statistics = np.sort(data)
	hill_estimators = np.log(-order_statistics[:k]) - np.log(-order_statistics[k])
	hill_estimator = np.mean(hill_estimators)
	se = np.std(hill_estimators)/np.sqrt(k)
	return hill_estimator, se
	

def plot_dendrogram(model, **kwargs):
	# Create linkage matrix and then plot the dendrogram

	# create the counts of samples under each node
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	linkage_matrix = np.column_stack(
		[model.children_, model.distances_, counts]
	).astype(float)

	# Plot the corresponding dendrogram
	dendrogram(linkage_matrix, **kwargs)