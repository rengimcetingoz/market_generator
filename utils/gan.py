from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow import convert_to_tensor
from math import floor, ceil
from scipy.stats import skew, kurtosis, wasserstein_distance
import matplotlib.pyplot as plt

import pickle
import tensorflow as tf
import numpy as np

import sys
sys.path.append('./utils')
import generate_charts as my_funcs

class GAN:

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    def __init__(self, discriminator, generator, training_input, real_returns, preprocessed_returns, noise_scale=1, lr_d=1e-4, lr_g=3e-4, epsilon=1e-8, beta_1=.0, beta_2=0.9, noise_dist='normal', from_logits=True, store_stats=False, store_freq = 50, post_process_func=None, file_loc=''):
        """Create a GAN instance

        Args:
            discriminator (tensorflow.keras.models.Model): Discriminator model.
            generator (tensorflow.keras.models.Model): Generator model.
            training_input (int): input size of temporal axis of noise samples.
            real_returns (array): original time-series for evaulation during the training
            preprocessed_returns (array): processed (trnasformed) original time-series for evaulation during the training
            noise_scale (float, optional): to increase the variance of the geenrated noise fed to the generator
            lr_d (float, optional): Learning rate of discriminator. Defaults to 1e-4.
            lr_g (float, optional): Learning rate of generator. Defaults to 3e-4.
            epsilon (float, optional): Epsilon paramater of Adam. Defaults to 1e-8.
            beta_1 (float, optional): Beta1 parameter of Adam. Defaults to 0.
            beta_2 (float, optional): Beta2 parameter of Adam. Defaults to 0.9.
            noise_dist (str, optional) : noise distribution (should be 'normal' or 'uniform')
            from_logits (bool, optional): Output range of discriminator, logits imply output on the entire reals. Defaults to True.
            store_stats (bool, optional): stores some stats during the training if True
            store_freq (int, optional) : storing frequency for computed stats
            post_process_func (func) : to do post-processing during training to evaluate
            file_loc (str, optional) : location to save the generated charts and the models

        """
        self.discriminator = discriminator
        self.generator = generator
        self.noise_shape = [self.generator.input_shape[1], training_input, self.generator.input_shape[-1]]
        self.scale = noise_scale
        self.real_returns = real_returns
        self.preprocessed_returns = preprocessed_returns
        self.store_stats = store_stats
        self.store_freq = store_freq
        self.post_process_func = post_process_func
        self.file_loc = file_loc
        self.noise_dist = noise_dist
        
        self.gen_losses = []
        self.disc_losses = []
        
        self.wasses = []
        self.means = []
        self.vols = []
        self.skews = []
        self.kurts = []
        self.value_at_risks = []
        self.expected_shortfalls = []
        self.acfs_linear = []
        self.acfs_vol_cluster = []
        self.acfs_leverage_eff = []
        
        self.wasses_post = []
        self.means_post = []
        self.vols_post = []
        self.skews_post = []
        self.kurts_post = []
        self.value_at_risks_post = []
        self.expected_shortfalls_post = []
        self.acfs_linear_post = []
        self.acfs_vol_cluster_post = []
        self.acfs_leverage_eff_post = []        

        self.loss = BinaryCrossentropy(from_logits=from_logits)

        self.generator_optimizer = Adam(lr_g, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
        self.discriminator_optimizer = Adam(lr_d, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)

    def train(self, data, batch_size, n_batches, additional_d_steps):
        """training function of a GAN instance.
        Args:
            data (4d array): Training data in the following shape: (samples, n_series, timesteps, 1).
            batch_size (int): Batch size used during training.
            n_batches (int): Number of update steps taken.
            additional_d_steps (int): Number of extra discriminator training steps during each update.
        """ 
        progress = Progbar(n_batches)

        for n_batch in range(n_batches):
            # sample uniformly
            batch_idx = np.random.choice(np.arange(data.shape[0]), size=batch_size, replace=(batch_size > data.shape[0]))
            batch = data[batch_idx]
            
            if self.store_stats and n_batch%self.store_freq==0:
                
                # batch loss
                if self.noise_dist=='normal':
                    noise = tf.random.normal([batch_size, *self.noise_shape]) * self.scale
                elif self.noise_dist=='uniform':
                    noise = tf.random.uniform([batch_size, *self.noise_shape]) * self.scale
                synthetic_data = self.generator(noise).numpy()
                fake_output = self.discriminator(synthetic_data, training=False)
                real_output = self.discriminator(batch, training=False)
                self.disc_losses.append(self.discriminator_loss(real_output, fake_output).numpy())
                self.gen_losses.append(self.generator_loss(fake_output).numpy())   
                
                # save loss chart
                fig, ax = plt.subplots()
                ax.plot(self.gen_losses, label='Generator loss', color='red')
                ax.plot(self.disc_losses, label='Discriminator loss', color='grey')
                ax.set_xlim(0,n_batches//self.store_freq)
                
                ax.legend()
                fig.savefig(self.file_loc+'/loss_training.png')
                plt.close()
                
                # generate syn data
                length_syn = len(self.real_returns)
                window = int((self.noise_shape[1]+1)/2)
                if self.noise_dist=='normal':
                    noise = tf.random.normal([batch_size, 1, length_syn+window-1, 3]) * self.scale
                elif self.noise_dist=='uniform':
                    noise = tf.random.uniform([batch_size, 1, length_syn+window-1, 3]) * self.scale
                synthetic_data = self.generator(noise).numpy().squeeze()
                synthetic_data = (synthetic_data - synthetic_data.mean())/synthetic_data.std()
                post_synthetic_data = self.post_process_func(synthetic_data)
                
                # plot cum perfs
                my_funcs.plot_cum_perfs(synthetic_data, self.preprocessed_returns, post_synthetic_data, self.real_returns, loc=self.file_loc+'/cum_return_charts', name='_'+str(n_batch))
                
                # plot stylized facts
                img_pre = my_funcs.fig2img(my_funcs.plot_stylized_facts(self.preprocessed_returns, synthetic_data, window_acf=window, title='Pre-processed', save_bool=False, save_name=self.file_loc+'',color='red'))
                img_post = my_funcs.fig2img(my_funcs.plot_stylized_facts(self.real_returns, post_synthetic_data, window_acf=window, title='Post-processed', save_bool=False, save_name=self.file_loc+'',color='blue'))
                img_final = my_funcs.Images_pad_vertical([img_pre,img_post], h_center=True)
                img_final.save(self.file_loc+"/stylized_facts_charts/iter_"+str(n_batch)+".png")
                
                # compute stats for pre-syn
                wasses, means,vols,skews,kurts,vars,es,acfs_linear,acfs_vol_cluster,acfs_lev_eff = my_funcs.stats(self.preprocessed_returns, synthetic_data, acf_window=window)
                self.wasses.append(wasses)
                self.means.append(means)
                self.vols.append(vols)
                self.skews.append(skews)
                self.kurts.append(kurts)
                self.value_at_risks.append(vars)
                self.expected_shortfalls.append(es)
                self.acfs_linear.append(acfs_linear)
                self.acfs_vol_cluster.append(acfs_vol_cluster)
                self.acfs_leverage_eff.append(acfs_lev_eff)
                
                # plot stats for pre-syn
                my_funcs.save_stats_charts(wasses = self.wasses,
                                            means_ = self.means,
                                            vols = self.vols,
                                            skews = self.skews,
                                            kurts = self.kurts,
                                            value_at_risks = self.value_at_risks,
                                            expected_shortfalls = self.expected_shortfalls,
                                            acfs_linear = self.acfs_linear,
                                            acfs_vol_cluster = self.acfs_vol_cluster,
                                            acfs_leverage_eff = self.acfs_leverage_eff,
                                            real_returns = self.preprocessed_returns,
                                            save=True,
                                            name='Pre',
                                            loc=self.file_loc,
                                            up_xlim=n_batches//self.store_freq)
                
                # compute stats for post-syn
                wasses_post, means_post,vols_post,skews_post,kurts_post,vars_post,es_post,acfs_linear_post,acfs_vol_cluster_post,acfs_lev_eff_post = my_funcs.stats(self.real_returns, post_synthetic_data, acf_window=window)
                self.wasses_post.append(wasses_post)
                self.means_post.append(means_post)
                self.vols_post.append(vols_post)
                self.skews_post.append(skews_post)
                self.kurts_post.append(kurts_post)
                self.value_at_risks_post.append(vars_post)
                self.expected_shortfalls_post.append(es_post)
                self.acfs_linear_post.append(acfs_linear_post)
                self.acfs_vol_cluster_post.append(acfs_vol_cluster_post)
                self.acfs_leverage_eff_post.append(acfs_lev_eff_post)
                
                # plot stats for post-syn
                my_funcs.save_stats_charts(wasses = self.wasses_post,
                                            means_ = self.means_post,
                                            vols = self.vols_post,
                                            skews = self.skews_post,
                                            kurts = self.kurts_post,
                                            value_at_risks = self.value_at_risks_post,
                                            expected_shortfalls = self.expected_shortfalls_post,
                                            acfs_linear = self.acfs_linear_post,
                                            acfs_vol_cluster = self.acfs_vol_cluster_post,
                                            acfs_leverage_eff = self.acfs_leverage_eff_post,
                                            real_returns = self.real_returns,
                                            save=True,
                                            name='Post',
                                            loc=self.file_loc,
                                            up_xlim=n_batches//self.store_freq)
                
            self.train_step(batch, batch_size, additional_d_steps)

            self.train_hook(n_batch)

            progress.update(n_batch + 1)

    @tf.function
    def train_step(self, data, batch_size, additional_d_steps):

        for _ in range(additional_d_steps + 1):

            if self.noise_dist=='normal':
                noise = tf.random.normal([batch_size, *self.noise_shape]) * self.scale
            elif self.noise_dist=='uniform':
                noise = tf.random.uniform([batch_size, *self.noise_shape]) * self.scale
            # noise = tf.random.normal([batch_size, *self.noise_shape])
            generated_data = self.generator(noise, training=False)

            with tf.GradientTape() as disc_tape:
                real_output = self.discriminator(data, training=True)
                fake_output = self.discriminator(generated_data, training=True)
                disc_loss = self.discriminator_loss(real_output, fake_output)
            
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        noise = tf.random.normal([batch_size, *self.noise_shape])
        generated_data = self.generator(noise, training=False)
        
        noise = tf.random.normal([batch_size, *self.noise_shape])
        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.discriminator(generated_data, training=False)
            gen_loss = self.generator_loss(fake_output)
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train_hook(self, n_batch):
        """Override this method to insert behaviour at every training update.
           Acces to the instance namespace is provided.
        """        
        pass
