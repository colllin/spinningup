# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import gym
import time
import sklearn.cluster
import collections
import numpy_indexed as npi
from tqdm import tqdm
from spinup.algos.ddpg_mixup import core
from spinup.algos.ddpg_mixup.core import get_vars
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])
 

class MixupReplayBuffer:
    """
    Wraps another replay buffer and applies mixup when sampling batches.
    """
    def __init__(self, replay_buffer, mixup_alpha=0):
        self.replay_buffer = replay_buffer
        self.mixup_alpha = mixup_alpha
        
    def store(self, *args, **kwargs):
        self.replay_buffer.store(*args, **kwargs)
    
    def interpolate(self, a1, a2, lam):
        return lam*a1.astype(float) + (1-lam)*a2.astype(float)        
        
    def sample_batch(self, mixup_alpha=None, *args, **kwargs):
        if mixup_alpha == None:
            mixup_alpha = self.mixup_alpha
        batch = self.replay_buffer.sample_batch(*args, **kwargs)
        if mixup_alpha > 0:
            mixup_lambda = np.random.beta(mixup_alpha, mixup_alpha)
            b2 = self.replay_buffer.sample_batch(*args, **kwargs)
            batch = {
                k: np.minimum(batch[k], b2[k]) if k in ['done','d'] else self.interpolate(batch[k], b2[k], lam=mixup_lambda)
            for k in batch.keys()}
        return batch


class ClusteredMixupReplayBuffer:
    """
    Performs clustered mixup when sampling batches.
    """

    def __init__(self, *rb_args, n_clusters=32, mixup_alpha=0, cluster_on='obs2,done', batch_size=100, logger=None, **rb_kwargs):
        self.n_clusters = n_clusters
        self.dirty = True
        self.steps_since_clustered = 0
        self.mixup_alpha = mixup_alpha
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(*rb_args, **rb_kwargs)
        self.cluster_buf = np.zeros(self.replay_buffer.max_size, dtype=int)
        self.cluster_on = cluster_on.split(',')
        self.steps_before_recluster = 1e4
        self.logger = logger

    def store(self, *args, **kwargs):
        self.dirty = True
        self.steps_since_clustered += 1
        self.replay_buffer.store(*args, **kwargs)
        self.predict_cluster(self.replay_buffer.ptr - 1)

    def get_data_to_cluster(self):
        cluster_key2buf = {
            'obs1': self.replay_buffer.obs1_buf,
            'obs2': self.replay_buffer.obs2_buf,
            'acts': self.replay_buffer.acts_buf,
            'rews': self.replay_buffer.rews_buf,
            'done': self.replay_buffer.done_buf,
        }
        bufs = [cluster_key2buf[k] for k in self.cluster_on]
        bufs = [b.reshape(-1,1) if b.ndim == 1 else b for b in bufs]
        stacked_bufs = np.hstack(bufs)
        return stacked_bufs[:self.replay_buffer.size]

    def due_for_recluster(self):
        return self.steps_since_clustered >= self.steps_before_recluster
    
    def predict_cluster(self, idx):
        if hasattr(self, 'kmeans'):
            self.cluster_buf[idx:idx+1] = self.kmeans.predict(self.get_data_to_cluster()[idx:idx+1])
        else:
            # Randomly assign clusters until we fit kmeans.
            self.cluster_buf[idx:idx+1] = idx % self.n_clusters
        
    def recluster(self):
        data_to_cluster = self.get_data_to_cluster()
        self.kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=self.n_clusters)
        self.cluster_buf[:len(data_to_cluster)] = self.kmeans.fit_predict(data_to_cluster)
        self.steps_since_clustered = 0
        # self.cluster_counts = np.zeros(self.n_clusters)
        # for c in self.cluster_buf[:len(data_to_cluster)]:
        #     self.cluster_counts[c] += 1
        # self.logger.store(
        #     ClusterInertia=inertia,
        #     MinClusterCounts=self.cluster_counts.min(),
        #     MaxClusterCounts=self.cluster_counts.max(),
        #     AverageClusterCounts=self.cluster_counts.mean(),
        #     StdClusterCounts=self.cluster_counts.std()
        # )

    def update_cluster2idxs(self):
        sample_clusters = self.cluster_buf[:self.replay_buffer.size]
        groups = npi.group_by(sample_clusters)
        self.cluster2idxs = groups.split_array_as_list(np.arange(len(sample_clusters)))
        self.cluster_counts = groups.count
#         sample_clusters = self.cluster_buf[:self.replay_buffer.size]
#         self.cluster2idxs = collections.defaultdict(list)
#         self.cluster_counts = np.zeros(self.n_clusters)
#         for idx, cluster in enumerate(sample_clusters):
#             self.cluster2idxs[cluster].append(idx)
#             self.cluster_counts[cluster] += 1
#         self.cluster_counts = self.cluster_counts[:len(self.cluster2idxs.keys())]
        self.dirty = False
        
    def interpolate(self, a1, a2, lam):
        return lam*a1.astype(float) + (1-lam)*a2.astype(float)        
        
    def collate_batch(self, idxs):
        return dict(obs1=self.replay_buffer.obs1_buf[idxs],
                    obs2=self.replay_buffer.obs2_buf[idxs],
                    acts=self.replay_buffer.acts_buf[idxs],
                    rews=self.replay_buffer.rews_buf[idxs],
                    done=self.replay_buffer.done_buf[idxs])
                    
    def sample_batch(self, batch_size=32, mixup_alpha=None):
        """
        mixup_alpha (float): Optional, will override the mixup_alpha specified when constructing the instance.
        """
        if mixup_alpha == None:
            mixup_alpha = self.mixup_alpha
        if self.due_for_recluster():
            self.recluster()
        if self.dirty:
            self.update_cluster2idxs()
         
        # First, choose <batch_size> clusters at random
        cluster_p = self.cluster_counts / self.cluster_counts.sum()
        batch_clusters = np.random.choice(np.arange(len(self.cluster2idxs)), size=batch_size, p=cluster_p)
        
        # Then select a random member from each cluster to form the batch.
        idxs = [np.random.choice(self.cluster2idxs[c]) for c in batch_clusters]
        batch = self.collate_batch(idxs)
        
        if mixup_alpha > 0:
            mixup_lambda = np.random.beta(mixup_alpha, mixup_alpha)
            
            # Select another random member from each of THE SAME clusters to form a second batch.
            idxs = [np.random.choice(self.cluster2idxs[c]) for c in batch_clusters]
            b2 = self.collate_batch(idxs)
            
            batch = {
                k: np.minimum(batch[k], b2[k]) if k in ['done','d'] else self.interpolate(batch[k], b2[k], lam=mixup_lambda)
            for k in batch.keys()}

        return batch
                           

"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg_mixup(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, mixup_alpha=0, mixup_n_clusters=0, mixup_cluster_on='obs1,done', batch_size=100, start_steps=10000, 
         act_noise=0.1, max_ep_len=1000, logger_kwargs=dict(), save_freq=1, find_lr=False):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
            
        find_lr (bool): Whether to run the learning rate finder. Afterward,
            plot LossQ, LrQ, LossPi, LrPi to evaluate the best learning rate.

    """
    if find_lr:
    	epochs = 50

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target networks
    with tf.variable_scope('target'):
        # Note that the action placeholder going to actor_critic here is 
        # irrelevant, because we only need q_targ(s, pi_targ(s)).
        pi_targ, _, q_pi_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    if mixup_n_clusters > 0:
        assert mixup_alpha > 0, 'Did you mean to specify mixup_alpha=0 with mixup_n_clusters > 0?'
        replay_buffer = ClusteredMixupReplayBuffer(
            n_clusters=mixup_n_clusters, mixup_alpha=mixup_alpha, cluster_on=mixup_cluster_on, batch_size=batch_size,
            obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
            logger=logger
        )
    else:
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        if mixup_alpha > 0:
            replay_buffer = MixupReplayBuffer(replay_buffer, mixup_alpha=mixup_alpha)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

    # Separate train ops for pi, q
    pi_step = tf.Variable(0, trainable=False)
    q_step = tf.Variable(0, trainable=False)
    if find_lr:
        lr_finder_schedule = lambda global_step: tf.train.exponential_decay(
            learning_rate=1e-5,
            global_step=global_step,
            decay_steps=50000,
            decay_rate=10,
            staircase=False,
        )
        pi_lr = lr_finder_schedule(global_step=pi_step)
        q_lr = lr_finder_schedule(global_step=q_step)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'), global_step=pi_step)
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'), global_step=q_step)

    # Polyak averaging for target variables
    # TODO: Separate polyak per q and pi? Computed based on LR?
    polyak = 1 - (5 * tf.minimum(q_lr, pi_lr))
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q': q})

    def get_action(o, noise_scale):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(1,-1)})[0]
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in tqdm(range(total_steps)):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            # act_noise_gamma = 1 - pi_lr
            # act_noise_gamma = .4
            # act_noise_decayed = act_noise * act_noise_gamma ** np.log(t / start_steps) # Decays to act_noise at t=start_steps, act_noise*gamma at t=10*start_steps, ...
            # a = get_action(o, act_noise_decayed)
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }

                # Q-learning update
                if find_lr:
                    outs = sess.run([q_loss, q, train_q_op, q_lr], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1], LrQ=outs[3])
                else:
                    outs = sess.run([q_loss, q, train_q_op], feed_dict)
                    logger.store(LossQ=outs[0], QVals=outs[1])

                # Policy update
                if find_lr:
                    outs = sess.run([pi_loss, train_pi_op, target_update, pi_lr], feed_dict)
                    logger.store(LossPi=outs[0], LrPi=outs[3])
                else:
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            if find_lr:
                logger.log_tabular('LrPi', average_only=True)
                logger.log_tabular('LrQ', average_only=True)
#             if mixup_n_clusters > 0:
#                 logger.log_tabular('ClusterInertia', average_only=True)
#                 logger.log_tabular('AverageClusterCounts', average_only=True)
#                 logger.log_tabular('StdClusterCounts', average_only=True)
#                 logger.log_tabular('MaxClusterCounts', average_only=True)
#                 logger.log_tabular('MinClusterCounts', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg_mixup')
    parser.add_argument('--find_lr', type=bool, action=store_true)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ddpg_mixup(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs, find_lr=args.find_lr,
         logger_kwargs=logger_kwargs)
    