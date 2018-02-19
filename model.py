import numpy as np
import tensorflow as tf
from functools import partial

# Hidden layer standards
hidden_activation = tf.nn.relu
hidden_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=1.0, mode='FAN_IN', uniform=True )
hidden_layer = partial( tf.layers.dense,
    activation=hidden_activation, kernel_initializer=hidden_initializer )

class Actor( object ):
    def __init__( self, sess, hps, scope, trainable=True ):
        self.sess = sess
        self.s_dim = hps['s_dim']
        self.a_dim = hps['a_dim']
        self.a_bound = hps['a_bound']
        self.size_h1 = hps['h1_actor']
        self.size_h2 = hps['h2_actor']
        self.size_h3 = hps['h3_actor']
        self.actor_lr = hps['actor_lr']
        self.l2_reg_actor = hps['l2_reg_actor']
        self.trainable = trainable

        with tf.variable_scope( scope ):
            self.state = tf.placeholder( dtype=tf.float32,
                shape=[ None, self.s_dim ], name='state' )
            self.out = self.build_actor_network( self.state,
                self.trainable )

        self.actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope )

    def build_actor_network( self, state, trainable ):
        h1 = hidden_layer( state, self.size_h1, name='h1',
            trainable=trainable )
        h2 = hidden_layer( h1, self.size_h2, name='h2',
            trainable=trainable )
        h3 = hidden_layer( h2, self.size_h3, name='h3',
            trainable=trainable )
        out_initializer = tf.random_uniform_initializer( minval=-3*1e-3,
            maxval=3*1e-3 )
        out = tf.layers.dense( h2, self.a_dim, activation=tf.nn.tanh,
            kernel_initializer=out_initializer, name='out',
            trainable=trainable )
        out_scaled = tf.multiply( out, self.a_bound )
        return out_scaled

    def build_train_op( self, critic, scope ):
        with tf.variable_scope( scope ):
            self.critic = critic
            self.loss = -1 * tf.reduce_mean( self.critic.q2 )
            for var in self.actor_vars:
                if not 'bias' in var.name:
                    self.loss += self.l2_reg_actor * 0.5 * tf.nn.l2_loss( var )
            self.optimize = tf.train.AdamOptimizer( self.actor_lr ).\
                minimize( self.loss, var_list=self.actor_vars )

    def act( self, state ):
        return self.sess.run( self.out,
            feed_dict = { self.state: state } )

    def backward( self, state ):
        return self.sess.run( [ self.loss, self.optimize ],
            feed_dict = { self.critic.actor.state: state } )


class Critic( object ):
    def __init__( self, sess, hps, scope, trainable=True ):
        self.sess = sess
        self.s_dim = hps['s_dim']
        self.a_dim = hps['a_dim']
        self.size_h1 = hps['h1_critic']
        self.size_h2 = hps['h2_critic']
        self.size_h3 = hps['h3_critic']
        self.critic_lr = hps['critic_lr']
        self.l2_reg_critic = hps['l2_reg_critic']
        self.trainable = trainable

        with tf.variable_scope( scope ):
            self.state = tf.placeholder( dtype=tf.float32,
                shape=[ None, self.s_dim ], name='state' )
            self.action = tf.placeholder( dtype=tf.float32,
                shape=[ None, self.a_dim ], name='action' )
            self.q = self.build_critic_network( self.state, self.action,
                reuse=False, trainable=self.trainable )
            self.critic_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope )

    def build_critic_network( self, state, action, reuse, trainable ):
        h1s = hidden_layer( state, self.size_h1 , name='c_1',
            reuse=reuse, trainable=trainable)
        h2s = hidden_layer( h1s, self.size_h2 , name='c_2',
            reuse=reuse, trainable=trainable)
        h1a = hidden_layer( action, self.size_h2 , name='c_3',
            reuse=reuse, trainable=trainable)
        h3 = tf.concat( [ h2s, h1a ], axis=1 )
        h3 = hidden_layer( h3, self.size_h3 , name='c_4',
            reuse=reuse, trainable=trainable)
        q_initializer = tf.random_uniform_initializer( minval=-3*1e-3,
            maxval=3*1e-3 )
        q = tf.layers.dense( h3, 1, activation=None,
            kernel_initializer=q_initializer, name='c_5',
            reuse=reuse, trainable=trainable )
        return q

    def build_train_op( self, actor, scope ):
        with tf.variable_scope( scope ):
            self.actor = actor
            self.q2 = self.build_critic_network( self.actor.state,
                self.actor.out, reuse=True, trainable=self.trainable )
            self.target = tf.placeholder( dtype=tf.float32,
                shape=[ None, 1 ], name='target' )
            self.loss = tf.losses.mean_squared_error(
                self.q, self.target )
            for var in self.critic_vars:
                if not 'bias' in var.name:
                    self.loss += self.l2_reg_critic * 0.5 * tf.nn.l2_loss( var )
            self.optimize = tf.train.AdamOptimizer( self.critic_lr ).\
                minimize( self.loss, var_list=self.critic_vars )

    def predict( self, state, action ):
        return self.sess.run( self.q,
            feed_dict = { self.state: state,
                          self.action: action } )

    def backward( self, state, action, target ):
        return self.sess.run( [ self.loss, self.optimize ],
            feed_dict = { self.state: state,
                          self.action: action,
                          self.target: target } )
