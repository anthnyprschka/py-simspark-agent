import numpy as np
import tensorflow as tf
import gym
import itertools
from ddpg import Agent
from rm import ReplayMemory
from utils import do_hard_update
import itertools


def train(sess, hps):

    # Instantiate and load ReplayMemory
    rm = ReplayMemory( hps['buffer_size'],
                       hps['batch_size'] )

    # Instantiate agent
    agent = DDPGAgent( sess, hps, rm )

    # Setup TF and models
    sess.run( tf.global_variables_initializer() )
    do_hard_update( sess, 'actor_target', 'actor' )
    do_hard_update( sess, 'critic_target', 'critic' )

    # Connect to environment
    s1 = agent.connect()

    for i in range( 0, hps['num_episodes'] ):

        agent.ou.reset()
        ep_reward = 0

        for t in itertools.count():

            a1 = agent.think( s1, i )
            agent.act( a1 )
            s2, r, d = agent.sense()

            rm.add( s1, a1, s2, r )

            if rm.size() > hps['batch_size']:
                agent.learn()

            ep_reward += r
            s1 = s2

            if d:
                # TODO: Logging
                s1 = agent.reconnect()


def main():
    hps = {
        'render_every': 30,
        'mode': 'training',
        'num_episodes': 10000,
        'buffer_size': 100000,
        'batch_size': 128,
        'noise_decay': 0.9999,
        'actor_lr': 0.0001,
        'critic_lr': 0.001,
        'tau': 0.001,
        'gamma': 0.99,
        'h1_actor': 400,
        'h2_actor': 300,
        'h3_actor': 300,
        'h1_critic': 400,
        'h2_critic': 400,
        'h3_critic': 300,
        'l2_reg_actor': 1e-6,
        'l2_reg_critic': 1e-6 }

    tf.reset_default_graph()
    with tf.Session() as sess:
        train(sess, hps)


if __name__ == '__main__':
    main()

