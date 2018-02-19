from agent import Agent
import numpy as np
from utils import OrnsteinUhlenbeck
from model import Actor, Critic
from utils import build_soft_update_op
import tensorflow as tf
from pprint import pprint

import logging
FORMAT = '%(asctime)-15s %(message)s'
logger = logging.basicConfig(level=logging.DEBUG, format=FORMAT)


class DDPGAgent( BaseAgent ):
    def __init__( self, sess, hps, rm ):

        # TODO: Here muss ich vermutlich auch noch die Parameter
        # für den BaseAgent einfügen
        super(DDPGAgent, self).__init__()

        self.sess = sess
        self.hps = hps
        self.rm = rm
        self.ou = OrnsteinUhlenbeck( hps['a_dim'] )
        self.gamma = hps['gamma']
        self.tau = hps['tau']
        self.a_bound = hps['a_bound']
        self.noise_decay = hps['noise_decay']

        self.actor = Actor( self.sess, self.hps,
            'actor', trainable=True )
        self.actor_target = Actor( self.sess, self.hps,
            'actor_target', trainable=False )
        self.critic = Critic( self.sess, self.hps,
            'critic', trainable=True )
        self.critic_target = Critic( self.sess, self.hps,
            'critic_target', trainable=False )

        self.critic.build_train_op( self.actor, 'critic' )
        self.actor.build_train_op( self.critic, 'actor' )

        self.actor_soft_update_op = build_soft_update_op(
            self.sess, 'actor_target', 'actor', self.tau )
        self.critic_soft_update_op = build_soft_update_op(
            self.sess, 'critic_target', 'critic', self.tau )

    def explore( self, state, i ):
        action = self.actor.act( state )
        # action += ( self.ou.sample() * self.a_bound * self.noise_decay ** i )
        action += ( self.ou.sample() * self.noise_decay ** i )
        return action

    def exploit( self, state ):
        action = self.actor.act( state )
        return action

    def think( self, state, i ):
        if self.hps['mode'] == 'training':
            self.explore( state, i )
        else:
            self.exploit( state )

    def learn( self ):
        s1, a1, r1, s2 = self.rm.sample()

        # Optimize critic
        a2 = self.actor_target.act( s2 )
        q2 = self.critic_target.predict( s2, a2 )
        y1 = r1 + self.gamma * q2
        loss, _ = self.critic.backward( s1, a1, y1 )

        # Optimize actor
        loss, _ = self.actor.backward( s1 )

        self.sess.run( self.actor_soft_update_op )
        self.sess.run( self.critic_soft_update_op )



