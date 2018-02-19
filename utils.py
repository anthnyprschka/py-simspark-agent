import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pprint import pprint


class OrnsteinUhlenbeck( object ):
    def __init__( self, a_dim, mu=0.0, sigma=0.2, theta=0.15 ):
        self.a_dim = a_dim
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.x = self.reset()

    def reset( self ):
        x = np.ones( self.a_dim ) * self.mu
        return x

    def sample( self ):
        dx = self.theta * ( self.mu - self.x )
        dx += self.sigma * np.random.randn( self.a_dim )
        self.x = self.x + dx
        return self.x

    def plot_process( self ):
        process = []
        for x in range(0, 100):
            process.append( ou.sample() )
        plt.plot( process )
        plt.show()


def do_hard_update( sess, target_scope, source_scope ):
    target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope=target_scope )
    source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=source_scope )
    hard_update = [ tf.assign( target_params[i], source_params[i] )
        for i in range( len(  target_params ) ) ]
    sess.run( hard_update )


def build_soft_update_op( sess, target_scope, source_scope, tau ):
    target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
        scope=target_scope )
    source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=source_scope )
    soft_update_op = [ tf.assign( target_params[i],
        tf.multiply( target_params[i], ( 1 - tau ) ) +
        tf.multiply( source_params[i], tau ) )
        for i in range( len( target_params ) ) ]
    return soft_update_op


def save_model(  ):
    # TODO
    #
    #
    return


def load_model(  ):
    # TODO
    #
    #
    return


def build_summaries(  ):
    # TODO
    #
    #
    return


if __name__ == '__main__':
    ou = OrnsteinUhlenbeck( 1 )
    ou.plot_process()
