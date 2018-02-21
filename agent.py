import socket
import struct
from threading import Thread
from math import pi, atan2, asin, cos, sin
from sexpr import str2sexpr
import numpy as np
from interface import Perception, Action


class BaseAgent(object):
    def __init__(self,
                 simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='L2R',
                 player_id=0,
                 sync_mode=True):

        # Set sync_mode on or off
        self.sync_mode = sync_mode
        self.simspark_ip = simspark_ip
        self.simspark_port = simspark_port

        # Connect to environment
        self.connect(self.simspark_ip,
                     self.simspark_port)

        while player_id == 0:
            self.sense()
            self.act('')
            player_id = self.perception.game_state.unum

        self.player_id = player_id
        self.counter = 0

    def connect(self, simspark_ip, simspark_port):
        """
        Creates socket, connects to server, instantiates perception
        and sends required initial commands

        """
        self.socket = socket.socket(socket.AF_INET,
            socket.SOCK_STREAM)
        self.socket.connect((simspark_ip, simspark_port))
        self.perception = Perception()
        self.act('(scene rsg/agent/naov4/nao.rsg)')
        self.sense()  # only need to get msg from simspark
        init_cmd = ('(init (unum ' + str(self.player_id) \
                   + ')(teamname ' + teamname + '))')
        self.act(init_cmd)
        self.thread = None

    def reconnect(self):
        """
        Hacky way to env.reset() and start new episode used until
        easy way is found to reset joint values of robot.

        """
        # stackoverflow.com/questions/409783/socket-shutdown-vs-socket-close
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.connect(self.simspark_ip,
                     self.simspark_port)

    def sense(self):
        """
        Function to receive sensory data from the
        simulator and parse it the desired format.

        """
        length = ''
        while(len(length) < 4):
            length += self.socket.recv(4 - len(length))
        length = struct.unpack("!I", length)[0]
        msg = ''
        while len(msg) < length:
            msg += self.socket.recv(length - len(msg))
        sexp = str2sexpr(msg)
        self.perception.update(sexp)
        return self.perception


    def think(self, perception):
        """
        Think should be the function that can later be
        overwritten by higher-order Agent classes that
        implement more sophisticated control algorithms.

        """
        action = Action(perception)
        commands = action.create_command(self.counter)
        return commands


    def act(self, commands):
        # commands = action.create_joint_cmds()
        # commands = '(say hi)'
        if self.sync_mode:
            commands += '(syn)'
        self.socket.sendall(struct.pack("!I",
            len(commands)) + commands)


    def run(self):
        """
        This is the default code for testing the BaseAgent.
        When using DDPGAgent, I will

        """
        while True:
            perception = self.sense()
            action = self.think(perception)
            self.act(action)

    # def start(self):
    #     """
    #        Not sure what this function should be used for?
    #
    #     """
    #     if self.thread is None:
    #         self.thread = Thread(target=self.run)
    #         self.thread.daemon = True
    #         self.thread.start()


if '__main__' == __name__:
    agent = BaseAgent()
    agent.run()
