import numpy as np
from math import pi, atan2, asin, cos, sin
DEG_TO_RAD = pi / 180

# PERCEPTORS
# For L2R, I think only the joint sensors (& effectors) are relevant.
HINGE_JOINT_PERCEPTOR                       = "HJ"               #  0
UNIVERSAL_JOINT_PERCEPTOR                   = "UJ"               #  1
TOUCH_PERCEPTOR                             = "TCH"              #  2
FORCE_RESISTANCE_PERCEPTOR                  = "FRP"              #  3
ACCELEROMETER_PERCEPTOR                     = "ACC"              #  4
GYRO_RATE_PERCEPTOR                         = "GYR"              #  5
GAME_STATE_PERCEPTOR                        = "GS"               #  6
GPS_PERCEPTOR                               = "GPS"              #  7
BAT_PERCEPTOR                               = "BAT"              #  8
VISION_PERCEPTOR                            = "See"              #  9
VISION_PERCEPTOR_TRUE_BALL                  = "ballpos"          # 10
VISION_PERCEPTOR_TRUE_POS                   = "mypos"            # 11
VISION_PERCEPTOR_BALL                       = "B"                # 12
VISION_PERCEPTOR_LINE                       = "L"                # 13
VISION_PERCEPTOR_TOP_RIGHT_FIELD_CORNER     = "F1R"              # 14
VISION_PERCEPTOR_BOTTOM_RIGHT_FIELD_CORNER  = "F2R"              # 15
VISION_PERCEPTOR_TOP_LEFT_FIELD_CORNER      = "F1L"              # 16
VISION_PERCEPTOR_BOTTOM_LEFT_FIELD_CORNER   = "F2L"              # 17
VISION_PERCEPTOR_TOP_RIGHT_GOAL_POST        = "G1R"              # 18
VISION_PERCEPTOR_BOTTOM_RIGHT_GOAL_POST     = "G2R"              # 19
VISION_PERCEPTOR_TOP_LEFT_GOAL_POST         = "G1L"              # 20
VISION_PERCEPTOR_BOTTOM_LEFT_GOAL_POST      = "G2L"              # 21
VISION_PERCEPTOR_AGENT                      = "P"                # 22
VISION_PERCEPTOR_HEAD                       = "head"             # 23
VISION_PERCEPTOR_RIGHT_LOWER_ARM            = "rlowerarm"        # 24
VISION_PERCEPTOR_LEFT_LOWER_ARM             = "llowerarm"        # 25
VISION_PERCEPTOR_RIGHT_FOOT                 = "rfoot"            # 26
VISION_PERCEPTOR_LEFT_FOOT                  = "lfoot"            # 27
BOTTOM_CAMERA                               = 'BottomCamera'     # 28
TOP_CAMERA                                  = 'TopCamera'        # 29
REWARD_PERCEPTOR                            = 'R'                # 30
ISFALLEN_PERCEPTOR                          = 'isfallen'         # 31


# Note that direction of mapping is reversed in the subseding object!
JOINT_SENSOR_NAMES = { "hj1":  'HeadYaw',                        #  0
                       "hj2":  'HeadPitch',                      #  1
                       "laj1": 'LShoulderPitch',                 #  2
                       "laj2": 'LShoulderRoll',                  #  3
                       "laj3": 'LElbowYaw',                      #  4
                       "laj4": 'LElbowRoll',                     #  5
                       "llj1": 'LHipYawPitch',                   #  6
                       "llj2": 'LHipRoll',                       #  7
                       "llj3": 'LHipPitch',                      #  8
                       "llj4": 'LKneePitch',                     #  9
                       "llj5": 'LAnklePitch',                    # 10
                       "llj6": 'LAnkleRoll',                     # 11
                       "raj1": 'RShoulderPitch',                 # 12
                       "raj2": 'RShoulderRoll',                  # 13
                       "raj3": 'RElbowYaw',                      # 14
                       "raj4": 'RElbowRoll',                     # 15
                       "rlj1": 'RHipYawPitch',                   # 16
                       "rlj2": 'RHipRoll',                       # 17
                       "rlj3": 'RHipPitch',                      # 18
                       "rlj4": 'RKneePitch',                     # 19
                       "rlj5": 'RAnklePitch',                    # 20
                       "rlj6": 'RAnkleRoll' }                    # 21

# EFFECTORS
# Note that direction of mapping is reversed in the preceding object!
JOINT_CMD_NAMES = { 'HeadYaw':        "he1",                     #  0
                    'HeadPitch':      "he2",                     #  1
                    'LShoulderPitch': "lae1",                    #  2
                    'LShoulderRoll':  "lae2",                    #  3
                    'LElbowYaw':      "lae3",                    #  4
                    'LElbowRoll':     "lae4",                    #  5
                    'LHipYawPitch':   "lle1",                    #  6
                    'LHipRoll':       "lle2",                    #  7
                    'LHipPitch':      "lle3",                    #  8
                    'LKneePitch':     "lle4",                    #  9
                    'LAnklePitch':    "lle5",                    # 10
                    'LAnkleRoll':     "lle6",                    # 11
                    'RShoulderPitch': "rae1",                    # 12
                    'RShoulderRoll':  "rae2",                    # 13
                    'RElbowYaw':      "rae3",                    # 14
                    'RElbowRoll':     "rae4",                    # 15
                    'RHipYawPitch':   "rle1",                    # 16
                    'RHipRoll':       "rle2",                    # 17
                    'RHipPitch':      "rle3",                    # 18
                    'RKneePitch':     "rle4",                    # 19
                    'RAnklePitch':    "rle5",                    # 20
                    'RAnkleRoll':     "rle6" }                   # 21


# Some joints are inverted in simspark compared with real NAO
INV_JOINTS = [ 'HeadPitch',
               'LShoulderPitch',
               'RShoulderPitch',
               'LHipPitch',
               'RHipPitch',
               'LKneePitch',
               'RKneePitch',
               'LAnklePitch',
               'RAnklePitch' ]


class Perception:
    def __init__(self):
        self.time = 0
        self.joint = {}
        self.joint_temp = {}
        self.fsr = {}
        self.see = [{}, {}]
        self.game_state = GameState()
        self.gps = {}
        self.imu = [0, 0] # [AngleX, AngleY]

    def update(self, sexp):
        for s in sexp:
            name = s[0]

            # Time
            if name == 'time':
                self.time = float(s[1][1])

            # GameState
            elif name == GAME_STATE_PERCEPTOR:
                self.game_state.update(s[1:])

            # GyroRate
            elif name == GYRO_RATE_PERCEPTOR:
                self.gyr = [float(v) for v in s[2][1:]]

            # Accelerometer
            elif name == ACCELEROMETER_PERCEPTOR:
                self.acc = [float(v) for v in s[2][1:]]

            # HingeJoints
            elif name == HINGE_JOINT_PERCEPTOR:
                jointv = {}

                # In this case, s[0] is the meta-identifier
                # "HJ". In every other element of s, it seems
                # there is another list of length 2 that holds
                #
                for i in s[1:]:
                    jointv[i[0]] = i[1]

                # What is this thing doing? Where is the 'n'
                # coming from? 'n' must have been the first
                # element of one of the s[1:] lists, right?
                # Because the keys of jointv have been defined
                # by the first elements of the lists in s[1:]
                name = JOINT_SENSOR_NAMES[jointv['n']]

                # I assume that 'ax' stands for accerelation.
                # Here we transform the value that is delivered
                # by the server into our internal value for the
                # joint. We store the joint *accelerations* if
                # I am not mistaken. We do not store the joint
                # positions or its rotation for that matter.
                #
                # This is what I care for. This stuff I wanna
                # store in an array, in a definitive order, in
                # order to be able to feed it to the network.
                if 'ax' in jointv:
                    self.joint[name] = float(jointv['ax']) * \
                        DEG_TO_RAD * \
                        (-1 if name in INV_JOINTS else 1)

                # I assume that 'tp' stands for temperature
                # Apparently, temperature needn't be converted
                # to anything. I assume temperature is a quantity
                # introduced to enhance realism of the simulation.
                #
                if 'tp' in jointv:
                    self.joint_temp[name] = float(jointv['tp'])

            # Vision / TopCamera
            elif name == VISION_PERCEPTOR or \
                 name == TOP_CAMERA:
                self.see[0] = self._parse_vision(s[1:])

            # BottomCamera
            elif name == BOTTOM_CAMERA:
                self.see[1] = self._parse_vision(s[1:])

            # ForceResistance
            elif name == FORCE_RESISTANCE_PERCEPTOR:
                self.fsr[s[1][1]] = \
                    { s[2][0]: [float(v) for v in s[2][1:]],
                      s[3][0]: [float(v) for v in s[3][1:]] }

            # GPS
            elif name == GPS_PERCEPTOR:
                self.gps[s[1][1]] = [float(v) for v in s[2][1:]]

            # BAT?
            elif name == BAT_PERCEPTOR:
                self.bat = float(s[1])

            # Reward
            elif name == REWARD_PERCEPTOR:
              print s

            # IsFallen
            elif name == ISFALLEN_PERCEPTOR:
              print s

            # else:
            #     raise RuntimeError('unknown sensor: ' + str(s))

        if 'torso' in self.gps:
            data = self.gps['torso']
            angX = atan2(data[9], data[10])
            angY = asin(-data[8])
            # convert angle range: angY in [-pi, pi],
            # angX in [-pi/2, pi/2]
            if (abs(angX) > pi / 2):
                angX = pi + angX
                angX = atan2(sin(angX), cos(angX))  # normalize
                angY = pi - angY
                angY = atan2(sin(angY), cos(angY))  # normalize
            self.imu = [angX, angY]

    def _parse_vision(self, sexp):
        see = {}
        see[VISION_PERCEPTOR_LINE] = []
        see[VISION_PERCEPTOR_AGENT] = []

        for i in sexp:
            if i[0] == VISION_PERCEPTOR_LINE or \
               i[0] == VISION_PERCEPTOR_AGENT:
                see[i[0]].append(i[1:])
            else:
                see[i[0]] = i[1:]
        return see


class Action(object):
    def __init__(self, perception):
        self.speed = {}
        self.stiffness = {}
        self.counter = 0


    def create_joint_cmds(self):
        """
        Converts joint commands to message ready to be send
        to the environment.

        """
        speed = ['(%s %.2f)' % (JOINT_CMD_NAMES[k], v \
            * (-1 if k in INV_JOINTS else 1)) \
            for k, v in self.speed.iteritems()]
        # What is the stiffness?
        stiffness = ['(%ss %.2f)' % (JOINT_CMD_NAMES[k], v) \
            for k, v in self.stiffness.iteritems()]
        return ''.join(speed + stiffness)


    def create_reset_cmd(self, counter):
        """
        Function to test whether the reseteffector works.

        """
        if counter % 6 < 3:
          x, y, rot = (0.0, 0.0, 0.0)
        else:
          x, y, rot = (1.0, 1.0, 1.0)
        return '(reset {:.2f} {:.2f} {:.2f})'\
            .format(x, y, rot)


class GameState:
    def __init__(self):
        self.time = 0
        self.play_mode = 'unknown'
        self.unum = 0
        self.team = 'unknown'

    def update(self, sexp):
        for s in sexp:
            name = s[0]
            if name == 't':
                self.time = float(s[1])
            elif name == 'pm':
                self.play_mode = s[1]
            elif name == 'unum':
                self.unum = int(s[1])
            elif name == 'team':
                self.team = s[1]

