'''this module collect all the variables involved in the bayesian network and initialise them'''
import enum

class Robot_Assistance_Feedback(enum.Enum):
    lev_0_no = 0
    lev_1_no = 1
    lev_2_no = 2
    lev_3_no = 3
    lev_4_no = 4
    lev_0_yes = 0
    lev_1_yes = 1
    lev_2_yes = 2
    lev_3_yes = 3
    lev_4_yes = 4
    name = "robot_assistance_feedback"
    counter = 10

class User_Capability(enum.Enum):
    very_mild = 0
    mild = 1
    severe = 2
    name = "user_capability"
    counter = 3

class User_Action(enum.Enum):
    correct = 0
    wrong = 1
    timeout = 2
    name = "user_action"
    counter = 3

class Reactivity(enum.Enum):
    slow = 0
    medium = 1
    fast = 2
    name = "reactivity"
    counter = 3

class Memory(enum.Enum):
    low = 0
    medium = 1
    high = 2
    name = "memory"
    counter = 3

class Attention(enum.Enum):
    low = 0
    medium = 1
    high = 2
    name = "attention"
    counter = 3

class Robot_Assistance(enum.Enum):
    lev_0 = 0
    lev_1 = 1
    lev_2 = 2
    lev_3 = 3
    lev_4 = 4
    name = "robot_assistance"
    counter = 5

class Robot_Feedback(enum.Enum):
    yes = 0
    no = 1
    name = "robot_feedback"
    counter = 2

class Game_State(enum.Enum):
    beg = 0
    middle = 1
    end = 2
    name = "game_state"
    counter = 3

class Attempt(enum.Enum):
    at_1 = 0
    at_2 = 1
    at_3 = 2
    at_4 = 3
    name = "attempt"
    counter = 4

#test the module




