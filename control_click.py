from real_time_learning import hecatron
from pynput.mouse import Controller, Button

def click():
    mouse = Controller()
    mouse.click(Button.left)

def func():
    pass

def click2():
    mouse = Controller()
    mouse.press(Button.left)

def func2():
    mouse = Controller()
    mouse.release(Button.left)

board = hecatron.init_board(None, 38)
hecatron.run_live_session(board, action_functions=[func, click], num_actions=2, reference_channels=[0], filename='/Users/icansingh/Desktop/Work/Projects/Nathacks/eye-pilot/real_time_learning/best_model_yay')