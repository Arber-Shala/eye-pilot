import hecatron
from pynput.mouse import Button, Controller

mouse = Controller()

def func():
    pass

def click():
    mouse.click(Button.left)

board = hecatron.init_board(None, 38)
hecatron.run_live_session(board, action_functions=[func, click], num_actions=2, reference_channels=[0], filename="test_run_4")