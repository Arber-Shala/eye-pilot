import hecatron
import mouse

board = hecatron.init_board(None, 38)
hecatron.run_live_session(board, action_functions=[None, mouse.click('left')], num_actions=2, reference_channels=[0], filename="test_run_4")