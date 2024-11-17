# mind-pilot
This is the Github repo for the natHACKS 2024 project Mind Pilot.

Mind Pilot is a rehabilitative software designed to empower people with reduced mobility in their arms and hands. People across the world struggle with reduced mobility due to partial paralyzation, amputation, parkinsons, arthritis, pain, or injury. These people deserve the same access to technology as everyone else, to use for their work, education, and recreation. 

Mind Pilot consists of two key components. The first component of Mind Pilot uses Python's Computer Vision package OpenCV, for facial recognition which references the position and direction the user's nose is pointed in to map a position on screen for the cursor to move to. The second component uses Reinforcement Learning for Motor Imagery Classification. The EEG signals of the eyebrows being raised are detected by a trained RL agent, and interpreted to produce a click with the mouse. We trained the Reinforcement Learning agent in real time using Hecatron, providing it with reward when it was able to correctly predict the action to click when receiving EEG signals of the eyebrows being raised.
