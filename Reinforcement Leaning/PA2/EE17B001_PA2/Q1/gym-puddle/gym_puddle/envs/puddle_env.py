from gym import Env
from gym.utils import seeding
from gym import spaces
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

# Class for the Puddle-World Environment

class PuddleWorld(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.world = np.zeros([12,12], dtype=np.int64) # The matrix of rewards obtained on arriving at respective states
        self.actions = [[-1,0], [0,1], [0,-1], [1,0]] # North, East, West, South respectively
                         
        self.action_space = spaces.Discrete(4) # The action space is 4-discrete
        self.observation_space = spaces.Box(low = -3.0, high = 10.0, shape=self.world.shape)

        self.start_states = ((5,0),(6,0),(10,0),(11,0)) # The 4 start states given
        self.reward = 0

		# Setting up the puddle rewards in the puddle-world
        for i in [4,3,2]:
        	self.world[i:11-i,i+1:11-i] -= 1
        for i in [5,6,7]:	
        	self.world[i:i+2,i+1] += 1
    
    # Function to return the goal coordinates and the Westerly wind probabilities    
    def set_goal(self,goal):
        if goal=='A':
            [i,j] = [0,11]
            self.wind = 1
        elif goal=='B':
            [i,j] = [2,9]
            self.wind = 1
        elif goal=='C':
            [i,j] = [6,7]
            self.wind = 0
        self.world[i,j] = 10
        return i,j
	
	# Based on policy's action, this function chooses the final action after applying the world's stochasticity
    def final_act(self, selected_act):
        p_array = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
        p_array[selected_act] = 0.9       
        fin_act = np.random.choice(4, 1, p = p_array).item()
        return fin_act
    
    # Function that is used to select a random action during epsilon-exploration    
    def random_act(self):
    	self.act = np.random.choice(4)
    	return self.act
    	
    # Function to take a step in the environment, given current state and action
    def step(self, current_state, act):
        act = self.final_act(act)
        if self.wind:
            self.push = np.random.choice(2,1).item()
        else:
            self.push = 0
        transient_state = np.array(current_state) + np.array(self.actions[act]) + self.push*np.array([0,1])
        if ((transient_state < 0).any() or (transient_state > 11).any()): # If the agent moves out of the boundaries of the world:
            next_state = current_state
        else:
            next_state = transient_state
        self.reward = self.world[next_state[0], next_state[1]]
        return next_state, self.reward    
	
	# Function to reset the episode at one of the 4 start states
    def reset(self):
        sampler = np.random.choice(4)
        self.state = self.start_states[sampler]
        return np.array(self.state)
    
    # Function to render the gridworld (to verify the implementation of the world only, does not render the agent's trajectories
    def render(self, mode='human', close=False):
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY1 = (175, 175, 175)
        GRAY2 = (100, 100, 100)
        BLUE = (150, 220, 220)
        GREEN = (0, 255, 0)
        
        # This sets the WIDTH and HEIGHT of each world location
        WIDTH = 30
        HEIGHT = 30
        
        # This sets the margin between each cell
        MARGIN = 2
        
        pygame.init()
        screen_side = 12*(MARGIN + WIDTH) + MARGIN
        WINDOW_SIZE = [screen_side, screen_side]
        screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("The Puddle-World")
        done = False
        clock = pygame.time.Clock()
        
        while not done:
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True  # Flag that we are done so we exit this loop
        
            # Set the screen background
            screen.fill(BLACK)
        
            # Draw the world
            for row in range(12):
                for column in range(12):
                    color = WHITE
                    if self.world[row][column] == -1:
                        color = GRAY1
                    elif self.world[row][column] == -2:
                        color = GRAY2
                    elif self.world[row][column] == -3:
                        color = BLACK
                    if (row,column) in self.start_states:
                        color = BLUE
                    if self.world[row][column] == 10:
                        color = GREEN
                    pygame.draw.rect(screen, color, [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT])
        
            # Limit to 60 frames per second
            clock.tick(60)
            pygame.display.flip()
        pygame.quit()
