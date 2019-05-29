import pygame
import os
import time
import math
import sys
from pygame.locals import * 
from PIL import Image
import threading
import io
import random
import numpy as np
from Tensorflow_backend import NN as Network
import queue
from enemy import enemy
from player_agent import Policy, player_agent, number_images_to_observe
         

class replay_memory:
    def __init__(self,path_to_store='/dev/shm/1',max_image_number=100000):
        #(current_index,current_reward,current_action)
        self.memory = []

    def add_to_memory(self,generated_img_index,reward,action):
        self.memory.append((generated_img_index,reward,action))

    def fetch_transactions(self,batch_size):
        global number_images_to_observe



        


class Rewards:
    def __init__(self, enemies=[],player_agent_rect = None):
        self.enemies = enemies
        self.player_agent_rect = player_agent_rect
    def getReward(self):
        for ene in self.enemies:
            isCon = ene.isConflicted(self.player_agent_rect)
            if isCon is True:
                return -100000
        return 1
            


class GUI_engine:
    def __init__(self,width,height):
        self.width,self.height = width,height
        screen = pygame.display.set_mode((width,height))
        counter = 0
        enemy_lists = []
        for i in range(12):
            speed = random.randint(16,32)
            ene = enemy(width,height,screen,speed)
            enemy_lists.append(ene)
        player = player_agent(screen,width,height)
        rect = [0,0,0,0]
        reward = Rewards(enemy_lists,rect)

        while 1:
            screen.fill(0)
            for i in range(len(enemy_lists)):
                enemy_lists[i].run_single(rect)
            player.run_single(rect)
            r = reward.getReward()
            print(r)
            pygame.display.flip()
            
               



def main():
    ge = GUI_engine(1000,800)
    
if __name__ == '__main__':
    main()
