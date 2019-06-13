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
        self.memory_img = []
        self.memory_action = []
        self.memory_reward = []

    def add_to_memory(self,before_action_img_index,reward,action):
        if before_action_img_index is not None:
            self.memory_img.append(before_action_img_index)
        if reward is not None:
            self.memory_action.append(action)
        if action is not None:
            self.memory_reward.append(reward)

    def mem_checker(self):
        global number_images_to_observe
        if len(self.memory_img) < number_images_to_observe:
            return False
        return True

    def fetch_transactions(self,batch_size):
        global number_images_to_observe
        fetched_images_previous  = []
        fetched_images_after = []
        fetched_reward = []
        fetched_action = []
        for i in range(batch_size):
            #print(len(self.memory_img)-number_images_to_observe)
            start_index_pre = random.randint(0,len(self.memory_img)-number_images_to_observe) #start index of previous images
            fetched_p = self.memory_img[start_index_pre:start_index_pre+number_images_to_observe]
            fetched_a = self.memory_img[start_index_pre+1:start_index_pre+number_images_to_observe+1] #next window to calculate Qt+1
            reward = self.memory_reward[start_index_pre+number_images_to_observe-1]
            action = self.memory_action[start_index_pre+number_images_to_observe-1]
            fetched_images_previous.append(fetched_p)
            fetched_images_after.append(fetched_a)
            fetched_reward.append(reward)
            fetched_action.append(action)
        return fetched_images_previous,fetched_images_after,fetched_reward,fetched_action


        

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
        rpl_mem = replay_memory()
        counter = 0
        enemy_lists = []
        for i in range(12):
            speed = random.randint(16,32)
            ene = enemy(width,height,screen,speed)
            enemy_lists.append(ene)
        player = player_agent(screen,width,height,rpl_mem)
        rect = [0,0,0,0]
        reward = Rewards(enemy_lists,rect)

        while 1:
            screen.fill(0)
            for i in range(len(enemy_lists)):
                enemy_lists[i].run_single(rect)
            n_act = player.run_single(rect)
            if n_act > 0:
                r = reward.getReward()
                player.set_reward(r)
                player.training()
            pygame.display.flip()
            
               



def main():
    ge = GUI_engine(1000,800)
    
if __name__ == '__main__':
    main()
