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
from datetime import datetime
random.seed(datetime.now())
number_images_to_observe = 4

class Policy:
    def __init__(self,screen,width,height,policy_queue,epsilon,replay_mem):
        self.replay_mem = replay_mem
        self.screen = screen
        self.width,self.height = width,height
        self.epsilon = epsilon
        global number_images_to_observe
        self.number_shots= number_images_to_observe 
        self.policy_queue = policy_queue
        self.batch_size = 20 
        self.nn = Network(batchsize=self.batch_size,channelsize=self.number_shots,width=self.width,height=self.height)
        self.rewarwd = 0
        self.screen_shot_index = 0
        self.image_lists = [] #sequential images
        self.image_history = [] #if file is larger, relace it with files
        self.training_counter = 0

    def set_reward(self,reward):
        self.reward = reward
        self.replay_mem.add_to_memory(None,self.reward,None)

        #screen shots
        #image_string = pygame.image.tostring(self.screen,"RGB")
        #pil_obj = Image.frombytes("RGB",(self.width,self.height),image_string)



    def act(self): #record image,action in Xt
        #while acting/playing, batchsize = 1
        #screen shots
        image_string = pygame.image.tostring(self.screen,"RGB")
        pil_obj = Image.frombytes("RGB",(self.width,self.height),image_string)
        pil_obj = pil_obj.convert('L')
        img_array = np.array(pil_obj)
        img_array = np.transpose(img_array,(1,0))
        img_array = np.expand_dims(img_array,0)
        self.image_lists.append(img_array)
        self.image_history.append(img_array)

        if self.screen_shot_index %1 == 0:
            pass
            #saving to disk
            #pil_obj.save(os.path.join("/dev/shm/1",str(self.screen_shot_index)+".jpg"))
        self.screen_shot_index+=1 
        if len(self.replay_mem.memory_img) < self.number_shots:
            self.replay_mem.add_to_memory(self.screen_shot_index,0,[1,0,0,0,0]) #do not move 
            return []
        else:
            del(self.image_lists[0])
            self.replay_mem.add_to_memory(self.screen_shot_index,None,None)

        #random explore
        sampled = random.randint(0,100)
        act = [] 
        if self.training_counter < 10000:
            self.epsilon = 0.3
        else:
            self.epsilon = 0.05
        self.training_counter+=1
        if self.training_counter % 1000 == 0:
            print("Trained %d iterations \n" %(self.training_counter))
        if sampled/100 <= self.epsilon:
            act = np.zeros(5)
            act[random.randint(0,4)] = 1
            self.replay_mem.add_to_memory(None,None,act)
        else:
            array = np.concatenate(self.image_lists,axis=0)
            array = np.expand_dims(array,0)
            act = self.nn.forward(array)[0]
            #while acting/playing, batchsize = 1
            self.replay_mem.add_to_memory(None,None,act[0])
            #self.policy_queue.put((act1,act2))
            #self.policy_queue.put((act1,act2))
        return act
    def img_index_to_np(self,image_list):
        #to do
        #array = np.concatenate(image_list,axis = 0)
        #array = np.expand_dims(array,0)
        array = np.vstack(image_list)
        return array


    
    def training(self):
        fetch_images_pre,fetch_images_aft,fetched_reward, fetched_action = self.replay_mem.fetch_transactions(self.batch_size)
        if len(fetch_images_pre) == 0:
            return
        image_pre = []
        image_aft = []
        for i in fetch_images_pre:
            image_slice = np.concatenate(self.image_history[i[0]:i[-1]+1],axis=0)
            if len(image_slice) != self.number_shots:
                print("Return")
                return
            image_slice = np.expand_dims(image_slice,0) 
            image_pre.append(image_slice)
        for i in fetch_images_aft:
            image_slice = np.concatenate(self.image_history[i[0]:i[-1]+1],axis=0)
            if len(image_slice) != self.number_shots:
                print("Return")
                return
            image_slice = np.expand_dims(image_slice,0) 
            image_aft.append(image_slice)
        reward = np.array(fetched_reward)
        reward = np.expand_dims(reward,-1)
        action = np.array(fetched_action,dtype=np.float32)
        action = np.expand_dims(action,-1)
        image_pre = self.img_index_to_np(image_pre)
        image_aft = self.img_index_to_np(image_aft)
        gamma = 0.1
        self.nn.cal_loss(image_pre,image_aft,reward,action,gamma)







class player_agent:
    def __init__(self,screen,env_w,env_h,rpl_mem):
        self.replay_mem = rpl_mem
        self.screen = screen
        self.width,self.height = env_w,env_h 
        self.obj_w, self.obj_h = (50,25)
        self.obj_speed=20
        img = Image.open("pictures/car.png").convert("RGBA").resize((self.obj_w,self.obj_h))
        mode = img.mode
        size = img.size
        data=img.tobytes()
        self.obj= pygame.image.fromstring(data,size,mode)
        self.keys = [False, False, False, False]
        self.playerpos=[int(self.width/2),int(self.height/2)]
        self.policy_queue = queue.Queue(1)
        self.reward = 0
        #explore 
        self.epsilon = 0.1 
        self.pol = Policy(screen,self.width,self.height,self.policy_queue,self.epsilon,self.replay_mem)
    def training(self):
        self.pol.training()

    def run_single(self,rect):
        self.screen.blit(self.obj,(self.playerpos[0],self.playerpos[1]))
        rect[0] = self.playerpos[1]
        rect[1] = self.playerpos[0]
        rect[2] = self.playerpos[1]+self.obj_h
        rect[3] = self.playerpos[0]+self.obj_w
        acts = self.pol.act()
        if len(acts) == 0:
            return 0
        act = np.argmax(acts)
        #print(dirc,press)
        if act == 1: #up
            self.playerpos[1] = max(0,self.playerpos[1]-self.obj_speed) 
            #print("up")
        elif act == 2: #down
            self.playerpos[1] = min(self.height-self.obj_h,self.playerpos[1]+self.obj_speed)
            #print("Down")
        elif act == 3: #left
            self.playerpos[0] = max(0,self.playerpos[0]-self.obj_speed)
            #print("Left")
        elif act == 4: #right
            self.playerpos[0] = min(self.width-self.obj_w,self.playerpos[0]+self.obj_speed)
            #print("right")

       
        #############uncomment if you want to control it with keyboards###########################
        ''' 
        #buttons
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key==K_w:
                    self.keys[0]=True
                elif event.key==K_a:
                    self.keys[1]=True
                elif event.key==K_s:
                    self.keys[2]=True
                elif event.key==K_d:
                    self.keys[3]=True
            if event.type == pygame.KEYUP:
                if event.key==pygame.K_w:
                    self.keys[0]=False
                elif event.key==pygame.K_a:
                    self.keys[1]=False
                elif event.key==pygame.K_s:
                    self.keys[2]=False
                elif event.key==pygame.K_d:
                    self.keys[3]=False
        if self.keys[0]:
            self.playerpos[1] = max(0,self.playerpos[1]-self.obj_speed) 
        elif self.keys[2]:
            self.playerpos[1] = min(self.height-self.obj_h,self.playerpos[1]+self.obj_speed)
        if self.keys[1]:
            self.playerpos[0] = max(0,self.playerpos[0]-self.obj_speed)
        elif self.keys[3]:
            self.playerpos[0] = min(self.width-self.obj_w,self.playerpos[0]+self.obj_speed)
        '''
        ##########################################################################################
        return len(acts)

    def set_reward(self,reward):
        self.pol.set_reward(reward)

             


