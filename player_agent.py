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
number_images_to_observe = 4

class Policy:
    def __init__(self,screen,width,height,policy_queue):
        self.screen = screen
        self.width,self.height = width,height
        global number_images_to_observe
        self.number_shots= number_images_to_observe 
        self.policy_queue = policy_queue
        self.nn = Network(batchsize=1,channelsize=self.number_shots,width=self.width,height=self.height)
        self.nn.create_network()
    def capture(self):
        counter = 0
        image_lists=[]
        while True:
            image_string = pygame.image.tostring(self.screen,"RGB")
            pil_obj = Image.frombytes("RGB",(self.width,self.height),image_string)
            if counter % 1 == 0:
                pil_obj.save(os.path.join("/dev/shm/1",str(counter)+".jpg"))
            pil_obj = pil_obj.convert('L')
            img_array = np.array(pil_obj)
            img_array = np.transpose(img_array,(1,0))
            img_array = np.expand_dims(img_array,0)
            image_lists.append(img_array)
            if len(image_lists) > self.number_shots:
                del(image_lists[0])
            if len(image_lists) == self.number_shots: 
                array = np.concatenate(image_lists,axis=0)
                array = np.expand_dims(array,0)
                act = self.nn.forward(array)
                #print(act1,act2)
                self.policy_queue.put(act)
                #self.policy_queue.put((act1,act2))
                #self.policy_queue.put((act1,act2))
            counter = counter + 1

    def run(self):
        t_fetcher = threading.Thread(target=self.capture) 
        t_fetcher.daemon = True 
        t_fetcher.start()



class player_agent:
    def __init__(self,screen,env_w,env_h):
        self.screen = screen
        self.width,self.height = env_w,env_h 
        self.obj_w, self.obj_h = (100,50)
        self.obj_speed=20
        img = Image.open("pictures/car.png").convert("RGBA").resize((self.obj_w,self.obj_h))
        mode = img.mode
        size = img.size
        data=img.tobytes()
        self.obj= pygame.image.fromstring(data,size,mode)
        self.keys = [False, False, False, False]
        self.playerpos=[int(self.width/2),int(self.height/2)]
        self.policy_queue = queue.Queue(1)
        pol = Policy(screen,self.width,self.height,self.policy_queue)
        #explore 
        self. epsilon = 0.1
        pol.run()

    def run_single(self,rect):
        self.screen.blit(self.obj,(self.playerpos[0],self.playerpos[1]))
        rect[0] = self.playerpos[1]
        rect[1] = self.playerpos[0]
        rect[2] = self.playerpos[1]+self.obj_h
        rect[3] = self.playerpos[0]+self.obj_w
        acts = self.policy_queue.get()
        act = np.argmax(acts)
        #print(dirc,press)
        if act == 1:
            self.playerpos[1] = max(0,self.playerpos[1]-self.obj_speed) 
        elif act == 2:
            self.playerpos[1] = max(0,self.playerpos[1]-self.obj_speed) 
        elif act == 3:
            self.playerpos[0] = max(0,self.playerpos[0]-self.obj_speed)
        elif act == 4:
            self.playerpos[0] = min(self.width-self.obj_w,self.playerpos[0]+self.obj_speed)
       
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
             


