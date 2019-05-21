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
hit_counter = 0
class enemy:
    def __init__(self,width,height,screen,speed=1):
        img = Image.open("pictures/star.png")
        img = img.convert("RGBA")
        img = img.resize((10,10))
        mode = img.mode
        size = img.size
        data=img.tobytes()
        self.bomber = pygame.image.fromstring(data,size,mode)
        self.pos = (random.randint(0,width),random.randint(0,height))
        self.screen = screen
        self.angle = math.pi/3 
        self.speed = speed 
        self.width = width
        self.height = height


    def run_single(self,rect):
            pos_0,pos_1 = self.pos
            top,left,bottom,right = rect 
            if (pos_0 <= right and pos_0 >=left) and (pos_1<=bottom and pos_1>=top):
                global hit_counter
                hit_counter = hit_counter + 1
                #print("Hit %d" %(hit_counter))
                #exit(0)
            if pos_0 >= self.width or pos_0 <=0:
                self.angle = - self.angle
            if pos_1 >=self.height or pos_1 <=0:
                self.angle = math.pi-self.angle
            pos_0 = self.pos[0]+self.speed*math.sin(self.angle)
            pos_1 = self.pos[1]+self.speed*math.cos(self.angle)
            self.pos = (pos_0,pos_1)
            self.screen.blit(self.bomber,(self.pos[0],self.pos[1]))

            

    def run(self):
        t_fetcher = threading.Thread(target=self.run_single) 
        t_fetcher.daemon = True 
        t_fetcher.start()


class player_agent:
    def __init__(self,screen,env_w,env_h):
        self.screen = screen
        self.width,self.height = env_w,env_h 
        self.obj_w, self.obj_h = (100,50)
        self.obj_speed=1
        img = Image.open("pictures/car.png").convert("RGBA").resize((self.obj_w,self.obj_h))
        mode = img.mode
        size = img.size
        data=img.tobytes()
        self.obj= pygame.image.fromstring(data,size,mode)
        self.keys = [False, False, False, False]
        self.playerpos=[int(self.width/2),int(self.height/2)]

    def run_single(self,rect):
        self.screen.blit(self.obj,(self.playerpos[0],self.playerpos[1]))
        rect[0] = self.playerpos[1]
        rect[1] = self.playerpos[0]
        rect[2] = self.playerpos[1]+self.obj_h
        rect[3] = self.playerpos[0]+self.obj_w
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
             

class Policy:
    def __init__(self,screen,width,height):
        self.screen = screen
        self.width,self.height = width,height
        self.number_shots= 5
        self.nn = Network(batchsize=1,channelsize=self.number_shots,width=self.width,height=self.height)
        self.nn.create_network()
    def capture(self):
        counter = 0
        image_lists=[]
        while True:
            if counter % 3 == 0:
                image_string = pygame.image.tostring(self.screen,"RGB")
                pil_obj = Image.frombytes("RGB",(self.width,self.height),image_string)
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
                    act1,act2 = self.nn.forward(array)
                    print(act1,act2)
                counter = counter + 1
            counter = counter + 1
            #time.sleep(0.02)

    def run(self):
        t_fetcher = threading.Thread(target=self.capture) 
        t_fetcher.daemon = True 
        t_fetcher.start()




class GUI_engine:
    def __init__(self,width,height):
        self.width,self.height = width,height
        screen = pygame.display.set_mode((width,height))
        counter = 0
        enemy_lists = []
        pol = Policy(screen,width,height)
        pol.run()
        for i in range(12):
            speed = random.randint(1,2)
            ene = enemy(width,height,screen,speed)
            enemy_lists.append(ene)
        player = player_agent(screen,width,height)
        rect = [0,0,0,0]

        while 1:
            screen.fill(0)
            player.run_single(rect)
            for i in range(len(enemy_lists)):
                enemy_lists[i].run_single(rect)
            pygame.display.flip()
            
               



def main():
    ge = GUI_engine(1000,800)
    
if __name__ == '__main__':
    main()
