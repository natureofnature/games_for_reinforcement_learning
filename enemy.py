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
import queue
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

    def isConflicted(self,rect):
        pos_0,pos_1 = self.pos
        top,left,bottom,right = rect 
        if (pos_0 <= right and pos_0 >=left) and (pos_1<=bottom and pos_1>=top):
            return True
        else:
            return False

    def run(self):
        t_fetcher = threading.Thread(target=self.run_single) 
        t_fetcher.daemon = True 
        t_fetcher.start()


 
