import pygame
import time
import math
import sys
from pygame.locals import * 
from PIL import Image
import threading
import random
hit_counter = 0
class enemy:
    def __init__(self,width,height,screen):
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
        self.speed = 1 
        self.width = width
        self.height = height


    def run_single(self,rect):
            pos_0,pos_1 = self.pos
            top,left,bottom,right = rect 
            if (pos_0 <= right and pos_0 >=left) and (pos_1<=bottom and pos_1>=top):
                global hit_counter
                hit_counter = hit_counter + 1
                print("Hit %d" %(hit_counter))
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




class GUI_engine:
    def __init__(self,width,height):
        self.width,self.height = width,height
        self.obj_w, self.obj_h = (100,50)
        self.obj_speed=1
        screen = pygame.display.set_mode((width,height))
        img = Image.open("pictures/car.png").convert("RGBA").resize((self.obj_w,self.obj_h))
        mode = img.mode
        size = img.size
        data=img.tobytes()
        player = pygame.image.fromstring(data,size,mode)
        keys = [False, False, False, False]
        playerpos=[500,500]
        counter = 0
        enemy_lists = []
        for i in range(5):
            ene = enemy(width,height,screen)
            enemy_lists.append(ene)

        while 1:
            screen.fill(0)
            screen.blit(player,(playerpos[0],playerpos[1]))
            rect = (playerpos[1],playerpos[0],playerpos[1]+self.obj_h,playerpos[0]+self.obj_w) 
            for i in range(len(enemy_lists)):
                enemy_lists[i].run_single(rect)

            pygame.display.flip()
            time.sleep(0.001)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key==K_w:
                        keys[0]=True
                    elif event.key==K_a:
                        keys[1]=True
                    elif event.key==K_s:
                        keys[2]=True
                    elif event.key==K_d:
                        keys[3]=True
                if event.type == pygame.KEYUP:
                    if event.key==pygame.K_w:
                        keys[0]=False
                    elif event.key==pygame.K_a:
                        keys[1]=False
                    elif event.key==pygame.K_s:
                        keys[2]=False
                    elif event.key==pygame.K_d:
                        keys[3]=False
            if keys[0]:
                playerpos[1] = max(0,playerpos[1]-self.obj_speed) 
            elif keys[2]:
                playerpos[1] = min(self.height-self.obj_h,playerpos[1]+self.obj_speed)
            if keys[1]:
                playerpos[0] = max(0,playerpos[0]-self.obj_speed)
            elif keys[3]:
                playerpos[0] = min(self.width-self.obj_w,playerpos[0]+self.obj_speed)
                



def main():
    ge = GUI_engine(900,800)
    
if __name__ == '__main__':
    main()
