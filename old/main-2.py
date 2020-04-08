"""

omnipo - you are the omnipotent being

"""
import pygame
import random
import math
import numpy as np



class Actor(pygame.sprite.Sprite):
    def __init__(self,world_shape=(500,500),view_distance=10):
        super(Actor,self).__init__()
        self.surf = pygame.Surface((15,15))
        self.surf.fill((255,0,255))
        self.rect = self.surf.get_rect()
        
        self.view_distance = view_distance        
        self.world_shape = world_shape
        self.pos = (random.randint(0,self.world_shape[0]),random.randint(0,self.world_shape[1]))
        
        
        

def generate_random_actors(n_actors=5,max_height=500,max_width=500,max_depth=500,max_size=15,min_size=5,max_speed=5,max_health=100,max_angle=360):
    dist = np.random.uniform(size=(n_actors,))        
    
    X = np.random.uniform(0,max_width,n_actors)
    Y = np.random.uniform(0,max_height,n_actors)
    Z = np.random.uniform(0,max_depth,n_actors)
    
    sizes = (dist * (max_size-min_size)) + min_size     
    # speeds = np.abs(1. - dist) * max_speed
    speeds = dist * max_speed
    healths = dist * max_health
    
    r,g,b = np.random.randint(0,255,n_actors),np.random.randint(0,255,n_actors),np.random.randint(0,255,n_actors)
    
    dist = np.random.uniform(size=(n_actors,))
      
    angle = dist * max_angle
    
    actors = np.array([X,Y,Z,sizes,speeds,healths,r,g,b,angle]).T

    return actors

def get_xy(x,y,speed,degrees):
    rads = (math.pi/180)*degrees
    
    x += (speed*math.cos(rads))
    y += (speed*math.sin(rads))
    return x,y
        

def main():
    # https://realpython.com/pygame-a-primer/
    # Simple pygame program

    # Import and initialize the pygame library
    pygame.init()
    width = 750
    height = 500
    depth = 500
    
    # Set up the drawing window
    screen = pygame.display.set_mode([width, height])
    
    
    
    n_actors = 50
    max_size = 15.
    max_speed = 1.
    max_health = 1.
    max_depth = 5.
    max_angle = 360.
    
    x_idx = 0
    y_idx = 1
    z_idx = 2
    
    angle_idx = 9
    
    # xv_idx = 3
    # yv_idx = 4
    # zv_idx = 5
    
    size_idx = 3
    speed_idx = 4
    health_idx = 5
    
    r_idx = 6
    g_idx = 7
    b_idx = 8
    
    
    # actor = xpos,ypos,size,view_distance,speed,health
    actors = generate_random_actors(n_actors=25,max_height=height,max_width=width,max_size=max_size,max_depth=depth,max_angle=max_angle)

    # Run until the user asks to quit
    running = True
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))
        
        for actor in actors:
            # surf = pygame.Surface((int(actor[size_idx]),int(actor[size_idx])))
            # surf.set_alpha(actor[health_idx])
            
            # color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            color = (int(actor[r_idx]),int(actor[g_idx]),int(actor[b_idx]))
            pos = (int(actor[x_idx]),int(actor[y_idx]))
            size = int(actor[size_idx])
            pygame.draw.circle(screen,color,pos,size,0)
            
        # actors[:,y_idx] += actors[:,size_idx] * max_speed
        # move actors
        actors[:,x_idx] += actors[:,speed_idx]*np.cos((np.pi/180.)*actors[:,angle_idx])
        actors[:,y_idx] += actors[:,speed_idx]*np.sin((np.pi/180.)*actors[:,angle_idx])
        
        # if offscreen
        actors_idx = np.argwhere(np.logical_or.reduce((actors[:,x_idx]<=0.-actors[:,size_idx],actors[:,x_idx]>=width+actors[:,size_idx],actors[:,y_idx]<=0.-actors[:,size_idx],actors[:,y_idx]>=height+actors[:,size_idx]))).T
        
        if actors_idx.shape[0]:
            actors[actors_idx,:] = generate_random_actors(n_actors=actors_idx.shape[0],max_height=height,max_width=width,max_speed=max_speed,max_size=max_size,max_depth=max_depth)
            actors = actors[actors[:,size_idx].argsort()[::-1]]
        

        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()