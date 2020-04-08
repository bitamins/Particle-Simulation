"""

omnipo - you are the omnipotent being

"""
import pygame
import random
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
        
        
        

def generate_random_actors(n_actors=5,max_height=500,max_width=500,max_size=25,max_speed=5,max_health=100):
    dist = np.random.uniform(size=(n_actors,))
    sizes = dist * max_speed
    speeds = np.abs(1. - dist) * max_size
    healths = dist * max_health
    return np.array([np.random.uniform(0,max_width,n_actors),np.random.randint(0,max_height,n_actors),sizes,speeds,healths]).T
        
    

def main():
    # https://realpython.com/pygame-a-primer/
    # Simple pygame program

    # Import and initialize the pygame library
    pygame.init()

    
    width = 500
    height = 500
    n_actors = 25
    min_size = 5
    max_size = 25
    min_speed = 1
    max_speed = 5
    min_health = 20
    max_health = 100
    
    


    
    sizes = np.random.randint(min_size,max_size,n_actors)
    speeds = (((sizes - min_size) * (max_speed - min_speed)) / (max_size - min_size)) + min_speed
    healths = (((sizes - min_size) * (max_health - min_health)) / (max_size - min_size)) + min_health
    
    # actor = xpos,ypos,size,view_distance,speed,health
    actors = np.array([np.random.randint(0,width,n_actors),np.random.randint(0,height,n_actors),sizes,speeds,healths]).T
    
    actors = generate_random_actors(n_actors=25)

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
            pygame.draw.circle(screen,(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)),(int(actor[0]),int(actor[1])),int(actor[2]),0)
            
        # actors[:,0] += 1 
        actors[:,1] += 0.2
        
        actors_idx = np.argwhere(np.logical_or.reduce((actors[:,0]<0,actors[:,0]>width,actors[:,1]<0,actors[:,1]>height)))
        n_new_actors = actors_idx.shape[0]
        
        if n_new_actors:
            sizes = np.random.randint(min_size,max_size,n_new_actors) 
            speeds = (((sizes - min_size) * (max_speed - min_speed)) / (max_size - min_size)) + min_speed
            healths = (((sizes - min_size) * (max_health - min_health)) / (max_size - min_size)) + min_health
            
            # actor = xpos,ypos,size,view_distance,speed,health
            new_actors = np.array([np.random.randint(0,width,n_new_actors),0,sizes,speeds,healths]).T
            
            print(actors_idx)
            # actors = np.put(actors,actors_idx,new_actors,0)
            actors[actors_idx,:] = new_actors
        
        
        

        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()