"""

omnipo - you are the omnipotent being

"""
import pygame
import random
import math
import numpy as np


WIDTH = 750
HEIGHT = 500
        
        

def generate_random_agents(n_agents=5,max_height=HEIGHT,max_width=WIDTH,max_depth=500,max_size=10,min_size=2,max_speed=5,max_health=100,max_angle=360):
    dist = np.random.uniform(size=(n_agents,))        
    
    X = np.random.uniform(0,max_width,n_agents)
    Y = np.random.uniform(0,max_height,n_agents)
    Z = np.random.uniform(0,max_depth,n_agents)
        
    sizes = (dist * (max_size-min_size)) + min_size     
    speeds = np.abs(1. - dist) * max_speed
    # speeds = dist * max_speed
    healths = dist * max_health
    
    r,g,b = np.random.randint(0,255,n_agents),np.random.randint(0,255,n_agents),np.random.randint(0,255,n_agents)
    
    dist = np.random.uniform(size=(n_agents,))
      
    angle = dist * max_angle
    
    agents = np.array([X,Y,Z,sizes,speeds,healths,r,g,b,angle]).T

    return agents

def get_xy(x,y,speed,degrees):
    rads = (math.pi/180)*degrees
    
    x += (speed*math.cos(rads))
    y += (speed*math.sin(rads))
    return x,y
       
       
def move_agents_random(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_size,max_speed,width=WIDTH,height=HEIGHT):
    agents[:,x_idx] += agents[:,speed_idx]*np.cos((np.pi/180.)*agents[:,angle_idx])
    agents[:,y_idx] += agents[:,speed_idx]*np.sin((np.pi/180.)*agents[:,angle_idx])
    
    # if offscreen
    agents_idx = np.argwhere(np.logical_or.reduce((agents[:,x_idx]<=0.-agents[:,size_idx],agents[:,x_idx]>=width+agents[:,size_idx],agents[:,y_idx]<=0.-agents[:,size_idx],agents[:,y_idx]>=height+agents[:,size_idx]))).T
    
    # replace offscreen agents
    if agents_idx.shape[0]:
        agents[agents_idx,:] = generate_random_agents(n_agents=agents_idx.shape[0],max_height=height,max_width=width,max_speed=max_speed,max_size=max_size)
        agents = agents[agents[:,size_idx].argsort()[::-1]]
        
    return agents


def move_agents_bounce(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_size,max_speed,width=WIDTH,height=HEIGHT):
    agents[:,x_idx] += agents[:,speed_idx]*np.cos((np.pi/180.)*agents[:,angle_idx])
    agents[:,y_idx] += agents[:,speed_idx]*np.sin((np.pi/180.)*agents[:,angle_idx])
    
    # collisions
    agents_idx = np.argwhere(np.logical_or.reduce((agents[:,x_idx]<0.+agents[:,size_idx],agents[:,x_idx]>width-agents[:,size_idx],agents[:,y_idx]<0.+agents[:,size_idx],agents[:,y_idx]>height-agents[:,size_idx]))).T

    if agents_idx.shape[0]:
        agents[agents_idx,angle_idx] = (agents[agents_idx,angle_idx] + (np.where(agents_idx >= 180,-90,90) * 1.)) % 360.
    
    return agents

def move_agents_wrap(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_size,max_speed,width=WIDTH,height=HEIGHT):
    agents[:,x_idx] += agents[:,speed_idx]*np.cos((np.pi/180.)*agents[:,angle_idx])
    agents[:,y_idx] += agents[:,speed_idx]*np.sin((np.pi/180.)*agents[:,angle_idx])
    
    agents[:,x_idx] %= width
    agents[:,y_idx] %= height
        
    return agents

def move_agents_collision(agents,pairs,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_size,max_speed,width=WIDTH,height=HEIGHT):
    for pair in pairs:
        i,j = pair[0],pair[1]
        
        agents[i,angle_idx] = (agents[i,angle_idx] + 180) % 360
        agents[j,angle_idx] = (agents[j,angle_idx] + 180) % 360
         
        # agent[i,speed_idx] =
        # agent[j,speed_idx] = 
        
    return agents
    



def check_collisions(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_size,max_speed,width=WIDTH,height=HEIGHT):
    # collisions = np.zeros((agents.shape[0],agents.shape[0]))
    pairs = []
    for i in range(agents.shape[0]):
        for j in range(i+1,agents.shape[0]):
            dist = np.linalg.norm(np.array([agents[i,x_idx],agents[i,y_idx]]) - np.array([agents[j,x_idx],agents[j,y_idx]]))
            mass = agents[i,size_idx] + agents[j,size_idx]
            if dist < mass:
                pairs.append([i,j])
    return np.array(pairs)
            
def draw_collisions(screen,agents,pairs,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx):
    color = (0,0,0)
    thickness = 3 
    for pair in pairs:
        i,j = pair[0],pair[1]
        pygame.draw.line(screen,color,(agents[i,x_idx],agents[i,y_idx]),(agents[j,x_idx],agents[j,y_idx]),thickness)


def draw_agents(screen,agents,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx):
    for agent in agents:
        color = (int(agent[r_idx]),int(agent[g_idx]),int(agent[b_idx]))
        pos = (int(agent[x_idx]),int(agent[y_idx]))
        size = int(agent[size_idx])
        pygame.draw.circle(screen,color,pos,size,0)
        

def main():
    # https://realpython.com/pygame-a-primer/
    # Simple pygame program

    # Import and initialize the pygame library
    pygame.init()
    width = WIDTH
    height = HEIGHT
    depth = 500
    
    # Set up the drawing window
    screen = pygame.display.set_mode([width, height])
    
    
    
    n_agents = 100
    max_size = 10.
    max_speed = 1.
    max_health = 1.
    max_depth = 5.
    max_angle = 360.
    
    x_idx = 0
    y_idx = 1
    z_idx = 2
    
    angle_idx = 9
    
    xv_idx = 10
    yv_idx = 11
    zv_idx = 12
    
    size_idx = 3
    speed_idx = 4
    health_idx = 5
    
    r_idx = 6
    g_idx = 7
    b_idx = 8
    
    
    # agent = xpos,ypos,size,view_distance,speed,health
    agents = generate_random_agents(n_agents=n_agents,max_height=height,max_width=width,max_size=max_size,max_speed=max_speed,max_depth=depth,max_angle=max_angle)

    # Run until the user asks to quit
    running = True
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        draw_agents(screen,agents,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx)
            
        # agents = move_agents_random(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_speed,max_size,width,height)
        # agents = move_agents_bounce(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_speed,max_size,width,height)
        agents = move_agents_wrap(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_speed,max_size,width,height)
        
        pairs = check_collisions(agents,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_speed,max_size,width,height)
        
        draw_collisions(screen,agents,pairs,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx)
        
        agents = move_agents_collision(agents,pairs,x_idx,y_idx,speed_idx,angle_idx,size_idx,max_speed,max_size,width,height)

        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()