"""
omnipo - you are the omnipotent being
Author: Michael Bridges

Description: 200 lines of python and numpy to create a cool, fast visualization in pygame.
"""
import pygame
import random
import math
import numpy as np


WIDTH = 1080
HEIGHT = 720
DEPTH = 500

BG = {'white':(255,255,255),
      'black':(0,0,0),
      'gray':(5,5,5),
      }
        

def generate_random_particles(n_particles=5,max_height=HEIGHT,max_width=WIDTH,max_depth=DEPTH,max_velocity=5.,min_velocity=.1,max_mass=10.,min_mass=1.,max_size=10.,min_size=2.):     
    dist = np.random.uniform(size=(n_particles,)) 
    mass = (dist * (max_mass-min_mass)) + min_mass
    size = (dist * (max_size-min_size)) + min_size 
    
    # VX = np.random.uniform(size=(n_particles,)) * max_velocity  * np.random.choice(np.array([1.,-1.]),n_particles)
    # VY = np.random.normal(size=(n_particles,)) * max_velocity * np.random.choice(np.array([1.,-1.]),n_particles)
    # VZ = np.random.uniform(size=(n_particles,)) * max_velocity * -1.

    VX = np.random.normal(loc=0.0,scale=3.0,size=(n_particles,)) * max_velocity  * np.random.choice(np.array([1.,-1.]),n_particles)
    VY = np.random.normal(size=(n_particles,)) * max_velocity * np.random.choice(np.array([1.,-1.]),n_particles)
    VZ = np.random.uniform(size=(n_particles,)) * max_velocity * -1.
    
    X = np.ones((n_particles,)) / 2. * max_width
    Y = np.ones((n_particles,)) / 2. * max_height
    Z = np.ones((n_particles,)) * max_depth
            
    red = np.random.randint(200,245,n_particles)
    green = np.random.randint(200,245,n_particles)
    blue = np.random.randint(200,245,n_particles)
    
    particles = np.array([X,Y,Z,VX,VY,VZ,red,green,blue,mass,size]).T

    return particles
       
       
def move_particles_random(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size,step=1):
    # move particles by adding velocity
    particles[:,x_idx] += particles[:,vx_idx] * step
    particles[:,y_idx] += particles[:,vy_idx] * step
    particles[:,z_idx] += particles[:,vz_idx]**1.2 * step
    
    # if offscreen
    particles_idx = np.argwhere(np.logical_or.reduce((particles[:,x_idx]<=0.-particles[:,size_idx],
                                                      particles[:,x_idx]>=width+particles[:,size_idx],
                                                      particles[:,y_idx]<=0.-particles[:,size_idx],
                                                      particles[:,y_idx]>=height+particles[:,size_idx],
                                                      particles[:,z_idx]<=0.-particles[:,size_idx],
                                                      particles[:,z_idx]>=depth+particles[:,size_idx],
                                                      ))).T[0]
    # replace offscreen particles with new particles
    if particles_idx.shape[0]:
        particles[particles_idx,:] = generate_random_particles(n_particles=particles_idx.shape[0],max_height=height,max_width=width,max_depth=depth,max_velocity=max_velocity,min_velocity=min_velocity,max_mass=max_mass,min_mass=min_mass,max_size=max_size,min_size=min_size)
        particles = particles[particles[:,vz_idx].argsort()[::-1]]
        
    return particles

def move_particles_wrap(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth):
    # wrap particles around the screen
    particles[:,x_idx] += particles[:,vx_idx]
    particles[:,y_idx] += particles[:,vy_idx]
    particles[:,z_idx] += particles[:,vz_idx]
    
    particles[:,x_idx] %= width
    particles[:,y_idx] %= height
    particles[:,z_idx] %= depth
        
    return particles


def check_collisions(particles,x_idx,y_idx,z_idx,size_idx):
    # collisions = np.zeros((particles.shape[0],particles.shape[0]))
    pairs = []
    for i in range(particles.shape[0]):
        for j in range(i+1,particles.shape[0]):
            dist = np.linalg.norm(np.array([particles[i,x_idx],particles[i,y_idx],particles[i,z_idx]]) - np.array([particles[j,x_idx],particles[j,y_idx],particles[j,z_idx]]),axis=-1)
            overlap = particles[i,size_idx] + particles[j,size_idx]
            if dist < overlap:
                pairs.append([i,j])
    return np.array(pairs)
            
def draw_collisions(screen,particles,pairs,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx):
    # draw a line between two collided particles
    color = (0,0,0)
    thickness = 2 
    for pair in pairs:
        i,j = pair[0],pair[1]
        pygame.draw.line(screen,color,(particles[i,x_idx],particles[i,y_idx]),(particles[j,x_idx],particles[j,y_idx]),thickness)


def draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,max_depth,trail_size=3,trail_stride=1,trail_gradient=0.5):
    # draw particles with darkening gradient history tail
    for particle in particles:
        # compute color value ratios for history tail
        start,stop,step = (trail_gradient,1.,trail_size)
        trail = np.linspace(start,stop,step)[::-1]
        
        # if particle is moving away, render light then dark, else dark then light    
        if particle[vz_idx] < 0.:
            enum = reversed(list(enumerate(trail)))
        else:
            enum = enumerate(trail)

        # draw each particle on the screen with a history tail, (helpful for showing direction and giving perspective)
        for i,ratio in enum:
            color = (int(particle[r_idx] * ratio),int(particle[g_idx] * ratio),int(particle[b_idx] * ratio))
            pos = (int(particle[x_idx] - particle[vx_idx] * i * trail_stride),int(particle[y_idx] - particle[vy_idx] * i * trail_stride))
            size = int((particle[size_idx] * (1. - ((particle[z_idx] - particle[vz_idx] * i * trail_stride)/(max_depth+particle[size_idx])))))
            if size < 0.:
                size = 0
            pygame.draw.circle(screen,color,pos,size,0)

def main():
    # https://realpython.com/pygame-a-primer/
    # Simple pygame program
    
    width = WIDTH
    height = HEIGHT
    depth = DEPTH

    # Import and initialize the pygame library
    pygame.init()
    
    # Set up the drawing window
    screen = pygame.display.set_mode([width, height])
    
    # hyperparameters, these can be changed in all sorts of fun ways.
    n_particles = int(1e3)
    
    max_velocity=3.
    min_velocity=.5
    
    max_mass=10.
    min_mass=1.
    
    max_size=15.
    min_size=5.
    
    x_idx = 0
    y_idx = 1
    z_idx = 2

    vx_idx = 3
    vy_idx = 4
    vz_idx = 5

    r_idx = 6
    g_idx = 7
    b_idx = 8
    
    mass_idx = 9
    size_idx = 10
    
    # history_size = 3
    
    # initialize particles
    particles = generate_random_particles(n_particles=n_particles,max_height=height,max_width=width,max_depth=depth,max_velocity=max_velocity,min_velocity=min_velocity,max_mass=max_mass,min_mass=min_mass,max_size=max_size,min_size=min_size)
    # particles_history = np.array([particles] * history_size)
    
    # Run until the user asks to quit
    running = True
    while running:       

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background color
        screen.fill(BG['gray'])

        # draw the particles
        draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,depth)
            
        # move the particles
        particles = move_particles_random(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size)
        # particles_history = np.concatenate((np.array([particles]),particles_history),axis=0)[:history_size]
        
        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()