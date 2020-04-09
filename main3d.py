"""
omnipo - you are the omnipotent being
Author: Michael Bridges

Description: 200 lines of python and numpy to create a cool, fast visualization in pygame.
"""
import pygame
import random
import math
import numpy as np
import time


WIDTH = 500
HEIGHT = 500
DEPTH = 500

BG = {'white':(255,255,255),
      'black':(0,0,0),
      'gray':(5,5,5),
      }
        

def generate_random_particles(n_particles=5,max_height=HEIGHT,max_width=WIDTH,max_depth=DEPTH,max_velocity=5.,min_velocity=.1,max_mass=10.,min_mass=1.,max_size=10.,min_size=4.):     
    dist = np.random.uniform(size=(n_particles,))
    
    mass = (dist * (max_mass-min_mass)) + min_mass
    radius = (dist * (max_size-min_size)) + min_size 
        
    AX = np.zeros((n_particles,))
    AY = np.zeros((n_particles,))
    AZ = np.zeros((n_particles,))
    
    VX = np.ones((n_particles,)) / 2
    VY = np.ones((n_particles,)) / 2
    VZ = np.ones((n_particles,)) / 2
    
    X = np.random.uniform(size=(n_particles,)) * max_width
    Y = np.random.uniform(size=(n_particles,)) * max_height
    Z = np.random.uniform(size=(n_particles,)) * max_depth
    
            
    red = np.random.randint(100,245,n_particles)
    green = np.random.randint(100,245,n_particles)
    blue = np.random.randint(100,245,n_particles)
    
    particles = np.array([X,Y,Z,VX,VY,VZ,AX,AY,AZ,red,green,blue,mass,radius]).T

    return particles
       

def check_forces(particles,x_idx,y_idx,z_idx,ax_idx,ay_idx,az_idx,mass_idx):
    G = 6.67e-5
    # G = 1.
    for i in range(particles.shape[0]):
        F = np.array([0.,0.,0.])
        for j in range(i+1,particles.shape[0]):
            F[0] += G * particles[i,mass_idx] * particles[j,mass_idx] / (particles[i,x_idx] - particles[j,x_idx])**2 * np.sign((particles[j,x_idx] - particles[i,x_idx]))
            F[1] += G * particles[i,mass_idx] * particles[j,mass_idx] / (particles[i,y_idx] - particles[j,y_idx])**2 * np.sign((particles[j,y_idx] - particles[i,y_idx]))
            F[2] += G * particles[i,mass_idx] * particles[j,mass_idx] / (particles[i,z_idx] - particles[j,z_idx])**2 * np.sign((particles[j,z_idx] - particles[i,z_idx]))
        if np.sum(F):
            particles[i,ax_idx] = F[0] / particles[i,mass_idx]
            particles[i,ay_idx] = F[1] / particles[i,mass_idx]
            particles[i,az_idx] = F[2] / particles[i,mass_idx]
            print(F)
            
    return particles

def move_particles_wrap_force(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,ax_idx,ay_idx,az_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size,step=1.):
    # move particles by adding velocity
    particles[:,vx_idx] += particles[:,ax_idx] * step
    particles[:,vy_idx] += particles[:,ay_idx] * step
    # particles[:,vz_idx] += particles[:,az_idx] * step
    
    particles[:,x_idx] += particles[:,vx_idx] * step
    particles[:,y_idx] += particles[:,vy_idx] * step
    # particles[:,z_idx] += particles[:,vz_idx] * step
    
    # if offscreen
    particles_idx = np.argwhere(np.logical_or.reduce((particles[:,x_idx]<=0.-particles[:,size_idx],
                                                      particles[:,x_idx]>=width+particles[:,size_idx],
                                                      particles[:,y_idx]<=0.-particles[:,size_idx],
                                                      particles[:,y_idx]>=height+particles[:,size_idx],
                                                    #   particles[:,z_idx]<=0.-particles[:,size_idx],
                                                    #   particles[:,z_idx]>=depth+particles[:,size_idx],
                                                      ))).T[0]
    
    
    # particles[:,x_idx] %= width
    # particles[:,y_idx] %= height
    # particles[:,z_idx] %= depth
        
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

def inelastic_collision(particles,pairs,vx_idx,vy_idx,vz_idx,mass_idx):
    for i,j in pairs:
        # i,j = pair[0],pair[1]
        
        m1m2 = (particles[i,mass_idx] + particles[j,mass_idx])
        
        particles[i,vx_idx] =   particles[i,vx_idx] * ((particles[i,mass_idx] - particles[j,mass_idx]) / m1m2) + \
                                particles[j,vx_idx] * (2 * particles[j,mass_idx] / m1m2)
                                
        particles[j,vx_idx] =   particles[i,vx_idx] * (2 * particles[i,mass_idx] / m1m2) + \
                                particles[j,vx_idx] * ((particles[j,mass_idx] - particles[i,mass_idx]) / m1m2)
        
        particles[i,vy_idx] =   particles[i,vy_idx] * ((particles[i,mass_idx] - particles[j,mass_idx]) / m1m2) + \
                                particles[j,vy_idx] * (2 * particles[j,mass_idx] / m1m2)
        
        particles[j,vy_idx] =   particles[i,vy_idx] * (2 * particles[i,mass_idx] / m1m2) + \
                                particles[j,vy_idx] * ((particles[j,mass_idx] - particles[i,mass_idx]) / m1m2)
                                
        particles[i,vz_idx] =   particles[i,vz_idx] * ((particles[i,mass_idx] - particles[j,mass_idx]) / m1m2) + \
                                particles[j,vz_idx] * (2 * particles[j,mass_idx] / m1m2)
        
        particles[j,vz_idx] =   particles[i,vz_idx] * (2 * particles[i,mass_idx] / m1m2) + \
                                particles[j,vz_idx] * ((particles[j,mass_idx] - particles[i,mass_idx]) / m1m2) 
    
    return particles


def elastic_collision(particles,pairs,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,mass_idx):
    for i,j in pairs:

        particles[i,vx_idx] = particles[i,vx_idx] - (2 * particles[j,mass_idx] / (particles[i,mass_idx] + particles[j,mass_idx])) \
                                                    * np.dot((particles[i,vx_idx] - particles[j,vx_idx]),(particles[i,x_idx] - particles[j,x_idx])) \
                                                    / (np.abs(particles[i,x_idx] - particles[j,x_idx]) ** 2) * (particles[i,x_idx] - particles[j,x_idx])
        
        particles[j,vx_idx] = particles[j,vx_idx] - (2 * particles[i,mass_idx] / (particles[i,mass_idx] + particles[j,mass_idx])) \
                                                    * np.dot((particles[j,vx_idx] - particles[i,vx_idx]),(particles[j,x_idx] - particles[i,x_idx])) \
                                                    / (np.abs(particles[j,x_idx] - particles[i,x_idx]) ** 2) * (particles[j,x_idx] - particles[i,x_idx])
                                                    
        particles[i,vy_idx] = particles[i,vy_idx] - (2 * particles[j,mass_idx] / (particles[i,mass_idx] + particles[j,mass_idx])) \
                                                    * np.dot((particles[i,vy_idx] - particles[j,vy_idx]),(particles[i,y_idx] - particles[j,y_idx])) \
                                                    / (np.abs(particles[i,y_idx] - particles[j,y_idx]) ** 2) * (particles[i,y_idx] - particles[j,y_idx])
        
        particles[j,vy_idx] = particles[j,vy_idx] - (2 * particles[i,mass_idx] / (particles[i,mass_idx] + particles[j,mass_idx])) \
                                                    * np.dot((particles[j,vy_idx] - particles[i,vy_idx]),(particles[j,y_idx] - particles[i,y_idx])) \
                                                    / (np.abs(particles[j,y_idx] - particles[i,y_idx]) ** 2) * (particles[j,y_idx] - particles[i,y_idx])

    
    return particles                         

def check_walls(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth):
    left_right = np.argwhere(np.logical_or.reduce((particles[:,x_idx]<0.+particles[:,size_idx],particles[:,x_idx]>width-particles[:,size_idx])))
    top_bottom = np.argwhere(np.logical_or.reduce((particles[:,y_idx]<0.+particles[:,size_idx],particles[:,y_idx]>height-particles[:,size_idx])))
    front_back = np.argwhere(np.logical_or.reduce((particles[:,z_idx]<0.+particles[:,size_idx],particles[:,z_idx]>depth-particles[:,size_idx])))

    particles[left_right,vx_idx] *= -1.
    particles[top_bottom,vy_idx] *= -1.
    particles[front_back,vz_idx] *= -1.
    
    return particles


def check_collisions(particles,x_idx,y_idx,z_idx,size_idx):
    pairs = []
    for i in range(particles.shape[0]):
        for j in range(i+1,particles.shape[0]):
            # dist = np.linalg.norm( np.array([particles[i,x_idx],particles[i,y_idx],particles[i,z_idx]]) - np.array([particles[j,x_idx],particles[j,y_idx],particles[j,z_idx]]) )
            dist = np.linalg.norm(np.array([particles[i,x_idx],particles[i,y_idx]]) - np.array([particles[j,x_idx],particles[j,y_idx]]))
            overlap = particles[i,size_idx] + particles[j,size_idx]
            if dist < overlap:
                pairs.append([i,j])
    return np.array(pairs)

def merge_collisions(particles,pairs,x_idx,y_idx,z_idx,mass_idx,radius_idx):
    new_particles = 0
    for pair in pairs:
        i,j = pair[0],pair[1]
        new_particle = np.mean(np.array([particles[i],particles[j]]),axis=0)
            
def draw_collisions(screen,particles,pairs,r_idx,g_idx,b_idx,x_idx,y_idx):
    # draw a line between two collided particles
    color = (255,255,255)
    thickness = 2 
    if pairs.shape[0]:
        for pair in pairs:
            i,j = pair[0],pair[1]
            pygame.draw.line(screen,color,(particles[i,x_idx],particles[i,y_idx]),(particles[j,x_idx],particles[j,y_idx]),thickness)

def draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,size_idx,depth):
    # draw particles with darkening gradient history tail
    for particle in particles:
        color = (int(particle[r_idx]),int(particle[g_idx]),int(particle[b_idx]))
        pos = (int(particle[x_idx]),int(particle[y_idx]))
        # size = int((particle[size_idx] * (1. - ((particle[z_idx])/(depth+particle[size_idx])))))
        size = int(particle[size_idx])
        if size < 0.:
            size = 0
        pygame.draw.circle(screen,color,pos,size,0)


def draw_particles_history(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,max_depth,trail_size=3,trail_stride=1,trail_gradient=0.5):
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
    n_particles = int(10)
    
    max_velocity=3.
    min_velocity=.5
    
    max_mass=40.
    min_mass=20.
    
    max_size=15.
    min_size=5.
    
    x_idx = 0
    y_idx = 1
    z_idx = 2

    vx_idx = 3
    vy_idx = 4
    vz_idx = 5
    
    ax_idx = 6
    ay_idx = 7
    az_idx = 8
    
    r_idx = 9
    g_idx = 10
    b_idx = 11
    
    mass_idx = 12
    size_idx = 13
    
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
        draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,size_idx,depth)
        
        # identify collisions
        pairs = check_collisions(particles,x_idx,y_idx,z_idx,size_idx)
        
        draw_collisions(screen,particles,pairs,r_idx,g_idx,b_idx,x_idx,y_idx)
        
        particles = elastic_collision(particles,pairs,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,mass_idx)
        
        particles = check_walls(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth)
        
        # calculate forces
        particles = check_forces(particles,x_idx,y_idx,z_idx,ax_idx,ay_idx,az_idx,mass_idx)
            
        # move the particles
        particles = move_particles_wrap_force(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,ax_idx,ay_idx,az_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size)
        
        time.sleep(.01)
        # print(particles[:,3:6])
        # print(particles.shape)
        
        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()