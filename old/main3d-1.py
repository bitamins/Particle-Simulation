"""

omnipo - you are the omnipotent being

"""
import pygame
import random
import math
import numpy as np


WIDTH = 750
HEIGHT = 500
DEPTH = 100
        
        

def generate_random_particles(n_particles=5,max_height=HEIGHT,max_width=WIDTH,max_depth=DEPTH,max_velocity=5.,min_velocity=.1,max_mass=10.,min_mass=1.,max_size=10.,min_size=2.):     
    dist = np.random.uniform(size=(n_particles,)) 
    mass = (dist * (max_mass-min_mass)) + min_mass
    size = (dist * (max_size-min_size)) + min_size 
    
    # VX = ((np.random.uniform(size=(n_particles,)) * (max_velocity-min_velocity)) + min_velocity ) * np.random.choice(np.array([1.,-1.]),n_particles)
    # VY = ((np.random.uniform(size=(n_particles,)) * (max_velocity-min_velocity)) + min_velocity ) * np.random.choice(np.array([1.,-1.]),n_particles)
    # VZ = ((np.random.uniform(size=(n_particles,)) * (max_velocity-min_velocity)) + min_velocity ) * np.random.choice(np.array([1.,-1.]),n_particles)
    
    VX = np.random.uniform(size=(n_particles,)) * max_velocity  * np.random.choice(np.array([1.,-1.]),n_particles)
    VY = np.random.uniform(size=(n_particles,)) * max_velocity  * np.random.choice(np.array([1.,-1.]),n_particles)
    VZ = np.random.uniform(size=(n_particles,)) * max_velocity  * np.random.choice(np.array([1.,-1.]),n_particles)
    
    # VX = np.random.uniform(-max_velocity,max_velocity,n_particles)
    # VY = np.random.uniform(-max_velocity,max_velocity,n_particles)
    # VZ = np.random.uniform(-max_velocity,max_velocity,n_particles)
    
    # X = np.random.uniform(size=(n_particles,)) * max_width
    # Y = np.random.uniform(size=(n_particles,)) * max_height
    # Z = np.random.uniform(size=(n_particles,)) * max_depth
    
    # X = np.random.normal(loc=max_width/2,scale=20,size=(n_particles,)) * max_width
    # Y = np.random.normal(loc=max_height/2,scale=20,size=(n_particles,)) * max_height
    # Z = np.random.normal(loc=max_depth/2,scale=20,size=(n_particles,)) * max_depth
    
    # X = np.where(VX > 0.,0.,1.) * max_width
    # Y = np.where(VY > 0.,0.,1.) * max_height
    # Z = np.where(VZ > 0.,0.,1.) * max_depth
    
    X = np.ones((n_particles,)) / 2. * max_width
    Y = np.ones((n_particles,)) / 2. * max_height
    Z = np.ones((n_particles,)) / 2. * max_depth
    
    # X = np.random.uniform(0,max_width,n_particles)
    # Y = np.random.uniform(0,max_height,n_particles)
    # Z = np.random.uniform(0,max_depth,n_particles)
            
    red = np.random.randint(200,255,n_particles)
    green = np.random.randint(200,255,n_particles)
    blue = np.random.randint(200,255,n_particles)
    
    particles = np.array([X,Y,Z,VX,VY,VZ,red,green,blue,mass,size]).T

    return particles
       
       
def move_particles_random(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size):
    particles[:,x_idx] += particles[:,vx_idx]
    particles[:,y_idx] += particles[:,vy_idx]
    particles[:,z_idx] += particles[:,vz_idx]
    
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
        particles = particles[particles[:,size_idx].argsort()[::-1]]
        
    return particles

def move_particles_wrap(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth):
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
    color = (0,0,0)
    thickness = 2 
    for pair in pairs:
        i,j = pair[0],pair[1]
        pygame.draw.line(screen,color,(particles[i,x_idx],particles[i,y_idx]),(particles[j,x_idx],particles[j,y_idx]),thickness)

def draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,max_depth,trail_size=5,trail_stride=2):
    for particle in particles:
        if particle[vz_idx] < 0.:
            start,stop,step = (trail_size-1,-1,-1)
        elif particle[vz_idx] >= 0.:
            start,stop,step = (0,trail_size,1)
                
        for i in range(start,stop,step):
            color = (int(particle[r_idx] * (1./(i+1))),int(particle[g_idx] * (1./(i+1))),int(particle[b_idx] * (1./(i+1))))
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
    
    n_particles = 150
    
    max_velocity=2.
    min_velocity=.5
    
    max_mass=10.
    min_mass=1.
    
    max_size=15.
    min_size=3.
    
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
    
    # agent = xpos,ypos,size,view_distance,speed,health
    particles = generate_random_particles(n_particles=n_particles,max_height=height,max_width=width,max_depth=depth,max_velocity=max_velocity,min_velocity=min_velocity,max_mass=max_mass,min_mass=min_mass,max_size=max_size,min_size=min_size)

    # Run until the user asks to quit
    running = True
    while running:       

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        draw_particles(screen,particles,r_idx,g_idx,b_idx,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,depth)
            
        particles = move_particles_random(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth,max_velocity,min_velocity,max_mass,min_mass,max_size,min_size)    
        # particles = move_particles_wrap(particles,x_idx,y_idx,z_idx,vx_idx,vy_idx,vz_idx,size_idx,height,width,depth) 
        
        # pairs = check_collisions(particles,x_idx,y_idx,z_idx,size_idx)
        # draw_collisions(screen,particles,pairs,r_idx,g_idx,b_idx,x_idx,y_idx,size_idx)
        
        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()



if __name__ == '__main__':
    main()