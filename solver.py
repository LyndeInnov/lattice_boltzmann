import numpy as np
from matplotlib import pyplot

plot_every = 30

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def main(): 

    # number of lattice cells in the x direction 
    Nx = 400 
    # number of lattice cells in the y direction 
    Ny = 200
    # kinematic velocity or timscale 
    tau = 0.53
    # number of time iterations to run
    Nt = 5000

    # Lattice speeds and weights setup 
    # Number of lattice nodes 
    NL = 9
    # x discrete velocities from lattice center 
    cxs = np.array([0,0,1,1, 1, 0,-1,-1,-1])
    #y discrete velocities from lattice center
    cys = np.array([0,1,1,0,-1,-1,-1, 0, 1])
    #setting up node weights 
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    #Initial conditions 
    #now with those set up, we need to set up our 3 dimensional initial condition velocity array
    # this will be a 3 dimensional array for x position, y position, and velocity at each node. 
    F = np.ones((Ny, Nx, NL)) + 0.01 * np.random.randn(Ny, Nx, NL)

    # give each cell a rightwards velocity of 2.3. 
    F[:,:,3] = 2.3

    #Set up an obstical 

    cylinder = np.full((Ny, Nx), False)
    cyl_rad = 13

    for y in range (Ny):
        for x in range (Nx):
            if (distance(Nx//4, Ny//2, x, y)<cyl_rad):
                cylinder[y][x] = True

    # Main Loop 

    for it in range (Nt):
        print(it)

        F[:, -1, [6, 7, 8 ]] = F[:, -2, [6, 7, 8 ]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:,:,i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:,:,i], cy, axis = 0)

        #Setting all velocities at obsticle boundary  or inside of obsticle to be opposite
        
        bdryF = F[cylinder, :]
        bdryF = bdryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Fluid variables 
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2)/rho
        uy = np.sum(F * cys, 2)/rho

        #Set all velocities inside the obsticle to be zero 

        F[cylinder, :] = bdryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        #Collisions 

        #Calculate Fequillibrium 
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            Feq[:,:,i] = rho * w * (
                1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2
            )
        F = F + -(1/tau) * (F-Feq)

        if(it%plot_every==0):
            pyplot.imshow(np.sqrt(ux**2 + uy**2))
            pyplot.pause(.01)
            pyplot.cla()

if __name__=="__main__":
    main()
