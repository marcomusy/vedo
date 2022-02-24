"""Solve a random maze with
Markovian Decision Process"""
# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier & Anthony Strock
# Released under a BSD two-clauses license
#
# References: Bellman, Richard (1957), A Markovian Decision Process.
#             Journal of Mathematics and Mechanics. Vol. 6, No. 5.
# -----------------------------------------------------------------------------
#https://github.com/rougier/ML-Recipes/blob/master/recipes/MDP/value-iteration.py
#https://en.wikipedia.org/wiki/Markov_decision_process
import numpy as np
from scipy.ndimage import generic_filter

def maze(shape=(30, 50), complexity=0.8, density=0.8):
    shape = (np.array(shape)//2)*2 + 1
    n_complexity = int(complexity*(shape[0]+shape[1]))
    n_density = int(density*(shape[0]*shape[1]))
    Z = np.ones(shape, dtype=bool)
    Z[1:-1, 1:-1] = 0
    P = (np.dstack([np.random.randint(0, shape[0]+1, n_density),
                    np.random.randint(0, shape[1]+1, n_density)])//2)*2
    for (y,x) in P.squeeze():
        Z[y, x] = 1
        for j in range(n_complexity):
            neighbours = []
            if x > 1:           neighbours.append([(y, x-1), (y, x-2)])
            if x < shape[1]-2:  neighbours.append([(y, x+1), (y, x+2)])
            if y > 1:           neighbours.append([(y-1, x), (y-2, x)])
            if y < shape[0]-2:  neighbours.append([(y+1, x), (y+2, x)])
            if len(neighbours):
                next_1, next_2 = neighbours[np.random.randint(len(neighbours))]
                if Z[next_2] == 0:
                    Z[next_1] = Z[next_2] = 1
                    y, x = next_2
            else:
                break
    return Z

def solve(Z, start, goal):
    Z = 1 - Z
    G = np.zeros(Z.shape)
    G[start] = 1

    # We iterate until value at exit is > 0. This requires the maze
    # to have a solution or it will be stuck in the loop.
    def diffuse(Z, gamma=0.99):
        return max(gamma*Z[0], gamma*Z[1], Z[2], gamma*Z[3], gamma*Z[4])

    while G[goal] == 0.0:
        G = Z * generic_filter(G, diffuse, footprint=[[0, 1, 0],
                                                      [1, 1, 1],
                                                      [0, 1, 0]])

    # Descent gradient to find shortest path from entrance to exit
    y, x = goal
    dirs = (0,-1), (0,+1), (-1,0), (+1,0)
    P = []
    while (x, y) != start:
        P.append((y,x))
        neighbours = [-1, -1, -1, -1]
        if x > 0:            neighbours[0] = G[y, x-1]
        if x < G.shape[1]-1: neighbours[1] = G[y, x+1]
        if y > 0:            neighbours[2] = G[y-1, x]
        if y < G.shape[0]-1: neighbours[3] = G[y+1, x]
        a = np.argmax(neighbours)
        x, y  = x + dirs[a][1], y + dirs[a][0]
    P.append((y,x))

    return P, G

def printSolution(S, start, goal):
    for y,line in enumerate(Z):
        for x,c in enumerate(line):
            if   (y,x) == start: print("[]", end='')
            elif (y,x) == goal:  print("[]", end='')
            elif (y,x) in S[0]:  print("..", end='')
            elif c:              print("██", end='')
            else:                print("  ", end='')
        print()

def showSolution3D(S, start, goal):
    from vedo import Text3D, Cube, Line, Grid, merge, show

    pts, cubes, txts = [], [], []
    pts = [(x,-y) for y,x in S[0]]
    for y,line in enumerate(Z):
        for x,c in enumerate(line):
            if c: cubes.append(Cube([x,-y,0]))

    path = Line(pts).lw(6).c('red5')
    walls = merge(cubes).flat().c('orange1')

    sy, sx = S[1].shape
    gradient = np.flip(S[1], axis=0).ravel()
    grd = Grid(pos=((sx-1)/2, -(sy-1)/2, -0.49), s=[sx,sy], res=[sx,sy])
    grd.lw(0).wireframe(False).cmap('gist_earth_r', gradient, on='cells')
    grd.addScalarBar(title='Gradient', horizontal=True, c='k', nlabels=2)

    txts.append(__doc__)
    txts.append(Text3D('Start', pos=[start[1]-1,-start[0]+1.5,1], c='k'))
    txts.append(Text3D('Goal!', pos=[goal[1] -2,-goal[0] -2.7,1], c='k'))
    return show(path, walls, grd, txts, axes=0, zoom=1.2)


##########################################################################
if __name__ == '__main__':
    np.random.seed(4)

    Z = maze(shape=(50, 70))

    start, goal = (1,1), (Z.shape[0]-2, Z.shape[1]-2)
    print("Please wait..")
    S = solve(Z, start, goal)

    #printSolution(S, start, goal)
    showSolution3D(S, start, goal).close()
