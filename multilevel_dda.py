import pygame
import os
import math
import numpy as np
import pygame.gfxdraw

# -------------------------------------
# Setup
# -------------------------------------

# screen resolution
window_width, window_height = 1000, 1000

# initialize pygame
#pygame.init()

# configs
#pygame.display.set_caption("(DDA) Digital Differential Analyser")

# initialise window: high resolution display
#window = pygame.display.set_mode((window_width, window_height))

# colours
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
PINK = (255, 192, 203)

def event_handler(o_x, o_y, r_x, r_y, grid_pixel_x, grid_pixel_y):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()

    # get held keys
    keys = pygame.key.get_pressed()

    # update origin
    if keys[pygame.K_w]:
        o_y -= 1 / grid_pixel_y
    if keys[pygame.K_a]:
        o_x -= 1 / grid_pixel_x
    if keys[pygame.K_s]:
        o_y += 1 / grid_pixel_y
    if keys[pygame.K_d]:
        o_x += 1 / grid_pixel_x

    # get mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

    # compute direction
    r_x = (mouse_x / grid_pixel_x) - o_x
    r_y = (mouse_y / grid_pixel_y) - o_y

    # normalize
    r_norm = np.sqrt(r_x**2 + r_y**2)
    r_x = r_x / r_norm
    r_y = r_y / r_norm

    return o_x, o_y, r_x, r_y


def draw_grid(window, grid, grid_max_x, grid_max_y, grid_pixel_x, grid_pixel_y):
    for x in range(grid_max_x):
        for y in range(grid_max_y):
            # white border around each square
            rect = pygame.Rect(x * grid_pixel_x, y * grid_pixel_y, grid_pixel_x, grid_pixel_y)
            pygame.draw.rect(window, WHITE, rect, 1)
            # green square for filled squares
            if grid[y, x] == 1:
                rect = pygame.Rect(x * grid_pixel_x + 1, y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
                pygame.draw.rect(window, GREEN, rect, 0)

def draw_dot(window, grid_x, grid_y, grid_pixel_x, grid_pixel_y):
    pygame.draw.circle(window, PINK, (grid_x * grid_pixel_x, grid_y * grid_pixel_y), grid_pixel_x // 10, 0)

# -------------------------------------
# Multilevel Grid
# -------------------------------------

def multilevel_grid(grid, n, L):
    '''
    grid: base grid (level 0)
    n: size of grid squares to compress per level
    L: maximum level

    NOTE: need grid size | n**L
    '''

    # grid size
    M, N = grid.shape

    # check sizes match
    if (M % (n ** L) != 0) or (N % (n ** L)) != 0:
        raise Exception("Invalid grid scaling for size")

    # store grids in level order
    grid_list = [grid]

    # store current grid
    current_grid = grid

    # for each level
    for l in range(1, L + 1):

        # get current grid size
        M_current, N_current = current_grid.shape

        # compute higher grid size
        M_higher, N_higher = M_current // n, N_current // n

        # construct higher grid
        higher_grid = np.empty((M_higher, N_higher))

        # loop over larger grid
        for i in range(M_higher):
            for j in range(N_higher):

                # current grid index
                i_current = i * n
                j_current = j * n

                # slice current grid
                grid_square = current_grid[i_current:(i_current+n), j_current:(j_current+n)]

                # compute max
                entry = int(np.max(grid_square))

                # fill higher grid
                higher_grid[i, j] = entry

        # store grid
        grid_list.append(higher_grid)

        # update current grid
        current_grid = higher_grid

    return grid_list

# -------------------------------------
# DDA
# -------------------------------------

def DDA(window, grid, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y):

    # normalize direction
    r_norm = np.sqrt(r_x**2 + r_y**2)
    r_x = r_x / r_norm
    r_y = r_y / r_norm

    # initial grid index
    grid_x = math.floor(o_x)
    grid_y = math.floor(o_y)

    # t distance between each axis
    if r_x == 0:
        dt_x = np.inf
    else:
        dt_x = 1 / abs(r_x)
    if r_y == 0:
        dt_y = np.inf
    else:
        dt_y = 1 / abs(r_y)

    # initial t distance
    if r_x < 0:
        step_x = -1
        t_x = (o_x - grid_x) * dt_x
    else:
        step_x = 1
        t_x = (grid_x + 1 - o_x) * dt_x

    if r_y < 0:
        step_y = -1
        t_y = (o_y - grid_y) * dt_y
    else:
        step_y = 1
        t_y = (grid_y + 1 - o_y) * dt_y

    # cast ray until intersection
    intersection = False
    intersection_value = None
    t_max = 1000.0
    t = 0.0

    # check initial square
    if grid[grid_y, grid_x] != 0:

        intersection = True

        # draw red square for intersection
        rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
        pygame.draw.rect(window, RED, rect, 0)

    else:

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

    # draw dot at starting point
    draw_dot(window, o_x, o_y, grid_pixel_x, grid_pixel_y)

    while ((not intersection) and (t < t_max)):

        # select axis with smaller current t distance
        if t_x < t_y:

            # step in x axis to new grid sqaure
            grid_x += step_x

            # store as current t distance (NOTE: before incrementing t_x, as need check the square we are entering)
            t = t_x

            # increment t_x
            t_x += dt_x

        else:

            # repeat for y axis
            grid_y += step_y
            t = t_y
            t_y += dt_y

        # check if intersection with new grid square
        if (grid_x >= 0) and (grid_x < grid_max_x) and (grid_y >= 0) and (grid_y < grid_max_y):

            intersection_value = grid[grid_y, grid_x]
            
            # check if empty
            if intersection_value != 0:
                intersection = True

                # draw red square for intersection
                rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
                pygame.draw.rect(window, RED, rect, 0)

            else:

                # draw blue square for no intersection
                rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
                pygame.draw.rect(window, BLUE, rect, 0)

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

        # outside of grid
        else:

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

    # draw line from start to intersection
    pygame.draw.line(window, PINK, (o_x * grid_pixel_x, o_y * grid_pixel_y), ((o_x + t * r_x) * grid_pixel_x, (o_y + t * r_y) * grid_pixel_y), width=2)

# -------------------------------------
# Main
# -------------------------------------

grid = np.array([
    [1, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
])

print("A")

grid_list = multilevel_grid(grid, 2, 2)

print(grid_list)
print("A")

'''

# line
o_x, o_y = 1.4, 1.1
r_x, r_y = -1.1, 2.3

# grid
grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])

# grid size
grid_max_y, grid_max_x = grid.shape

# random grid
tau = 0.8
grid_max_y, grid_max_x = 20, 20
rng = np.random.default_rng()
grid = rng.uniform(0, 1, (grid_max_y, grid_max_x))
grid[grid < tau] = 0
grid[grid >= tau] = 1

# pixel size of grid squares
grid_pixel_x = window_width // grid_max_x
grid_pixel_y = window_height // grid_max_y

while True:

    window.fill(BLACK)

    # events
    o_x, o_y, r_x, r_y = event_handler(o_x, o_y, r_x, r_y, grid_pixel_x, grid_pixel_y)

    # clamp
    eps = 0.1
    if o_x < 0:
        o_x = 0
    if o_y < 0:
        o_y = 0
    if o_x >= grid_max_x:
        o_x = grid_max_x - eps
    if o_y >= grid_max_y:
        o_y = grid_max_y - eps

    draw_grid(window, grid, grid_max_x, grid_max_y, grid_pixel_x, grid_pixel_y)

    DDA(window, grid, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y)

    # update canvas
    pygame.display.flip()

'''