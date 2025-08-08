import pygame
import os
import math
import numpy as np
import pygame.gfxdraw
from copy import copy

# -------------------------------------
# Setup
# -------------------------------------

# screen resolution
window_width, window_height = 1200, 1200

# initialize pygame
pygame.init()

# configs
pygame.display.set_caption("(DDA) Digital Differential Analyser")

# initialise window: high resolution display
window = pygame.display.set_mode((window_width, window_height))

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

def get_index(x, y, l, n):
    '''
    Find index of position (x, y) in grid level l (scaling of n)
    '''

    scale = n ** l
    i = math.floor(y / scale)
    j = math.floor(x / scale)

    return i, j

def get_level(x, y, l, n, L, grid_list):
    '''
    Find the highest level where the position (x, y) is in an empty square
    Return the level and index
    
    (x, y): position
    l: current level
    n: grid scaling
    L: max level
    grid_list: grid levels
    '''

    # get index in current grid
    i, j = get_index(x, y, l, n)

    # get current grid entry
    try:
        current = grid_list[l][i, j]
    except IndexError:
        # outside of grid
        return i, j, l

    # empty
    if current == 0:

        # increase level until next full (or reach L)
        while l < L:

            # get index in level l + 1
            i_higher, j_higher = get_index(x, y, l + 1, n)

            # get entry
            higher = grid_list[l + 1][i_higher, j_higher]

            # empty
            if higher == 0:

                # increase level
                l += 1

                i = i_higher
                j = j_higher

            # full
            else:

                # found highest empty level
                break

    # full
    else:

        # lowest level full
        if l == 0:

            return i, j, -1

        # decrease level until empty
        while l > 0:

            # decrease level
            l -= 1

            # get index in level
            i, j = get_index(x, y, l, n)

            # get entry
            lower = grid_list[l][i, j]

            # empty
            if lower == 0:

                # found highest empty level
                break

            # full
            else:

                # lowest level full
                if l == 0:

                    l = -1

    # return level and corresponding index
    return i, j, l

def DDA(window, grid_list, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y, n, L, printing=False):

    # cast ray until intersection
    intersection = False
    t_max = 1000.0
    t = 0.0

    # normalize direction
    r_norm = np.sqrt(r_x**2 + r_y**2)
    r_x = r_x / r_norm
    r_y = r_y / r_norm

    # account for zero divide in 1 / r_x & 1 / r_y
    if r_x == 0:
        s_x = np.inf
    else:
        s_x = 1 / abs(r_x)
    if r_y == 0:
        s_y = np.inf
    else:
        s_y = 1 / abs(r_y)

    # start at highest level (??? or lowest l = 0 might be faster)
    l = L

    # find highest empty level and intial grid index
    grid_y, grid_x, l = get_level(o_x, o_y, l, n, L, grid_list)

    # cellsize (level l)
    s = n ** l

    # check initial square
    if l == -1:

        # flag intersection
        intersection = True

        # default cellsize
        s = 1

        # draw red square for intersection
        rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
        pygame.draw.rect(window, RED, rect, 0)

    else:

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, s * grid_pixel_x - 2, s * grid_pixel_y - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

    # t distance between each axis (level l)
    dt_x = s * s_x
    dt_y = s * s_y

    # initial t distance (level l)
    if r_x < 0:
        step_x = -1
        t_x = (o_x - grid_x * s) * s_x
    else:
        step_x = 1
        t_x = ((grid_x + 1) * s - o_x) * s_x

    if r_y < 0:
        step_y = -1
        t_y = (o_y - grid_y * s) * s_y
    else:
        step_y = 1
        t_y = ((grid_y + 1) * s - o_y) * s_y

    # flag initial distance (if axis has not moved beyond initial distance = True)
    initial_x = True
    initial_y = True

    # draw dot at starting point
    draw_dot(window, o_x, o_y, grid_pixel_x, grid_pixel_y)

    if printing:
        print(f"Start")
        print(f"(x, y): ({o_x}, {o_y})")
        print(f"(i, j): ({grid_y}, {grid_x})")
        print(f"l: {l}")
        print(f"t {t}")
        print(f"t_x {t_x}")
        print(f"t_y {t_y}")
        print("")

    while ((not intersection) and (t < t_max)):

        # DDA step at level l

        # record axis to be incremented
        axis_x = True

        # select axis with smaller current t distance
        if t_x < t_y:

            # step in x axis to new grid sqaure
            #grid_x += step_x

            # store as current t distance (NOTE: before incrementing t_x, as need check the square we are entering)
            t = t_x

            # increment t_x
            # t_x += dt_x
            axis_x = True
            initial_x = False

        else:

            # repeat for y axis
            #grid_y += step_y
            t = t_y
            #t_y += dt_y
            axis_x = False
            initial_y = False

        # get current position
        x = o_x + t * r_x
        y = o_y + t * r_y

        # adjust??
        if axis_x:
            x += step_x / 10**3
        else:
            y += step_y / 10**3

        # shift towards next square (avoid sitting on gridline???)
        # x += 
        # y += step_y / 2

        # find highest empty level and grid index
        grid_y, grid_x, l_new = get_level(x, y, l, n, L, grid_list)

        if printing:
            print(f"(x, y): ({x}, {y})")
            print(f"(i, j): ({grid_y}, {grid_x})")
            print(f"l: {l_new}")
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

        # get shape of new grid
        grid_max_y, grid_max_x = grid_list[l_new].shape

        # intersection guaranteed in grid
        if l_new == -1:
            grid_max_x, grid_max_y = np.inf, np.inf

        # check if intersection with new grid square
        if (grid_x >= 0) and (grid_x < grid_max_x) and (grid_y >= 0) and (grid_y < grid_max_y):
            
            # no empty level: intersection
            if l_new == -1:

                intersection = True

                # draw red square for intersection
                rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
                pygame.draw.rect(window, RED, rect, 0)

            # found empty level: no intersection
            else:

                # adjust scale
                s = n ** l_new

                # t distance between each axis (level l)
                dt_x = s * s_x
                dt_y = s * s_y

                # if still at initial t distance in axis: need to update
                # BUT: always doing this causes some issues: maybe when next up to be incremented???
                
                if initial_x:
                    # initial index (level l)
                    o_grid_y, o_grid_x = get_index(o_x, o_y, l_new, n)
                    # initial t distance (level l)
                    if r_x < 0:
                        t_x = (o_x - o_grid_x * s) * s_x
                    else:
                        t_x = ((o_grid_x + 1) * s - o_x) * s_x
                if initial_y:
                    # initial index (level l)
                    o_grid_y, o_grid_x = get_index(o_x, o_y, l_new, n)
                    # initial t distance (level l)
                    if r_y < 0:
                        t_y = (o_y - o_grid_y * s) * s_y
                    else:
                        t_y = ((o_grid_y + 1) * s - o_y) * s_y

                # draw blue square for no intersection
                rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, grid_pixel_x * s - 2, grid_pixel_y * s - 2)
                pygame.draw.rect(window, BLUE, rect, 0)

                # increment t's by adjusted amounts
                if axis_x:
                    t_x += dt_x
                else:
                    t_y += dt_y

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

        # outside of grid
        else:

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

        # update level
        l = l_new

        if printing:
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

    # draw line from start to intersection
    pygame.draw.line(window, PINK, (o_x * grid_pixel_x, o_y * grid_pixel_y), ((o_x + t * r_x) * grid_pixel_x, (o_y + t * r_y) * grid_pixel_y), width=2)


def compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l):

    # get index of origin (level l)
    grid_y, grid_x = get_index(o_x, o_y, l, n)

    # get scale
    s = n ** l

    # initial t distance (level l)
    if r_x < 0:
        step_x = -1
        t_x = (o_x - grid_x * s) * s_x
    else:
        step_x = 1
        t_x = ((grid_x + 1) * s - o_x) * s_x

    if r_y < 0:
        step_y = -1
        t_y = (o_y - grid_y * s) * s_y
    else:
        step_y = 1
        t_y = ((grid_y + 1) * s - o_y) * s_y

    return t_x, t_y, step_x, step_y

def compute_dt(s_x, s_y, n, l):

    # get scale
    s = n ** l

    # t distance between each axis (level l)
    dt_x = s * s_x
    dt_y = s * s_y

    return dt_x, dt_y

def new_DDA(window, grid_list, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y, n, L, printing=False):
    '''
    Try to fix issues with wrong t values when changing level
    '''

    # cast ray until intersection
    intersection = False
    t_max = 1000.0
    t = 0.0

    # normalize direction
    r_norm = np.sqrt(r_x**2 + r_y**2)
    r_x = r_x / r_norm
    r_y = r_y / r_norm

    # axis scaling: account for zero divide in 1 / r_x & 1 / r_y
    if r_x == 0:
        s_x = np.inf
    else:
        s_x = 1 / abs(r_x)
    if r_y == 0:
        s_y = np.inf
    else:
        s_y = 1 / abs(r_y)

    # start at highest level (??? or lowest l = 0 might be faster)
    l = L

    # find highest empty level and intial grid index
    grid_y, grid_x, l = get_level(o_x, o_y, l, n, L, grid_list)

    # cellsize (level l)
    s = n ** l

    # check initial square
    if l == -1:

        # flag intersection
        intersection = True

        # default cellsize
        s = 1

        # draw red square for intersection
        rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
        pygame.draw.rect(window, RED, rect, 0)

    else:

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, s * grid_pixel_x - 2, s * grid_pixel_y - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

    # t distance between each axis (level l)
    dt_x, dt_y = compute_dt(s_x, s_y, n, l)

    # initial t distance (level l)
    t_x, t_y, step_x, step_y = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l) 

    # flag initial distance (if axis has not moved beyond initial distance = True)
    initial_x = True
    initial_y = True

    # draw dot at starting point
    draw_dot(window, o_x, o_y, grid_pixel_x, grid_pixel_y)

    if printing:
        print(f"Start")
        print(f"(x, y): ({o_x}, {o_y})")
        print(f"(i, j): ({grid_y}, {grid_x})")
        print(f"l: {l}")
        print(f"t {t}")
        print(f"t_x {t_x}")
        print(f"t_y {t_y}")
        print("")

    while ((not intersection) and (t < t_max)):

        # DDA step at level l

        # select axis with smaller current t distance
        if t_x < t_y:

            # step in x axis to new grid sqaure
            #grid_x += step_x

            # store as current t distance (NOTE: before incrementing t_x, as need check the square we are entering)
            t = t_x

            # increment t_x
            # t_x += dt_x
            axis_x = True
            initial_x = False

        else:

            # repeat for y axis
            #grid_y += step_y
            t = t_y
            #t_y += dt_y
            axis_x = False
            initial_y = False

        # get current position
        x = o_x + t * r_x
        y = o_y + t * r_y

        # shift towards next square (avoid sitting on gridline???)
        if axis_x:
            x += step_x / 10**3
        else:
            y += step_y / 10**3

        # find highest empty level and grid index
        grid_y, grid_x, l_new = get_level(x, y, l, n, L, grid_list)

        if printing:
            print(f"(x, y): ({x}, {y})")
            print(f"(i, j): ({grid_y}, {grid_x})")
            print(f"l: {l_new}")
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

        # no empty level: intersection
        if l_new == -1:

            intersection = True

            # draw red square for intersection
            rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
            pygame.draw.rect(window, RED, rect, 0)

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

        # get shape of new grid
        grid_max_y, grid_max_x = grid_list[l_new].shape

        # outside of grid: stop
        if not((grid_x >= 0) and (grid_x < grid_max_x) and (grid_y >= 0) and (grid_y < grid_max_y)):

            # draw dot at edge intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

        # moving down level
        if l_new < l:

            print("Down")

            # compute new dt
            dt_x, dt_y = compute_dt(s_x, s_y, n, l_new)

            # t_x moved
            if axis_x:

                # t_x was moved, so find t_y position (adjust)
                x_old = o_x + t_y * r_x + step_x / 10**3
                y_old = o_y + t_y * r_y

                # compute index (level l_new)
                grid_y_old, _ = get_index(x_old, y_old, l_new, n)

                # difference in indices
                diff_y = abs(grid_y - grid_y_old)
                print(f"Y diff {diff_y}")

                # still at intial t_y value: update
                if initial_y:

                    # compute in level l_new
                    _, t_y_0, _, _ = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l_new)

                    # set
                    t_y = t_y_0 + (diff_y - 1) * dt_y

                # otherwise: update increment
                else:

                    dt_y = diff_y * dt_y

            # t_y moved
            else:

                # t_y was moved, so find t_x position (adjust)
                x_old = o_x + t_x * r_x
                y_old = o_y + t_x * r_y + step_y / 10**3

                # compute index (level l_new)
                _, grid_x_old = get_index(x_old, y_old, l_new, n)

                # difference in indices
                diff_x = abs(grid_x - grid_x_old)
                print(f"X diff {diff_x}")

                # still at intial t_x value: update
                if initial_x:

                    # compute in level l_new
                    t_x_0, _, _, _ = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l_new)

                    # set
                    t_x = t_x_0 + (diff_x - 1) * dt_x

                # otherwise: update increment
                else:

                    dt_x = diff_x * dt_x

        # moving up level
        elif l_new > l:
            print("UP")

            # get scaling
            factor = n ** (l_new - l)
            
            # compute old dt
            dt_x, dt_y = compute_dt(s_x, s_y, n, l)

            # t_x moved
            if axis_x:

                # t_x was moved, so find t_y position (adjust)
                x_old = o_x + t_y * r_x + step_x / 10**3
                y_old = o_y + t_y * r_y

                # compute index (level l)
                grid_y_old, _ = get_index(x_old, y_old, l, n)

                # difference in indices
                diff_y = abs(grid_y * factor - grid_y_old)
                print(f"Y diff {diff_y}")

                # still at intial t_y value: update
                if initial_y:

                    # compute in level l
                    _, t_y_0, _, _ = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l)

                    # set
                    t_y = t_y_0 + (diff_y - 1) * dt_y

                # otherwise: update increment
                else:

                    dt_y = diff_y * dt_y

            # t_y moved
            else:

                # t_y was moved, so find t_x position (adjust)
                x_old = o_x + t_x * r_x
                y_old = o_y + t_x * r_y + step_y / 10**3

                # compute index (level l)
                _, grid_x_old = get_index(x_old, y_old, l, n)

                # difference in indices
                diff_x = abs(grid_x * factor - grid_x_old)
                print(f"X diff {diff_x}")

                # still at intial t_x value: update
                if initial_x:

                    # compute in level l
                    t_x_0, _, _, _ = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l)

                    # set
                    t_x = t_x_0 + (diff_x - 1) * dt_x

                # otherwise: update increment
                else:

                    dt_x = diff_x * dt_x

        # same level
        else:

            print("Same")
            # ensure correct increment
            dt_x, dt_y = compute_dt(s_x, s_y, n, l)

        # increment chosen t
        if axis_x:
            t_x += dt_x
        else:
            t_y += dt_y

        # adjust scale
        s = n ** l_new

        # update level
        l = l_new

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, grid_pixel_x * s - 2, grid_pixel_y * s - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

        # draw dot at grid intersection
        draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

        if printing:
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

    print("")

    # draw line from start to intersection
    pygame.draw.line(window, PINK, (o_x * grid_pixel_x, o_y * grid_pixel_y), ((o_x + t * r_x) * grid_pixel_x, (o_y + t * r_y) * grid_pixel_y), width=2)

def newer_DDA(window, grid_list, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y, n, L, printing=False):
    '''
    Try to fix issues with wrong t values when changing level
    '''

    # cast ray until intersection
    intersection = False
    t_max = 1000.0
    t = 0.0

    # normalize direction
    r_norm = np.sqrt(r_x**2 + r_y**2)
    r_x = r_x / r_norm
    r_y = r_y / r_norm

    # axis scaling: account for zero divide in 1 / r_x & 1 / r_y
    if r_x == 0:
        s_x = np.inf
    else:
        s_x = 1 / abs(r_x)
    if r_y == 0:
        s_y = np.inf
    else:
        s_y = 1 / abs(r_y)

    # start at highest level (??? or lowest l = 0 might be faster)
    l = L

    # find highest empty level and intial grid index
    grid_y, grid_x, l = get_level(o_x, o_y, l, n, L, grid_list)

    # cellsize (level l)
    s = n ** l

    # check initial square
    if l == -1:

        # flag intersection
        intersection = True

        # default cellsize
        s = 1

        # draw red square for intersection
        rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
        pygame.draw.rect(window, RED, rect, 0)

    else:

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, s * grid_pixel_x - 2, s * grid_pixel_y - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

    # t distance between each axis (level l)
    dt_x, dt_y = compute_dt(s_x, s_y, n, l)

    # initial t distance (level l)
    t_x, t_y, step_x, step_y = compute_initial_dt(o_x, o_y, r_x, r_y, s_x, s_y, n, l) 

    # flag initial distance (if axis has not moved beyond initial distance = True)
    initial_x = True
    initial_y = True

    # draw dot at starting point
    draw_dot(window, o_x, o_y, grid_pixel_x, grid_pixel_y)

    if printing:
        print(f"Start")
        print(f"(x, y): ({o_x}, {o_y})")
        print(f"(i, j): ({grid_y}, {grid_x})")
        print(f"l: {l}")
        print(f"t {t}")
        print(f"t_x {t_x}")
        print(f"t_y {t_y}")
        print("")

    while ((not intersection) and (t < t_max)):

        # DDA step at level l

        # select axis with smaller current t distance
        if t_x < t_y:

            # step in x axis to new grid sqaure
            #grid_x += step_x

            # store as current t distance (NOTE: before incrementing t_x, as need check the square we are entering)
            t = t_x

            # increment t_x
            # t_x += dt_x
            axis_x = True
            initial_x = False

        else:

            # repeat for y axis
            #grid_y += step_y
            t = t_y
            #t_y += dt_y
            axis_x = False
            initial_y = False

        # get current position
        x = o_x + t * r_x
        y = o_y + t * r_y

        # shift towards next square (avoid sitting on gridline???)
        if axis_x:
            x += step_x / 10**3
        else:
            y += step_y / 10**3

        # find highest empty level and grid index
        grid_y, grid_x, l_new = get_level(x, y, l, n, L, grid_list)

        if printing:
            print(f"(x, y): ({x}, {y})")
            print(f"(i, j): ({grid_y}, {grid_x})")
            print(f"l: {l_new}")
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

        # no empty level: intersection
        if l_new == -1:

            intersection = True

            # draw red square for intersection
            rect = pygame.Rect(grid_x * grid_pixel_x + 1, grid_y * grid_pixel_y + 1, grid_pixel_x - 2, grid_pixel_y - 2)
            pygame.draw.rect(window, RED, rect, 0)

            # draw dot at grid intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

        # get shape of new grid
        grid_max_y, grid_max_x = grid_list[l_new].shape

        # outside of grid: stop
        if not((grid_x >= 0) and (grid_x < grid_max_x) and (grid_y >= 0) and (grid_y < grid_max_y)):

            # draw dot at edge intersection
            draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

            break

        # compute new dt (level l_new)
        dt_x, dt_y = compute_dt(s_x, s_y, n, l_new)

        if l_new < l:

            '''Down level'''

            # x moved
            if axis_x:

                # t_y may be too far ahead as moved in the higher level l
                # know t_x < t_y (as x moved) and at edge of level l_new
                # compute position of t_x and use to compute "initial" t distance from position to next y intersection (level l_new)
                # set t_y = t_x + initial t_y (l_new)

                # find t_x position
                x_old = o_x + t_x * r_x #- step_x / 10**3
                y_old = o_y + t_x * r_y #- step_y / 10**3

                # compute initial t distance in level l_new
                _, t_y_0, _, _ = compute_initial_dt(x_old, y_old, r_x, r_y, s_x, s_y, n, l_new)

                # update t_y
                t_y = t_x + t_y_0

            # y moved
            else:

                # find t_y position
                x_old = o_x + t_y * r_x #- step_x / 10**3
                y_old = o_y + t_y * r_y #- step_y / 10**3

                # compute initial t distance in level l_new
                t_x_0, _, _, _ = compute_initial_dt(x_old, y_old, r_x, r_y, s_x, s_y, n, l_new)

                # update t_x
                t_x = t_y + t_x_0

        elif l_new > l:

            '''Up level'''

            # t_x moved
            if axis_x:

                # t_y may now be in the centre of a level l_new square (as larger)
                # so need to compute position and "intial" t distance from position to next intersection (level l_new)
                # then add this to t_y

                # t_x was moved, so find t_y position (adjust behind to prevent addition when not needed)
                x_old = o_x + t_y * r_x #- step_x / 10**3
                y_old = o_y + t_y * r_y #- step_y / 10**3

                # compute initial t distance in level l_new
                _, t_y_0, _, _ = compute_initial_dt(x_old, y_old, r_x, r_y, s_x, s_y, n, l_new)

                # if equal to new dt_y: already on l_new intersection, no need to add
                if t_y_0 == dt_y:
                    # NOTE: might be issues here when very close / slightly off due to rounding
                    pass
                else:
                    # add to t_y
                    t_y += t_y_0

            # t_y moved
            else:

                # t_y was moved, so find t_x position (adjust)
                x_old = o_x + t_x * r_x #- step_x / 10**3
                y_old = o_y + t_x * r_y #- step_y / 10**3

                # compute initial t distance in level l_new
                t_x_0, _, _, _ = compute_initial_dt(x_old, y_old, r_x, r_y, s_x, s_y, n, l_new)

                # add to t_x
                if t_x_0 == dt_x:
                    pass
                else:
                    t_x += t_x_0

        else:

            '''same level'''

            pass

        # increment chosen t
        if axis_x:
            t_x += dt_x
        else:
            t_y += dt_y

        # adjust scale
        s = n ** l_new

        # update level
        l = l_new

        # draw blue square for no intersection
        rect = pygame.Rect(grid_x * s * grid_pixel_x + 1, grid_y * s * grid_pixel_y + 1, grid_pixel_x * s - 2, grid_pixel_y * s - 2)
        pygame.draw.rect(window, BLUE, rect, 0)

        # draw dot at grid intersection
        draw_dot(window, o_x + t * r_x, o_y + t * r_y, grid_pixel_x, grid_pixel_y)

        if printing:
            print(f"t {t}")
            print(f"t_x {t_x}")
            print(f"t_y {t_y}")
            print("")

    # draw line from start to intersection
    pygame.draw.line(window, PINK, (o_x * grid_pixel_x, o_y * grid_pixel_y), ((o_x + t * r_x) * grid_pixel_x, (o_y + t * r_y) * grid_pixel_y), width=2)

# -------------------------------------
# Main
# -------------------------------------

# line
o_x, o_y = 0.5, 2.5
r_x, r_y = 1, 0

# grid
'''
grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1],
])
'''

grid = np.array([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
])
n = 2
L = 2

grid_list = multilevel_grid(grid, n, L)

# grid size
grid_max_y, grid_max_x = grid.shape

# random grid
tau = 0.95
grid_max_y, grid_max_x = 32, 32
n = 2
L = 5
rng = np.random.default_rng(1)
grid = rng.uniform(0, 1, (grid_max_y, grid_max_x))
grid[grid < tau] = 0
grid[grid >= tau] = 1
#grid[63:65, 63:65] = 1
grid_list = multilevel_grid(grid, n, L)

# pixel size of grid squares
grid_pixel_x = window_width // grid_max_x
grid_pixel_y = window_height // grid_max_y

printing = True

while True:

    window.fill(BLACK)

    # events
    o_x, o_y, r_x, r_y = event_handler(o_x, o_y, r_x, r_y, grid_pixel_x, grid_pixel_y)
    #_, _, _, _ = event_handler(o_x, o_y, r_x, r_y, grid_pixel_x, grid_pixel_y)

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

    newer_DDA(window, grid_list, grid_pixel_x, grid_pixel_y, o_x, o_y, r_x, r_y, n, L, printing=printing)

    printing = False

    # update canvas
    pygame.display.flip()