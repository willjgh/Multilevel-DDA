import pygame
import os
import math
import numpy as np
import pygame.gfxdraw
from profilehooks import profile
from copy import copy

rng = np.random.default_rng(255)

class Raytracer:

    def __init__(self, window_width=700, window_height=700, canvas_width=150, canvas_height=150):

        # initialize pygame
        pygame.init()

        # configs
        pygame.display.set_caption("Raytracer")
        self.clock = pygame.time.Clock()
        self.dt = 0.0
        self.font = pygame.font.SysFont("Arial" , 18 , bold = True)

        # initialise window: high resolution, display
        self.window_width = window_width
        self.window_height = window_height
        self.window = pygame.display.set_mode((window_width, window_height))

        # canvas: low resolution, draw to
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas = pygame.Surface((canvas_width, canvas_height))

        # camera: position and angle
        self.camera_position = np.array([32.0, 32.0, 1.0])
        self.camera_direction = np.array([0.0, 0.0, 1.0])
        self.plane_direction_u = np.array([1.0, 0.0, 0.0])
        self.plane_direction_v = np.array([0.0, 1.0, 0.0])
        self.theta = 0.0
        self.phi = 0.0
        self.psi = 0.0

        # field of view
        self.t_max = 1.0
        self.s_max = 1.0

        # running flag
        self.running = True

        # construct grid
        self.construct_multilevel_grid(
            M = 32,
            N = 32,
            K = 32,
            n = 2,
            L = 5
        )

    def construct_multilevel_grid(self, M, N, K, n, L):
        '''
        grid: base grid (level 0)
        n: size of grid squares to compress per level
        L: maximum level

        NOTE: need grid size | n**L
        '''

        # check sizes match
        if (M % (n ** L) != 0) or (N % (n ** L)) != 0 or (K % (n ** L)) != 0:
            raise Exception("Invalid grid scaling for size")

        # generate grid
        rng = np.random.default_rng(0)
        grid = rng.uniform(0, 1, (M, N, K))

        # threshold
        tau = 0.95
        grid[grid < tau] = 0
        grid[grid >= tau] = 1

        '''
        grid = np.zeros((M, N, K))
        grid[0, :, :] = 1
        grid[:, 0, :] = 1
        grid[:, :, 0] = 1
        '''

        # store grids in level order
        grid_list = [grid]

        # store current grid
        current_grid = grid

        # for each level
        for l in range(1, L + 1):

            # get current grid size
            M_current, N_current, K_current = current_grid.shape

            # compute higher grid size
            M_higher, N_higher, K_higher = M_current // n, N_current // n, K_current // n

            # construct higher grid
            higher_grid = np.empty((M_higher, N_higher, K_higher))

            # loop over larger grid
            for i in range(M_higher):
                for j in range(N_higher):
                    for k in range(K_higher):

                        # current grid index
                        i_current = i * n
                        j_current = j * n
                        k_current = k * n

                        # slice current grid
                        grid_square = current_grid[i_current:(i_current+n), j_current:(j_current+n), k_current:(k_current+n)]

                        # compute max
                        entry = int(np.max(grid_square))

                        # fill higher grid
                        higher_grid[i, j, k] = entry

            # store grid
            grid_list.append(higher_grid)

            # update current grid
            current_grid = higher_grid

        # store
        self.grid_list = grid_list
        self.n = n
        self.L = L
    
    def framerate_counter(self):
        """Calculate and display frames per second."""
        # get fps
        fps = f"fps: {int(self.clock.get_fps())}"
        # create text
        fps_t = self.font.render(fps , 1, (0, 255, 0))
        # display on canvas
        self.window.blit(fps_t,(0, 0))

    def test_movement(self, x_movement, y_movement):
        '''Test if movement collides with grid, move if no issues'''

        # new position
        new_position_x = self.camera_position[0] + x_movement
        new_position_y = self.camera_position[1] + y_movement

        # new grid square
        new_grid_x = math.floor(new_position_x)
        new_grid_y = math.floor(new_position_y)

        # check if in grid
        if (new_grid_x >= 0) and (new_grid_x < self.n) and (new_grid_y >= 0) and (new_grid_y < self.n):

            # check if empty
            if self.grid[new_grid_y][new_grid_x] == 0:

                # update position
                self.camera_position = [new_position_x, new_position_y]

    
    def input(self):
        '''Take inputs'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

        # get held keys
        keys = pygame.key.get_pressed()

        # movement scaled by time since last frame
        step = 0.1 #0.005 * self.dt
        # turn = 0.005 * self.dt
        
        # update camera position (if no collision) and camera angle
        if keys[pygame.K_w]:
            self.camera_position += self.camera_direction * step
        if keys[pygame.K_s]:
            self.camera_position -= self.camera_direction * step
        if keys[pygame.K_a]:
            self.camera_position -= self.plane_direction_u * step
        if keys[pygame.K_d]:
            self.camera_position += self.plane_direction_u * step
        if keys[pygame.K_SPACE]:
            self.camera_position -= self.plane_direction_v * step
        if keys[pygame.K_LSHIFT]:
            self.camera_position += self.plane_direction_v * step
        '''   
        if keys[pygame.K_LEFT]:
            self.camera_angle += turn
        if keys[pygame.K_RIGHT]:
            self.camera_angle -= turn
        '''

    def render(self):

        # draw background
        self.canvas.fill((0, 0, 0))

        # for each pixel in canvas
        for x in range(self.canvas_width):
            for y in range(self.canvas_height):

                # get corresponding ray direction
                t = (((2 * x) / (self.canvas_width - 1)) - 1) * self.t_max
                s = (((2 * y) / (self.canvas_height - 1)) - 1) * self.s_max
                ray_direction = self.camera_direction + t * self.plane_direction_u + s * self.plane_direction_v

                # define ray
                o_x = self.camera_position[0]
                o_y = self.camera_position[1]
                o_z = self.camera_position[2]
                r_x = ray_direction[0]
                r_y = ray_direction[1]
                r_z = ray_direction[2]

                # raycast to get intersection
                intersection, intersection_distance, grid_value, axis_x, axis_y, axis_z = self.DDA(self.grid_list, o_x, o_y, o_z, r_x, r_y, r_z, self.n, self.L)

                # if inside cell: t = 0
                if intersection_distance == 0.0:

                    # fill screen
                    #self.canvas.fill((255, 255, 255))

                    # break to skip rays
                    #break

                    pygame.gfxdraw.pixel(self.canvas, x, y, (255, 255, 255))

                # if intersection with grid
                elif intersection:

                    # compute colour
                    if grid_value == 1:

                        # distance factor
                        factor = (2 / (intersection_distance + 2))
                        level = int(255 * factor)

                        if level < 0:
                            level = 0
                        elif level > 255:
                            level = 255

                        if axis_x:
                            colour = (0, 0, level)
                        if axis_y:
                            colour = (0, level, 0)
                        if axis_z:
                            colour = (level, 0, 0)

                    # draw pixel
                    try:
                        pygame.gfxdraw.pixel(self.canvas, x, y, colour)
                    except TypeError:
                        print(colour)
        
        # blit surface to window
        self.window.blit(pygame.transform.scale(self.canvas, self.window.get_rect().size), (0, 0))
        self.framerate_counter()

        # update canvas
        pygame.display.flip()

    def old_DDA(self, ray_direction):

        # compute ray distance scaling for each axis 
        s_x = 1 / abs(ray_direction[0])
        s_y = 1 / abs(ray_direction[1])

        # intial grid position
        grid_x = math.floor(self.camera_position[0])
        grid_y = math.floor(self.camera_position[1])

        # initial distances to boundary of current grid square
        if ray_direction[0] < 0:
            step_x = -1
            ray_length_x = (self.camera_position[0] - grid_x) * s_x
        else:
            step_x = 1
            ray_length_x = (grid_x + 1 - self.camera_position[0]) * s_x
        
        if ray_direction[1] < 0:
            step_y = -1
            ray_length_y = (self.camera_position[1] - grid_y) * s_y
        else:
            step_y = 1
            ray_length_y = (grid_y + 1 - self.camera_position[1]) * s_y

        # cast ray until intersection
        intersection = False
        intersection_face = None
        intersection_value = None
        max_distance = 1000.0
        current_distance = 0.0

        while ((not intersection) and (current_distance < max_distance)):

            # choose smaller ray length axis
            if ray_length_x < ray_length_y:

                # step in x axis to new grid sqaure
                grid_x += step_x

                # store current distance to new grid square
                current_distance = ray_length_x

                # current intersection with x-axis face
                intersection_face = True

                # update ray length due to new step
                ray_length_x += s_x

            else:
                
                grid_y += step_y
                current_distance = ray_length_y
                intersection_face = False
                ray_length_y += s_y

            # check if intersection with new grid square
            if (grid_x >= 0) and (grid_x < self.n) and (grid_y >= 0) and (grid_y < self.n):

                intersection_value = self.grid[grid_y][grid_x]
                
                # check if empty
                if intersection_value != 0:
                    intersection = True
            else:
                break

        # threshold minimum distance: prevent zero division errors
        if current_distance < 0.1:
            current_distance = 0.1

        # return status and distance
        return intersection, current_distance, intersection_face, intersection_value
    
    def get_index(self, x, y, z, l, n):
        '''
        Find index of position (x, y, z) in grid level l (scaling of n)
        '''

        scale = n ** l
        i = math.floor(y / scale)
        j = math.floor(x / scale)
        k = math.floor(z / scale)

        return i, j, k

    def get_level(self, x, y, z, l, n, L, grid_list):
        '''
        Find the highest level where the position (x, y, z) is in an empty square
        Return the level and index
        
        (x, y, z): position
        l: current level
        n: grid scaling
        L: max level
        grid_list: grid levels
        '''

        # get index in current grid
        i, j, k = self.get_index(x, y, z, l, n)

        # get current grid entry
        try:
            current = grid_list[l][i, j, k]
        except IndexError:
            # outside of grid
            return i, j, k, l

        # empty
        if current == 0:

            # increase level until next full (or reach L)
            while l < L:

                # get index in level l + 1
                i_higher, j_higher, k_higher = self.get_index(x, y, z, l + 1, n)

                # get entry
                higher = grid_list[l + 1][i_higher, j_higher, k_higher]

                # empty
                if higher == 0:

                    # increase level
                    l += 1

                    i = i_higher
                    j = j_higher
                    k = k_higher

                # full
                else:

                    # found highest empty level
                    break

        # full
        else:

            # lowest level full
            if l == 0:

                return i, j, k, -1

            # decrease level until empty
            while l > 0:

                # decrease level
                l -= 1

                # get index in level
                i, j, k = self.get_index(x, y, z, l, n)

                # get entry
                lower = grid_list[l][i, j, k]

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
        return i, j, k, l
    
    def compute_initial_dt(self, o_x, o_y, o_z, r_x, r_y, r_z, s_x, s_y, s_z, n, l):

        # get index of origin (level l)
        grid_y, grid_x, grid_z = self.get_index(o_x, o_y, o_z, l, n)

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

        if r_z < 0:
            step_z = -1
            t_z = (o_z - grid_z * s) * s_z
        else:
            step_z = 1
            t_z = ((grid_z + 1) * s - o_z) * s_z

        return t_x, t_y, t_z, step_x, step_y, step_z

    def compute_dt(self, s_x, s_y, s_z, n, l):

        # get scale
        s = n ** l

        # t distance between each axis (level l)
        dt_x = s * s_x
        dt_y = s * s_y
        dt_z = s * s_z

        return dt_x, dt_y, dt_z
    
    def DDA(self, grid_list, o_x, o_y, o_z, r_x, r_y, r_z, n, L):
        '''
        Step through grid using DDA algorithm

        Args:
            grid_list: list of multilevel grids
            (o_x, o_y, o_z): starting position
            (r_x, r_y, r_z): ray direction
            n: grid level scale factor
            L: max grid level

        Returns:
            (bool) intersection
            (float) distance to intersection
            (int) value of grid at intersection
            (bool) x-axis intersection flag
            (bool) y-axis intersection flag
            (bool) z-axis intersection flag


        '''

        # normalize ray direction
        r_norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)
        r_x = r_x / r_norm
        r_y = r_y / r_norm
        r_z = r_z / r_norm

        # axis scaling: account for zero divide in 1 / r
        if r_x == 0:
            s_x = np.inf
        else:
            s_x = 1 / abs(r_x)
        if r_y == 0:
            s_y = np.inf
        else:
            s_y = 1 / abs(r_y)
        if r_z == 0:
            s_z = np.inf
        else:
            s_z = 1 / abs(r_z)

        # start at highest level (lowest l = 0 might be faster??)
        l = L

        # find highest empty level and intial grid index
        grid_y, grid_x, grid_z, l = self.get_level(o_x, o_y, o_z, l, n, L, grid_list)

        # intersection: inside a filled cell
        if l == -1:

            # get value of starting cell
            grid_value = self.grid_list[0][grid_y, grid_x, grid_z]

            return True, 0.0, grid_value, False, False, False

        # t distance between each axis (level l)
        dt_x, dt_y, dt_z = self.compute_dt(s_x, s_y, s_z, n, l)

        # initial t distance (level l)
        t_x, t_y, t_z, step_x, step_y, step_z = self.compute_initial_dt(o_x, o_y, o_z, r_x, r_y, r_z, s_x, s_y, s_z, n, l)

        # cast ray until intersection
        t_max = 100.0
        t = 0.0

        # flag axis changes
        axis_x = False
        axis_y = False
        axis_z = False

        while t < t_max:

            # DDA step at level l

            # select axis with smaller current t distance
            if t_x < t_y:

                if t_x < t_z:

                    # t_x smallest
                    t = t_x
                    axis_x = True
                    axis_y = False
                    axis_z = False

                else:

                    # t_z smallest
                    t = t_z
                    axis_x = False
                    axis_y = False
                    axis_z = True

            else:

                if t_y < t_z:

                    # t_y smallest
                    t = t_y
                    axis_x = False
                    axis_y = True
                    axis_z = False

                else:

                    # t_z smallest
                    t = t_z
                    axis_x = False
                    axis_y = False
                    axis_z = True

            # get current position
            x = o_x + t * r_x
            y = o_y + t * r_y
            z = o_z + t * r_z

            # shift towards next square (move off edge of cell)
            if axis_x:
                x += step_x / 10**3
            if axis_y:
                y += step_y / 10**3
            if axis_z:
                z += step_z / 10**3

            # find highest empty level and grid index
            grid_y, grid_x, grid_z, l_new = self.get_level(x, y, z, l, n, L, grid_list)

            # no empty level: intersection
            if l_new == -1:

                # get value of cell
                grid_value = self.grid_list[0][grid_y, grid_x, grid_z]

                return True, t, grid_value, axis_x, axis_y, axis_z

            # get shape of new grid
            grid_max_y, grid_max_x, grid_max_z = grid_list[l_new].shape

            # outside of grid: stop
            if not((grid_x >= 0) and (grid_x < grid_max_x) and (grid_y >= 0) and (grid_y < grid_max_y) and (grid_z >= 0) and (grid_z < grid_max_z)):

                return False, t, None, False, False, False

            # compute new dt (level l_new)
            dt_x, dt_y, dt_z = self.compute_dt(s_x, s_y, s_z, n, l_new)

            if l_new < l:

                '''
                Down level
                
                move t_a and drop to level l_new
                t_b, t_c may now be too far ahead, as stepped in higher level l
                know t_a < all (as just moved) and must be at first intersection with level l_new grid
                compute position of t_a and use to compute 'initial' t distance from position to next b and c axis intersections

                => avoid overshooting intersections when moving down into lower level grid

                NOTE: may be issue if t_a is also a t_b or t_c intersection, as then will skip to next: so add check
                '''

                # t_x moved
                if axis_x:

                    # t_x position
                    x_old = o_x + t_x * r_x
                    y_old = o_y + t_x * r_y
                    z_old = o_z + t_x * r_z

                    # compute initial t distance in level l_new
                    _, t_y_0, t_z_0, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # update t_y, t_z
                    if not (t_y_0 == dt_y):
                        t_y = t_x + t_y_0
                    if not (t_z_0 == dt_z):
                        t_z = t_x + t_z_0

                # t_y moved
                if axis_y:

                    # t_y position
                    x_old = o_x + t_y * r_x
                    y_old = o_y + t_y * r_y
                    z_old = o_z + t_y * r_z

                    # compute initial t distance in level l_new
                    t_x_0, _, t_z_0, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # update t_x, t_z
                    if not (t_x_0 == dt_x):
                        t_x = t_y + t_x_0
                    if not (t_z_0 == dt_z):
                        t_z = t_y + t_z_0

                # t_z moved
                if axis_z:

                    # t_z position
                    x_old = o_x + t_z * r_x
                    y_old = o_y + t_z * r_y
                    z_old = o_z + t_z * r_z

                    # compute initial t distance in level l_new
                    t_x_0, t_y_0, _, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # update t_x, t_y
                    if not (t_x_0 == dt_x):
                        t_x = t_z + t_x_0
                    if not (t_y_0 == dt_y):
                        t_y = t_z + t_y_0

            elif l_new > l:

                '''
                Up level
                
                move t_a and move up to level l_new
                t_b, t_c may now be in the middle of higher level l_new square, as stepped in lower level grid (smaller cells)
                so compute position and 'initial' t distance from position to next b and c axis intersections (in higher grid)

                => avoid undershooting intersections when moving up into higher level grid

                NOTE: if already on grid l_new then should not skip to next, so add check
                '''

                # t_x not moved
                if not axis_x:

                    # t_x position
                    x_old = o_x + t_x * r_x
                    y_old = o_y + t_x * r_y
                    z_old = o_z + t_x * r_z

                    # compute initial t distance in level l_new
                    t_x_0, _, _, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # check not on grid
                    if not (t_x_0 == dt_x):
                        t_x += t_x_0

                # t_y not moved
                if not axis_y:

                    # t_y position
                    x_old = o_x + t_y * r_x
                    y_old = o_y + t_y * r_y
                    z_old = o_z + t_y * r_z

                    # compute initial t distance in level l_new
                    _, t_y_0, _, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # check not on grid
                    if not (t_y_0 == dt_y):
                        t_y += t_y_0
                    
                # t_z not moved
                if not axis_z:

                    # t_z position
                    x_old = o_x + t_z * r_x
                    y_old = o_y + t_z * r_y
                    z_old = o_z + t_z * r_z

                    # compute initial t distance in level l_new
                    _, _, t_z_0, _, _, _ = self.compute_initial_dt(x_old, y_old, z_old, r_x, r_y, r_z, s_x, s_y, s_z, n, l_new)

                    # check not on grid
                    if not (t_z_0 == dt_z):
                        t_z += t_z_0

            # increment t in chosen axis
            if axis_x:
                t_x += dt_x
            if axis_y:
                t_y += dt_y
            if axis_z:
                t_z += dt_z

            # update level
            l = l_new

        # exceed max distance
        return False, t, None, False, False, False

    def run(self):
        
        # loop
        while self.running:

            # clock
            self.dt = self.clock.tick()

            # take input
            self.input()

            # draw
            self.render()

        # quit
        pygame.quit()


#@profile
def main():
    game = Raytracer(
        window_width  = 700,
        window_height = 700,
        canvas_width  = 50,
        canvas_height = 50
    )
    game.run()


if __name__ == "__main__":
    main()