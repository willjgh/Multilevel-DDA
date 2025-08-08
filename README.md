# Multilevel DDA

An implemenation of the (DDA) Digitial Differential Analyzer algorithm to efficiently trace a ray through a grid, only checking for intersections when the ray enters a new cell in the grid. Using a 'multilevel grid' as an acceleration structure reduces the number of steps further, allowing the ray to skip over large empty grid regions which are often present in e.g. 3D scenes.

Examples of the algorithm (without / with multilevel grid):

<p align="center">
<img src="/Images/dda_example_1.png?" width="400">
<img src="/Images/multilevel_dda_example_1.png?" width="400">
</p>
<p align="center">
<img src="/Images/dda_example_2.png?" width="400?">
<img src="/Images/multilevel_dda_example_2.png?" width="400">
</p>

The first level of the grid is stored as an array with empty cells as 0 and full squares as 1. Given a scale factor $n$, $n \times n$ slices of the grid are stored as a single value in the next level array: $1$ if the slice contains at least one $1$ entry, otherwise $0$. At each step of the DDA algorithm the highest grid level that is empty at the current position is used to step the ray, iterating until no empty level can be found i.e. intersection with the original grid.
