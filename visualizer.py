from queue import PriorityQueue  # for A*
from collections import deque  # for BFS & DFS

from typing import Generator
from math import inf
import pygame

# FRONTEND
WIN_HEIGHT = 900  # window height
WAIT = 0.0015  # in seconds

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (60, 240, 90)
RED = (247, 55, 37)
VIOLET = (195, 59, 219)  # not soft purple
PURPLE = (235, 191, 255)  # softish purple
CYAN = (131, 252, 224)  # softish cyan
TURQUOISE = (49, 208, 232)
ORANGE = (255, 165 ,0)
YELLOW = (242, 252, 131)

GRID_LINE = GRAY
UNWEIGHTED = WHITE
WEIGHTED = None  # make it very light yellow
BARRIER = BLACK
QUEUED = PURPLE  # in queue but not processed yet
PASSED = CYAN  # out of queue and done processing
START = TURQUOISE
TARGET = ORANGE
PATH = YELLOW
CURRENT = VIOLET  # too quick to see

# BACKEND
COORD_OFFSETS_CORNERS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
COORD_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # without corner nodes


class Node:

    def __init__(self, visualizer, row: int, col: int):
        # frontend stuff
        self.parent = visualizer  # visualizer class that holds these nodes
        self.size = visualizer.cell_size  # side length of the cell
        self.x, self.y = col * self.size, row * self.size  # coord in the pygame window
        self.state = UNWEIGHTED  # state of node represented by color

        # backend stuff
        self.row, self.col = row, col  # coord in the grid (graph)
        self.weight = 0  # weight of node (is also the current state of the node)

    def get_coord(self) -> tuple[int, int]:
        """ Returns node's coordinate. """
        return (self.row, self.col)

    def reset(self):
        """ Resets node's state to unweighted. """
        self.state = UNWEIGHTED
        self.weight = 0

    def make_start(self):
        """ Set node to start node. """
        self.state = START
        self.weight = 0

    def make_target(self):
        """ Set node to target node. """
        self.state = TARGET
        self.weight = 0

    def make_barrier(self):
        """ Set node to barrier node. """
        self.state = BARRIER
        self.weight = inf

    def is_barrier(self) -> bool:
        """ Returns whether node is a barrier. """
        return self.state == BARRIER

    def is_empty(self) -> bool:
        """ Returns whether node is a traversable/empty node (not barrier). """
        return self.state != BARRIER


class Visualizer:
    """ NOTE: window is referring to the pygame window, where everything is drawn (frontend).
    grid refers to the graph of nodes where the pathfinding algorithms actually work (backend). """

    def __init__(self, rows: int = 50, cols: int = 80, win_height: int = WIN_HEIGHT, win_title: str = 'Pathfinding Algorithms Visualized'):
        # pygame window dimensions
        self.win_height = win_height  # window height
        self.cell_size = self.win_height // rows  # visual grid cell size
        self.win_width = self.cell_size * cols  # window width

        # pygame window (everything is drawn here)
        self.win = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption(win_title)
        # pygame.event.set_blocked(pygame.MOUSEMOTION)  # ignores all mouse movement events
        self.click_state = START  # type of node to place when clicking

        # grid (graph) represented as a matrix of nodes
        self.rows, self.cols = rows, cols  # grid dimensions
        self.grid = self.gen_grid()  # matrix of node objects
        self.start = self.target = None  # start and target nodes

        # maps each pathfinding algorithm to a number keypress
        self.algorithms = {
            pygame.K_1: self.dfs,
            pygame.K_2: self.bfs,
            # pygame.K_3: self.dijkstra,
            # pygame.K_4: self.a_star
        }

    def reset_grid(self):
        """ Resets grid to empty. """
        self.start = self.target = None

        # goes through entire grid and sets each node's weight back to 0.
        for r in range(self.rows):
            for c in range(self.cols):
                self.get_node(r, c).state = UNWEIGHTED

    def clean_grid(self):
        """ Resets nodes colored during pathfinding. """

        # goes through entire grid and sets each node's weight back to 0.
        for r in range(self.rows):
            for c in range(self.cols):
                if self.get_node(r, c).state not in (UNWEIGHTED, WEIGHTED, START, TARGET, BARRIER):
                    self.get_node(r, c).state = UNWEIGHTED

    def gen_grid(self) -> list[list]:
        """ Generate the grid of empty nodes. """
        grid = []

        for r in range(self.rows):
            grid.append([])  # append new list for next row
            for c in range(self.cols):
                # make new node and append it to last row created
                node = Node(self, r, c)
                grid[-1].append(node)

        return grid

    def get_clicked_coord(self, pos):
        """ Gets the coord of node clicked based on position clicked in window. """
        x, y = pos

        # divides window position by cell width to see row/col number
        row = y // self.cell_size
        col = x // self.cell_size

        return row, col

    def get_clicked_node(self, pos):
        """ Gets the node clicked based on position in window clicked. """
        return self.get_node(*self.get_clicked_coord(pos))

    # === FRONTEND (pygame/drawing) ===
    def draw_grid_lines(self):
        """ Draws the grid lines onto pygame window. """
        for r in range(self.rows):
            # draw row line from left edge at row pos, to right edge
            pygame.draw.line(self.win, GRID_LINE, (0, r*self.cell_size), (self.win_width, r*self.cell_size))
            for c in range(self.cols):
                # draw column line from top edge at col pos, to bottom edge
                pygame.draw.line(self.win, GRID_LINE, (c*self.cell_size, 0), (c*self.cell_size, self.win_height))

    def draw_node(self, node):
        """ Draws given node onto pygame window. """
        pygame.draw.rect(self.win, node.state, (node.x, node.y, self.cell_size, self.cell_size))

    def draw_node_grid(self, node):
        """ Only draws the grid lines that are supposed to be around given node.  """
        # draw top line
        pygame.draw.line(self.win, GRID_LINE, (node.x, node.y), (node.x+self.cell_size, node.y))

        # draw left line
        pygame.draw.line(self.win, GRID_LINE, (node.x, node.y), (node.x, node.y+self.cell_size))

        # draw bottom line
        pygame.draw.line(self.win, GRID_LINE, (node.x, node.y+self.cell_size), (node.x+self.cell_size, node.y+self.cell_size))

        # draw right line
        pygame.draw.line(self.win, GRID_LINE, (node.x+self.cell_size, node.y), (node.x+self.cell_size, node.y+self.cell_size))

    def delay(self, wait: float = WAIT):
        """ Delay some amount of time, for animation/visual purposes. """
        pygame.time.delay(int(wait*1000))

    def update_node(self, node, wait: float = WAIT):
        """ Draws given node and its grid, then updates display (more efficient then redrawing whole window every time). """
        self.draw_node(node)
        self.draw_node_grid(node)
        if wait is not None:
            self.delay(wait)
        pygame.display.update()

    def draw(self):
        """ Redraws all elements onto window (updates display). """
        # fill with with white
        self.win.fill(WHITE)

        # goes through every node in the grid and draws it
        for row in self.grid:
            for node in row:
                self.draw_node(node)

        # draw grid lines
        self.draw_grid_lines()

        pygame.display.update()

    def place_start(self, node: Node):
        """ Place start node and redraw (also handles conflicts). """
        # if clicked node is already target, can't draw over it
        if node == self.target:
            print('cannot overwrite target node')
            return
        # start not already set, just place it
        elif self.start == None:
            self.start = node  # set start node to this node
            node.make_start()  # switch node's state to start node
        # start node already set, so move it
        else:
            self.start.state = UNWEIGHTED  # set old start to unweighted
            self.update_node(self.start, None)
            self.start = node  # set new start to clicked node
            node.make_start()  # set clicked node state to start
        # we allow drawing over barrier nodes

    def place_target(self, node: Node):
        """ Place target node and redraw (also handles conflicts). """
        # if clicked node is already start, can't draw over it
        if node == self.start:
            print('cannot overwrite start node')
            return
        # target not already set, just place it
        elif self.target == None:
            self.target = node  # set target node to this node
            node.make_target()  # switch node's state to target node
        # target node already set, so move it
        else:
            self.target.state = UNWEIGHTED  # set old target to unweighted
            self.update_node(self.target, None)
            self.target = node  # set new target to clicked node
            node.make_target()  # set clicked node state to target
        # we allow drawing over barrier nodes

    def place_barrier(self, node: Node):
        """ Place barrier node and redraw (also handles conflicts). """
        # as long as node isn't start or target, draws barrier
        if node != self.start and node != self.target:
            node.make_barrier()  # switch node's state to barrier node
        else:
            print('cannot overwrite start/target node')

    # === BACKEND (algorithms) ===

    # pathfinding algorithms
    def trace_path(self, parents: dict[Node: Node]):
        """ Traces path back to start node given parent dict (just visually). """
        curr = parents[self.target]  # starts trace from target node
        ## path = [self.target]

        # keeps tracing each node's parent and building the path until it reaches the start node
        while curr != self.start:
            curr.state = PATH
            self.update_node(curr, WAIT*6)
            ## path.append(curr)  # adds node to path
            curr = parents[curr]  # traces to parent

        ## path.append(self.start)  # adds start node to complete the path
        ## path.reverse()  # and reverses path since it was built backwards

        ## return path

    # TODO: drawing the arrows of every node's parent would be a dope little side thing to make sometime

    def bfs(self):
        """ Runs breadth first search for target node and then traces path back to start (unweighted). """
        queue = deque([self.start])  # append and popleft
        discovered = {self.start}  # keeps track of discovered nodes
        parents = {self.start: None}  # keeps track of every node's parent

        # while there's still nodes to check
        while len(queue) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # pops next node to process and colors it
            curr = queue.popleft()

            # if target node is found, traces paths back to start
            if curr == self.target:
                curr.state = TARGET  # color curr back to target
                self.update_node(curr)
                self.trace_path(parents)
                return

            # add adjacent nodes to the queue
            for adj in self.adjacent_nodes(curr):
                if adj not in discovered and not adj.is_barrier():  # can't add barrier nodes
                    parents[adj] = curr  # adjacent node's parent is curr node
                    discovered.add(adj)
                    queue.append(adj)
                    if adj != self.target:
                        adj.state = QUEUED
                    self.update_node(adj)

            # set current state to passed once done processing
            if curr != self.start:
                curr.state = PASSED
                self.update_node(curr)

    def dfs(self):
        """ Runs depth first search for target node and then traces path back to start (unweighted). """
        stack = deque([self.start])  # append and pop
        discovered = {self.start}  # keeps track of discovered nodes
        parents = {self.start: None}  # keeps track of every node's parent

        # while there's still nodes to check
        while len(stack) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # pops next node to process and colors it
            curr = stack.pop()

            # if target node is found, traces paths back to start
            if curr == self.target:
                curr.state = TARGET  # color curr back to target
                self.update_node(curr)
                self.trace_path(parents)
                return

            # add adjacent nodes to the stack
            for adj in self.adjacent_nodes(curr):
                if adj not in discovered and not adj.is_barrier():  # can't add barrier nodes
                    parents[adj] = curr  # adjacent node's parent is curr node
                    discovered.add(adj)
                    stack.append(adj)
                    if adj != self.target:
                        adj.state = QUEUED
                    self.update_node(adj)

            # set current state to passed once done processing
            if curr != self.start:
                curr.state = PASSED
                self.update_node(curr)


    def next_min_dist(self, dist: list, spt: list) -> int:
        """ Returns the shortest distance node from the set of unexplored nodes. """

        min_dist = inf  # initializes current min distance as int max
        min_index = -1  # keeps track of index of min distance node

        for u in range(self.vertices):  # for node in vertices
            # set node as new min if node not explored and closer than current min
            if not spt[u] and dist[u] < min_dist:
                min_dist = dist[u]
                min_index = u

        return min_index

    # def dijkstra(self, src: int, dest: int) -> list:  # O(n^2)
    #     """ Returns a list of the shortest paths from the source node to
    #         every node in the graph (calculated using Dijkstra's algorithm). """

    #     dist = [inf] * self.vertices  # list keeps track of node's distances
    #     dist[src] = 0  # distance from source to itself is clearly 0
    #     spt = [False] * self.vertices  # keeps track of shortest path to each node

    #     # iterates number of nodes times
    #     for iteration in range(self.vertices):  # O(n)
    #         if spt[dest]:
    #             return dist[dest]

    #         # picks the next closest out of all the unexplored nodes
    #         curr = self.next_min_dist(dist, spt)

    #         # adds next closest to the shortest path tree
    #         spt[curr] = True

    #         # iterates through every node as destination
    #         for dst in range(self.vertices):  # O(n)
    #             """ I separated the conditional to explain and understand it a lot better. """
    #             # if not spt[dst] and self.graph[curr][dst] > 0 and dist[dst] > dist[curr] + self.graph[curr][dst]:

    #             # then this node was already checked
    #             if spt[dst]:
    #                 continue

    #             # then this is the same node or there's no edge between them
    #             elif self.graph[curr][dst] == 0:
    #                 continue

    #             # dest's dist should be big, if not then it was already calculated
    #             elif dist[dst] <= dist[curr] + self.graph[curr][dst]:
    #                 continue

    #             """ calculate destination's path distance as current's path
    #             distance plus distance from current to destination. """
    #             dist[dst] = dist[curr] + self.graph[curr][dst]

    #     return dist  # returns the list of shortest paths from the source to every node

    # def dijkstra(self):
    #     """ Runs Dijkstra's search for target node and then traces path back to start (weighted). """
    #     node_count = self.rows * self.cols
    #     # TODO: fix comment for difference of distances and shortest paths set
    #     # start and target node's coords
    #     (sr, sc), (tr, tc) = self.start.get_coord(), self.target.get_coord()

    #     distances = self.gen_matrix(inf)  # keeps track of every node's distance from start node
    #     distances[sr][sc] = 0  # distance from source to itself is clearly 0
    #     spt = self.gen_matrix(False)  # keeps track of shortest path to each node

    #     for _ in range(node_count):
    #         if spt[tr][tc]:
    #             curr.state = TARGET  # color curr back to target
    #             self.update_node(curr)
    #             self.trace_path(distances[tr][tc])
    #             return

    #         # picks the next closest node out of all the unexplored nodes
    #         cr, cc = self.next_min_dist(distances, spt)

    #         # adds next closest node to the shortest path tree
    #         spt[cr][cc] = True

    #         # iterates through every node as destination
    #         for r in range(self.rows):
    #             for c in range(self.cols):
    #                 """ I separated the conditional to explain and understand it a lot better. """
    #                 # if not spt[dst] and self.graph[curr][dst] > 0 and dist[dst] > dist[curr] + self.graph[curr][dst]:

    #                 # NOTE: cr, cc is curr and r, c is dst

    #                 # then this node was already checked
    #                 if spt[r][c]:
    #                     continue

    #                 # then this is the same node or there's no edge between them
    #                 elif (cr, cc) == (r, c):
    #                     continue

    #                 # dest's distance should be big, if not then it was already calculated
    #                 elif distances[r][c] <= distances[cr][cc] + self.graph[curr][dst]:
    #                     continue

    #                 """ calculate destination's path distance as current's path
    #                 distance plus distance from current to destination. """
    #                 dist[dst] = dist[curr] + self.graph[curr][dst]

    #     return dist  # returns the list of shortest paths from the source to every node


    def a_star(self):
        """ Runs A* 😍😍😍 for target node and then traces path back to start (weighted). """
        print('running a_star...')

    # maze generation algorithms
    def generate_maze(self):
        """ Generates maze of barrier nodes using BLANK algorithm. """
        pass

    # helper functions
    def gen_matrix(self, default=None) -> list[list]:
        """ Generates a matrix the size of the grid. """
        matrix = []

        for r in range(self.rows):
            matrix.append([])  # append new list for next row
            if default is not None:
                for c in range(self.cols):
                    matrix[-1].append(default)

        return matrix

    def adjacent_nodes(self, node: Node) -> Generator[Node, None, None]:
        """ Generates the traversable nodes adjacent to the given node. """
        for offset in COORD_OFFSETS:
            # offsets cord, then yields node at that coord as long as coord is within bounds
            adj = self.offset_coord(node.get_coord(), offset)
            if self.in_bounds(*adj):
                yield self.get_node(*adj)

    def offset_coord(self, coord: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
        """ Returns a coord with the given offset. """
        return tuple(crd + ofst for crd, ofst in zip(coord, offset))

    def in_bounds(self, r: int, c: int) -> bool:
        """ Returns whether the given coord is within bounds. """
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_node(self, r: int, c: int) -> Node:
        """ Returns the node at the given coord. """
        return self.grid[r][c]

    # === MAIN ===
    def run(self):
        """ Main function. """

        # main update loop
        running = True
        self.draw()  # updates whole display
        while running:
            # MOUSEMOTION event is important because it allows you to hold down and place

            for event in pygame.event.get():

                # window exit button
                if event.type == pygame.QUIT:
                    running = False
                    break  # breaks event loop, not main loop

                """ # mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # get position on window clicked and then get node at position
                    pos = pygame.mouse.get_pos()
                    node = self.get_clicked_node(pos) """

                # left click, place node based on click state
                if pygame.mouse.get_pressed()[0]:
                    # get position on window clicked and then get node at position
                    pos = pygame.mouse.get_pos()
                    node = self.get_clicked_node(pos)

                    if self.click_state == START:  # place start node
                        self.place_start(node)  # NOTE: there's an update node in here for when you're moving the node
                        self.update_node(node, None)
                    elif self.click_state == TARGET:  # place target node
                        self.place_target(node)  # NOTE: there's an update node in here for when you're moving the node
                        self.update_node(node, None)
                    elif self.click_state == BARRIER:  # place barrier node
                        self.place_barrier(node)
                        self.update_node(node, None)

                # right click, delete node (regardless of click state)
                elif pygame.mouse.get_pressed()[2]:
                    # get position on window clicked and then get node at position
                    pos = pygame.mouse.get_pos()
                    node = self.get_clicked_node(pos)
                    node.reset()  # reset node to unweighted
                    self.update_node(node, None)

                    # if node was start or target, reset that too
                    if node == self.start:
                        self.start = None
                    elif node == self.target:
                        self.target = None

                # key press
                if event.type == pygame.KEYDOWN:
                    # number, run corresponding pathfinding algorithm (start & target node are also set)
                    if event.key in self.algorithms and self.start and self.target:
                        self.algorithms[event.key]()

                    # R, clear all nodes and reset start & target nodes
                    elif event.key == pygame.K_r:
                        self.reset_grid()
                        self.draw()

                    # C, wipes nodes that were colored during pathfinding
                    elif event.key == pygame.K_c:
                        self.clean_grid()
                        self.draw()

                    # S, switch click state to place start node
                    elif event.key == pygame.K_s:
                        self.click_state = START

                    # T, switch click state to place target node
                    elif event.key == pygame.K_t:
                        self.click_state = TARGET

                    # B, switch click state to place barrier nodes
                    elif event.key == pygame.K_b:
                        self.click_state = BARRIER

                    # W, switch click state to place barrier nodes TODO: decide how to handle this with adding the weight
                    elif event.key == pygame.K_w:
                        pass

                    # M, generate random maze of barriers
                    elif event.key == pygame.K_m:
                        # TODO: implement random maze generation
                        pass

                    # else, some other key was pressed

        pygame.quit()


def main():
    # TODO: make different size presets
    graph = Visualizer(25, 40)
    graph.run()


if __name__ == "__main__":
    main()