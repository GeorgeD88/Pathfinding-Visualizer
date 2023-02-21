# data structures
from queue import PriorityQueue  # for A*
from collections import defaultdict, deque  # for BFS & DFS
from pqdict import pqdict  # for Dijkstra's

# random generation
from random import seed, random, randint
from time import time

# user interface
import pygame
import ptext

# miscellaneous
from typing import Generator
from math import inf

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
PASTEL_YELLOW = (255, 240, 145)

GRID_LINE = GRAY
UNWEIGHTED = WHITE
WEIGHTED = PASTEL_YELLOW  # make it very light yellow
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
MAX_NODE_WEIGHT = 9  # max weight of weighted nodes


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

    def make_unweighted(self):
        """ Set node to weighted node. """
        self.state = UNWEIGHTED
        self.weight = 0

    def make_weighted(self, weight: int):
        """ Set node to weighted node. """
        self.state = WEIGHTED
        self.weight = weight

    def is_barrier(self) -> bool:
        """ Returns whether node is a barrier. """
        return self.state == BARRIER

    def is_empty(self) -> bool:
        """ Returns whether node is a traversable/empty node (not barrier). """
        return self.state != BARRIER

    def is_weighted(self) -> bool:
        """ Returns whether node is a weighted node. """
        return 0 < self.weight < inf


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
            pygame.K_3: self.dijkstra,
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
                    self.get_node(r, c).state = WEIGHTED if self.get_node(r, c).is_weighted() else UNWEIGHTED

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

    def draw_node(self, node: Node):
        """ Draws given node onto pygame window. """
        pygame.draw.rect(self.win, node.state, (node.x, node.y, self.cell_size, self.cell_size))

    def draw_node_grid(self, node: Node):
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

    def update_node(self, node: Node, wait: float = WAIT):
        """ Draws given node and its grid, then updates display (more efficient then redrawing whole window every time). """
        self.draw_node(node)

        # draw weight if it's a weighted node
        if node.is_weighted():
            self.draw_weight(node)

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
                if node.is_weighted():
                    self.draw_weight(node)

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

    def place_weighted(self, node: Node, max_weight: int = MAX_NODE_WEIGHT):
        """ Place weighted node and redraw (also handles conflicts). """
        # as long as node isn't start or target, draws barrier
        if node != self.start and node != self.target:
            seed(time())
            node.make_weighted(randint(1, max_weight))  # set to random weight
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
        print('running BFS...')
        queue = deque([self.start])  # append and popleft
        discovered = {self.start}  # keeps track of discovered nodes
        parents = {self.start: None}  # keeps track of every node's parent

        # while there's still nodes to check
        while len(queue) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # pops next node to process
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
        print('running DFS...')
        stack = deque([self.start])  # append and pop
        discovered = {self.start}  # keeps track of discovered nodes
        parents = {self.start: None}  # keeps track of every node's parent

        # while there's still nodes to check
        while len(stack) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # pops next node to process
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


    def dijkstra(self):
        """ Runs Dijkstra's. """
        print('running Dijkstra\'s...')
        distances, parents = {}, defaultdict(lambda: None, {self.start: None})
        visited = set()  # keeps track of processed nodes
        ipq = pqdict({self.start: 0})  # indexed priority queue

        # continue running while the MST is not complete and there are still edges in the queue to process
        while len(ipq) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            # pops next node to process
            curr, distance = ipq.popitem()  # pop next smallest edge to process
            distances[curr] = distance

            # if target node is found, traces paths back to start
            if curr == self.target:
                curr.state = TARGET  # color curr back to target
                self.update_node(curr, wait=0.15)
                self.trace_path(parents)
                return

            visited.add(curr)  # set current node as visited

            # queue adjacent edges
            for adj in self.adjacent_nodes(curr):
                # add adjacent edge if its node hasn't been visited
                if adj in visited:
                    continue

                # color the adjacent node
                if adj != self.target and not adj.is_barrier():
                    adj.state = QUEUED
                    self.update_node(adj, wait=0.15)

                # simply add edge if node isn't already queue
                if adj not in ipq:
                    parents[adj] = curr
                    ipq.additem(adj, distance + adj.weight)

                # else the node is already queued so relax the edge
                elif distance + adj.weight < ipq[adj]:
                    parents[adj] = curr
                    ipq.updateitem(adj, distance + adj.weight)

            # set current state to passed after it finishes processing
            if curr != self.start:
                curr.state = PASSED
                self.update_node(curr, wait=0.15)

    def a_star(self):
        """ Runs A* 😍😍😍 for target node and then traces path back to start (weighted). """
        print('running A*...')

    # weighted graphs/maze generation algorithms
    def generate_weighted_grid(self, density: float = 0.3, max_weight: int = MAX_NODE_WEIGHT):
        """ Randomly fills grid with weighted nodes. """
        seed(time())  # set the seed to the current time ensure randomness
        for node in self.loop_all_nodes():
            # skip barrier nodes
            if node.state == BARRIER:
                continue

            # add weighted node
            if random() < density:  # generate nodes by density
                node.make_weighted(randint(1, max_weight))  # random weight
                self.update_node(node)
            else:
                node.make_unweighted()
                self.update_node(node)

    def generate_maze_prims(self):
        """ Generates random maze of barrier nodes using Prim's algorithm. """
        pass

    def generate_maze_kruskals(self):
        """ Generates random maze of barrier nodes using Kruskal's algorithm. """
        pass

    # helper functions
    def gen_matrix(self, default=None) -> list[list]:
        """ Generates a matrix the size of the grid. """
        matrix = []

        for _ in range(self.rows):
            matrix.append([])  # append new list for next row
            if default is not None:
                for _ in range(self.cols):
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

    def loop_all_nodes(self) -> Generator[Node, None, None]:
        """ Generator that yields all nodes in the grid (gets rid of need for 2 loops). """
        for r in range(self.rows):
            for c in range(self.cols):
                yield self.get_node(r, c)

    def loop_all_coords(self) -> Generator[tuple[int, int], None, None]:
        """ Generator that yields coords of all nodes in the grid (gets rid of need for 2 loops). """
        for r in range(self.rows):
            for c in range(self.cols):
                yield (r, c)

    def draw_weight(self, node: Node):
        """ Draws node's weight onto pygame window. """
        ptext.draw(str(node.weight), centerx=node.x+self.cell_size//2, centery=node.y+self.cell_size//2, fontsize=int(self.cell_size/4*3), color=BLACK)#3:2

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
                    elif self.click_state == WEIGHTED:  # place weighted node
                        self.place_weighted(node)
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

                    # W, switch click state to place weighted nodes TODO: try to make it so that you can place individual weighted nodes at some point
                    # W, randomly fill grid with weighted nodes
                    elif event.key == pygame.K_w:
                        self.click_state = WEIGHTED
                        # self.generate_weighted_grid()

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
