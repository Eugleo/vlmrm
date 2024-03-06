import itertools
import random
from typing import Literal, Tuple, Dict, Union


MAZE_W = 8
MAZE_H = 8
MAZE_BAD_PATHS = 5  # number of walls to secretly delete

def rgb_to_shell_code(rgb):
    """Convert an RGB tuple to a shell color code. If rgb is None, reset the color to default."""
    if rgb is None:
        return "\033[0m"
    r, g, b = rgb
    return f"\033[38;2;{r};{g};{b}m"

def to_color(text, color):
    """Return text in the given color"""
    return rgb_to_shell_code(color) + text + rgb_to_shell_code(None)


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tree = None

    def __repr__(self):
        return f"Cell({self.x}, {self.y}, {self.tree})"


class Edge:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def __eq__(self, other):
        return (
            self.x == other.x
            and self.y == other.y
            and self.direction == other.direction
        )

    def __hash__(self):
        return hash((self.x, self.y, self.direction))

    def __repr__(self) -> str:
        # print as an arrow between two cells
        if self.direction == "N":
            return f"({self.x}, {self.y}) -> ({self.x}, {self.y+1})"
        elif self.direction == "E":
            return f"({self.x}, {self.y}) -> ({self.x+1}, {self.y})"


class Tree:
    def __init__(self, label, cells):
        self.cells = cells
        self.label = label

        for cell in cells:
            if cell.tree is not None:
                raise ValueError("Cell already in a tree")
            cell.tree = label

    def __repr__(self):
        return f"Tree({self.label}, {self.cells})"


Tile = Union[Literal["road"], Literal["grass"], Literal["flowers"]]


class Maze:
    def __init__(self):
        # use kruskal's algorithm

        # generate grid of cells
        self.cells = [[Cell(x, y) for y in range(MAZE_H)] for x in range(MAZE_W)]
        # generate list of edges
        east_edge_rows = [
            [Edge(x, y, "E") for x in range(MAZE_W - 1)] for y in range(MAZE_H)
        ]
        north_edge_rows = [
            [Edge(x, y, "N") for x in range(MAZE_W)] for y in range(MAZE_H - 1)
        ]
        # interleave the sublists
        self.all_rows = [
            [
                v
                for v in itertools.chain(*itertools.zip_longest(north_row, east_row))
                if v is not None
            ]
            for north_row, east_row in zip(north_edge_rows, east_edge_rows)
        ]
        # print("all edge rows\n", self.all_rows)
        self.all_edges = list(itertools.chain(*self.all_rows))
        # self.all_edges = list(itertools.chain(*[[v for v in itertools.chain(*itertools.zip_longest(north_row, east_row)) if v is not None] for north_row, east_row in zip(north_edge_rows, east_edge_rows)]))
        # print("All edges:", self.all_edges)

        self.possible_edges = [edge for edge in self.all_edges]
        self.trees = {}

        # start out with each cell in its own tree, and no selected edges yet
        for idx, cell in enumerate(itertools.chain(*self.cells)):
            self.trees[idx] = Tree(idx, [cell])

        # holes in the walls of the maze (start out with a grid of walls)
        self.edges = set()

        while len(self.trees) > 1:
            if len(self.possible_edges) == 0:
                print("trees:", self.trees)
                print("edges:", self.edges)
                raise ValueError("No possible edges left")

            # choose a random edge to remove from the list of possible edges and possibly add to the maze
            edge = self.possible_edges.pop(random.randrange(len(self.possible_edges)))

            # check whether the two cells the edge connects are in different trees
            cell1, cell2 = self.endpoints(edge)
            if cell1.tree != cell2.tree:
                # merge the two trees
                self.merge(self.trees[cell1.tree], self.trees[cell2.tree])
                # add the edge to the list of edges
                self.edges.add(edge)

        # "hidden" holes in the walls of the maze
        self.possible_secret_edges = [
            edge for edge in self.all_edges if edge not in self.edges
        ]
        self.secret_edges = set()
        while len(self.secret_edges) < MAZE_BAD_PATHS:
            edge = self.possible_secret_edges.pop(
                random.randrange(len(self.possible_secret_edges))
            )
            self.secret_edges.add(edge)

    def endpoints(self, edge):
        # return the two cells that the edge connects
        if edge.direction == "N":
            return self.cells[edge.x][edge.y], self.cells[edge.x][edge.y + 1]
        elif edge.direction == "E":
            return self.cells[edge.x][edge.y], self.cells[edge.x + 1][edge.y]
        else:
            raise ValueError("Invalid direction")

    def merge(self, tree1, tree2):
        # merge tree2 into tree1
        tree1.cells.extend(tree2.cells)
        for cell in tree2.cells:
            cell.tree = tree1.label
        del self.trees[tree2.label]

    def __repr__(self):
        out = ""

        # write the maze
        # write the top row of walls
        out += " "
        out += "_" * (2 * MAZE_W - 1)
        out += " \n"

        for row in self.all_rows[:-1]:
            out += "|"
            for edge in row:
                if edge.direction == "N":
                    if edge in self.edges:
                        out += " "
                    elif edge in self.secret_edges:
                        out += "."
                    else:
                        out += "_"
                elif edge.direction == "E":
                    if edge in self.edges:
                        out += "_"
                    elif edge in self.secret_edges:
                        out += ":"
                    else:
                        out += "|"
                else:
                    raise ValueError("Invalid direction")
            out += "|\n"
        out += "|"
        for edge in self.all_rows[-1]:
            if edge.direction == "N":
                if edge in self.edges:
                    out += "_"
                elif edge in self.secret_edges:
                    out += "."
                else:
                    out += "_"
            elif edge.direction == "E":
                if edge in self.edges:
                    out += "_"
                elif edge in self.secret_edges:
                    out += "'"
                else:
                    out += "|"
            else:
                raise ValueError("Invalid direction")
        out += "|"

        return out
    
    def _default_pcwcsc_color(self):
        path_color = (0, 255, 0)
        wall_color = (255, 0, 0)
        secret_color = (0, 0, 255)
        full_block = "█"

        def color_block(color):
            return to_color(full_block, color)
        
        return color_block(path_color), color_block(wall_color), color_block(secret_color)

    def _default_pcwcsc_bw(self):
        return (" ", "█", "▒")

    def to_string(self, blocks=None):
        if blocks is None:
            pc, wc, sc = self._default_pcwcsc_bw()
        else:
            pc, wc, sc = blocks

        out = ""

        # write the maze
        # write the top row of walls
        out += wc * (2 * MAZE_W + 1)
        out += "\n"

        for row in self.all_rows[:-1]:
            out += wc
        
            for edge in row:
                if edge.direction == "N":
                        out += pc
                elif edge.direction == "E":
                    if edge in self.edges:
                        out += pc
                    elif edge in self.secret_edges:
                        out += sc
                    else:
                        out += wc
                else:
                    raise ValueError("Invalid direction")
            out += wc + "\n" + wc
            for edge in row:
                if edge.direction == "N":
                    if edge in self.edges:
                        out += pc
                    elif edge in self.secret_edges:
                        out += sc
                    else:
                        out += wc
                elif edge.direction == "E":
                    if edge in self.edges:
                        out += wc
                    elif edge in self.secret_edges:
                        out += wc
                    else:
                        out += wc
                else:
                    raise ValueError("Invalid direction")      

            out += wc + "\n"
        out += wc
        for edge in self.all_rows[-1]:
            if edge.direction == "N":
                out += pc
            elif edge.direction == "E":
                if edge in self.edges:
                    out += pc
                elif edge in self.secret_edges:
                    out += sc
                else:
                    out += wc
            else:
                raise ValueError("Invalid direction")
        out += wc + "\n" + wc * (2 * MAZE_W + 1)

        return out

    def to_branch_list(self):
        """return a list of branches, where each branch is a list of cells connected by edges"""
        start_cell = self.cells[0][0]
        branches = [[]]
        # the cell queue consists of tuples of (cell, branch index) which we need to add to branches
        cell_queue = [(start_cell, 0)]

        while cell_queue:
            cell, idx = cell_queue.pop()
            branches[idx].append(cell)
            # find the edges connected to the cell by checking self.edges for the
            # 4 possible edges that could be connected to cell
            connected_edges = [
                edge
                for edge in [
                    Edge(cell.x, cell.y, "N"),
                    Edge(cell.x, cell.y, "E"),
                    Edge(cell.x - 1, cell.y, "E"),
                    Edge(cell.x, cell.y - 1, "N"),
                ]
                if edge in self.edges
            ]
            # print(f"connected edges to cell {cell}:", connected_edges)

            # add all the following cells to the queue, adding a new branch for each one beyond the first
            # each branch starts with the cell where it diverged from the previous branch,
            # so that some cells will be in multiple branches
            new_idx = idx
            while connected_edges:
                cell1, cell2 = self.endpoints(connected_edges.pop())
                new_cell = cell1 if cell1 != cell else cell2
                # check whether new_cell is already in a branch
                for branch in branches:
                    if new_cell in branch:
                        break
                else:
                    cell_queue.append((new_cell, new_idx))
                    if new_idx != idx:
                        branches.append([new_cell])
                    new_idx = len(branches)

        return branches

    def display_branches(self):
        """Display the branches just like the maze, but with different colors for each branch"""
        branches = [set(branch) for branch in self.to_branch_list()][:1]
        # print("branches:", branches)

        def edge_to_color(edge):
            cell1, cell2 = self.endpoints(edge)

            for idx, branch in enumerate(branches):
                if cell1 in branch:
                    bg = 41 + (idx % 7)
                    fg = 30
                    return "\x1b[%sm" % f"0;{fg};{bg}"
            return "\033[0m"

        out = ""

        # write the maze
        # write the top row of walls
        out += " "
        out += "_" * (2 * MAZE_W - 1)
        out += "\n"

        for row in self.all_rows[:-1]:
            out += "|"
            for edge in row:
                out += edge_to_color(edge)
                if edge.direction == "N":
                    if edge in self.edges:
                        out += " "
                    elif edge in self.secret_edges:
                        out += "."
                    else:
                        out += "_"
                elif edge.direction == "E":
                    if edge in self.edges:
                        out += "_"
                    elif edge in self.secret_edges:
                        out += "\033[0m"
                        out += ":"
                    else:
                        out += "\033[0m"
                        out += "|"
                else:
                    raise ValueError("Invalid direction")
            out += "\033[0m"
            out += "|\n"
        out += "|"
        for edge in self.all_rows[-1]:
            out += edge_to_color(edge)
            if edge.direction == "N":
                out += "_"
            elif edge.direction == "E":
                if edge in self.edges:
                    out += "_"
                elif edge in self.secret_edges:
                    out += "\033[0m"
                    out += ":"
                else:
                    out += "\033[0m"
                    out += "|"
            else:
                raise ValueError("Invalid direction")
        out += "\033[0m"
        out += "|"

        return out
