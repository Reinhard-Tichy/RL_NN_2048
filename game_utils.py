from random import randint, shuffle
import numpy as np
import random

# https://github.com/Mekire/console-2048/blob/master/console2048.py

def push_left(grid):
    # score 移动总得分，未移动则为-1
    # moved 表示是否发生移动
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):  # 遍历各行
        # i: 本行中最左侧第一个空格的坐标
        # last: 上一个非空格的分数
        # 保证[0,i-1]中不存在空格
        i, last = 0, 0
        for j in range(columns):  # 从左到右顺序遍历该行各元素
            e = grid[k, j]
            if e:  # 若grid[k,j]不为空, 此时必有 e > 0
                if e == last:  # 若e与其前一个非空网格的分数相同
                    grid[k, i-1] += e  # 前一个网格的分数加e
                    score += e  # 总分也加e
                    last, moved = 0, True  # 刷新last, 即一次移动每个非空格最多只能被合并一次
                else:  # 若e存在，且与其前一个非空网格分数不同
                    moved |= (i != j)  # 将该值移动到i处
                    last = grid[k, i] = e
                    i += 1
        while i < columns:  # 将i及其之后的所有格均置为空
            grid[k, i] = 0
            i += 1
    return score if moved else -1


def push_right(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(rows):
        i = columns-1
        last = 0
        for j in range(columns-1, -1, -1):
            e = grid[k, j]
            if e:
                if e == last:
                    grid[k, i+1] += e
                    score += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[k, i] = e
                    i -= 1
        while i >= 0:
            grid[k, i] = 0
            i -= 1
    return score if moved else -1


def push_up(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = 0, 0
        for j in range(rows):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i-1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i += 1
        while i < rows:
            grid[i, k] = 0
            i += 1
    return score if moved else -1


def push_down(grid):
    moved, score = False, 0
    rows, columns = grid.shape[0], grid.shape[1]
    for k in range(columns):
        i, last = rows-1, 0
        for j in range(rows-1, -1, -1):
            e = grid[j, k]
            if e:
                if e == last:
                    score += e
                    grid[i+1, k] += e
                    last, moved = 0, True
                else:
                    moved |= (i != j)
                    last = grid[i, k] = e
                    i -= 1
        while i >= 0:
            grid[i, k] = 0
            i -= 1
    return score if moved else -1


def push(grid, direction):
    # [0,1,2,3]分别对应左、上、右、下
    if direction & 1:
        if direction & 2:  # 11
            score = push_down(grid)
        else:  # 01
            score = push_up(grid)
    else:
        if direction & 2:  # 10
            score = push_right(grid)
        else:  # 00
            score = push_left(grid)
    return score


def put_new_cell(grid):
    # grid: 4*4
    n = 0
    r = 0
    i_s = [0] * 16
    j_s = [0] * 16
    # 找所有空格的坐标
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not grid[i, j]:  # 若grid[i, j]==0
                i_s[n] = i
                j_s[n] = j
                n += 1
    if n > 0:
        r = randint(0, n-1)  # 从这些空格中随机选一个
        # 0.9的概率new cell的值为2, 0.1的概率为4
        grid[i_s[r], j_s[r]] = 2 if random.random() < 0.9 else 4
    return n


def any_possible_moves(grid):
    """Return True if there are any legal moves, and False otherwise."""
    rows = grid.shape[0]
    columns = grid.shape[1]
    for i in range(rows):
        for j in range(columns):
            e = grid[i, j]
            if not e:  # 有空格
                return True
            if j and e == grid[i, j-1]:  # 横向存在相邻相同格
                return True
            if i and e == grid[i-1, j]:  # 纵向存在相邻相同格
                return True
    return False


def prepare_next_turn(grid):
    """
    Spawn a new number on the grid; then return the result of
    any_possible_moves after this change has been made.
    """
    empties = put_new_cell(grid)
    # 若能放置或可移动则返回True, 否则返回False
    return empties > 1 or any_possible_moves(grid)


def print_grid(grid_array):
    """Print a pretty grid to the screen."""
    print("")
    wall = "+------" * grid_array.shape[1] + "+"
    print(wall)
    for i in range(grid_array.shape[0]):
        meat = "|".join("{:^6}".format(grid_array[i, j])
                        for j in range(grid_array.shape[1]))
        print("|{}|".format(meat))
        print(wall)

def find_empty_cell(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if(mat[i][j] == 0):
                count += 1
    return count

class Game:
    def __init__(self, cols=4, rows=4):
        self.grid_array = np.zeros(shape=(rows, cols), dtype='uint16')
        self.grid = self.grid_array
        for i in range(2):  # 初始随机放置2个grid
            put_new_cell(self.grid)
        self.score = 0
        self.end = False
        self.reward = 0

    def copy(self):
        rtn = Game(self.grid.shape[0], self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                rtn.grid[i, j] = self.grid[i, j]
        rtn.score = self.score
        rtn.end = self.end
        return rtn

    def max(self):
        # 找grid中的最大值
        m = 0
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] > m:
                    m = self.grid[i, j]
        return m

    def move(self, direction):
        # 成功移动返回1，否则返回0
        # 若无法再放置或移动，则游戏结束
        if direction & 1:
            if direction & 2:
                score = push_down(self.grid)  # 3
            else:
                score = push_up(self.grid)  # 1
        else:
            if direction & 2:
                score = push_right(self.grid)  # 2
            else:
                score = push_left(self.grid)  # 0
        if score == -1:
            return 0
        self.score += score
        if not prepare_next_turn(self.grid):
            self.end = True
        return 1

    def display(self):
        print_grid(self.grid_array)


def random_play(game):
    moves = [0, 1, 2, 3]
    while not game.end:
        # 每次随机选择一个可移动的方向，直到游戏结束
        shuffle(moves)
        for m in moves:
            if game.move(m):
                break
    return game.score
