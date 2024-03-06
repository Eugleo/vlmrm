from typing import Dict, Tuple
import numpy as np
from vlmrm.envs.box2d.obstacle_course import Maze
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from vlmrm.envs.box2d.keyboard_input_core import main_loop

# subclass FrozenLakeEnv to use Maze from maze_core to generate the map
class MazeTextEnv(FrozenLakeEnv):
    def __init__(self, render_mode: str = "ansi") -> None:
        self.maze = np.array([list(row) for row in Maze().to_string((" ", "#", "0")).split("\n")])
        super().__init__(desc=self.maze, is_slippery=False, render_mode=render_mode)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        state, reward, terminated, info = super().step(action)
        return state, reward, terminated, info

    def render(self, mode: str = "ansi") -> None:
        if mode == "ansi":
            print(self.maze)
        else:
            super().render(mode)


if __name__ == "__main__":
    main_loop(MazeTextEnv, render_mode="ansi")

