# based on https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/car_racing.py

import itertools
import math
import os
import random
from typing import Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import (
    ENGINE_POWER,
    FRICTION_LIMIT,
    SIZE,
    WHEEL_MOMENT_OF_INERTIA,
)
from gymnasium.envs.box2d.car_dynamics import Car as GymCar
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

from vlmrm.envs.box2d.maze_core import Maze, Edge, MAZE_W, MAZE_H, rgb_to_shell_code
from vlmrm.envs.box2d.keyboard_input_core import main_loop

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e

# maze parameters
EDGE_LEN = 2  # how long each edge is after rendering (in tiles)

FLOWER_COLOR = (255, 0, 255)
FRICTION_LIMIT_GRASS = FRICTION_LIMIT * 0.6
GRASS_DRAG = 100

SCALE = 2.0  # Track scale
PLAYFIELD = 50 * max(MAZE_W, MAZE_H) / SCALE  # 4000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 10  # 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_WIDTH = 40 / SCALE
GRASS_DIM = PLAYFIELD / 20.0
TRACK_DETAIL_STEP = 21 / SCALE
MAX_SHAPE_DIM = (
    max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE
)


STATE_W = 600  # int(TRACK_WIDTH) * MAZE_W  # 600  # 96  # less than Atari 160x192
STATE_H = 400  # int(TRACK_WIDTH) * MAZE_H  # 400  # 96
VIDEO_W = 600  # int(TRACK_WIDTH) * MAZE_W # 600  # 600
VIDEO_H = 400  # int(TRACK_WIDTH) * MAZE_H # 400  # 400
WINDOW_W = 1000
WINDOW_H = 800


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)


class Car(GymCar):
    def step(self, dt):
        """Copied and pasted from car_dynamics.py so that the friction limit can be changed"""
        for w in self.wheels:
            # Steer each wheel
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = dir * min(50.0 * val, 3.0)

            # Position => friction_limit
            grass = True
            friction_limit = FRICTION_LIMIT_GRASS  # Grass friction if no tile
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False

            # Force
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]  # forward speed
            vs = side[0] * v[0] + side[1] * v[1]  # side speed

            # WHEEL_MOMENT_OF_INERTIA*np.square(w.omega)/2 = E -- energy
            # WHEEL_MOMENT_OF_INERTIA*w.omega * domega/dt = dE/dt = W -- power
            # domega = dt*W/WHEEL_MOMENT_OF_INERTIA/w.omega

            # add small coef not to divide by zero
            w.omega += (
                dt
                * ENGINE_POWER
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15  # radians per second
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)  # low speed => same as = 0
                w.omega += dir * val

            # here we want it to be very disfavorable to cross the grass
            if grass:
                dir = -np.sign(w.omega)
                val = min(abs(w.omega), abs(GRASS_DRAG))
                w.omega += dir * GRASS_DRAG

            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad  # rotating wheel speed
            f_force = -vf + vr  # force direction is direction of speed difference
            p_force = -vs

            # Physically correct is to always apply friction_limit until speed is equal.
            # But dt is finite, that will lead to oscillations if difference is already near zero.

            # Random coefficient to cut oscillations in few steps (have no effect on friction_limit)
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            # Skid trace
            if abs(force) > 2.0 * friction_limit:
                if (
                    w.skid_particle
                    and w.skid_particle.grass == grass
                    and len(w.skid_particle.poly) < 30
                ):
                    w.skid_particle.poly.append((w.position[0], w.position[1]))
                elif w.skid_start is None:
                    w.skid_start = w.position
                else:
                    w.skid_particle = self._create_particle(
                        w.skid_start, w.position, grass
                    )
                    w.skid_start = None
            else:
                w.skid_start = None
                w.skid_particle = None

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit  # Correct physics here
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )


class RoadBuilder:
    def __init__(self, origin, edges=[]):
        self.origin = origin
        self.edges = edges

    def add_road(self, direction: Literal["up", "down", "left", "right"]):
        x, y = self.origin
        if direction == "up":
            new_origin = (x, y + 1)
            new_edge = Edge(x, y, "N")
        elif direction == "down":
            new_origin = (x, y - 1)
            new_edge = Edge(x, y - 1, "N")
        elif direction == "right":
            new_origin = (x + 1, y)
            new_edge = Edge(x, y, "E")
        elif direction == "left":
            new_origin = (x - 1, y)
            new_edge = Edge(x - 1, y, "E")
        self.edges.append(new_edge)
        return RoadBuilder(new_origin, self.edges)

class TestingGround:
    def __init__(self, spacing=5, origin=(0, 0)) -> None:
        self.secret_edges = []  # for compatibility with Maze
        self.edges = []

        ox, oy = origin

        # Crossroad
        b = RoadBuilder(origin).add_road("up")
        b.add_road("left")
        b.add_road("right")
        b.add_road("up")
        self.edges += b.edges

        # Right turn
        b = RoadBuilder((ox + 1 * spacing, oy)).add_road("up").add_road("right")
        self.edges += b.edges

        # Left turn
        b = RoadBuilder((ox + 2 * spacing, oy)).add_road("up").add_road("left")
        self.edges += b.edges

        # T-junction
        b = RoadBuilder((ox + 3 * spacing, oy)).add_road("up")
        b.add_road("left")
        b.add_road("right")
        self.edges += b.edges

        # U-turn
        b = (
            RoadBuilder((ox + 4 * spacing, oy))
            .add_road("up")
            .add_road("right")
            .add_road("down")
        )
        self.edges += b.edges

        # Straight road
        b = (
            RoadBuilder((ox + 5 * spacing, oy))
            .add_road("up")
            .add_road("up")
            .add_road("up")
        )

        # Grid
        size = 3
        for i in range(size + 1):
            bv = RoadBuilder((ox + 6 * spacing + i, oy))
            bh = RoadBuilder((ox + 6 * spacing, oy + i))
            for _ in range(size):
                bv = bv.add_road("up")
                bh = bh.add_road("right")
            self.edges += bv.edges + bh.edges


class ObstacleCourse(gym.Env, EzPickle):
    """
    ## Description
    The easiest control task to learn from pixels - a top-down
    racing environment. The generated track is random every episode.

    Some indicators are shown at the bottom of the window along with the
    state RGB buffer. From left to right: true speed, four ABS sensors,
    steering wheel position, and gyroscope.
    To play yourself (it's rather fast for humans), type:
    ```shell
    python gymnasium/envs/box2d/car_racing.py
    ```
    Remember: it's a powerful rear-wheel drive car - don't press the accelerator
    and turn at the same time.

    ## Action Space
    If continuous there are 3 actions :
    - 0: steering, -1 is full left, +1 is full right
    - 1: gas
    - 2: breaking

    If discrete there are 5 actions:
    - 0: do nothing
    - 1: steer left
    - 2: steer right
    - 3: gas
    - 4: brake

    ## Observation Space

    A top-down 96x96 RGB image of the car and race track.

    ## Rewards
    The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
     visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

    ## Starting State
    The car starts at rest in the center of the road.

    ## Episode Termination
    The episode finishes when all the tiles are visited. The car can also go outside the playfield -
     that is, far off the track, in which case it will receive -100 reward and die.

    ## Arguments

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v2", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v2>>>>>

    ```

    * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
     the agent before a lap is considered complete.

    * `domain_randomize=False` enables the domain randomized variant of the environment.
     In this scenario, the background and track colours are different on every reset.

    * `continuous=True` converts the environment to use discrete action space.
     The discrete action space has 5 actions: [do nothing, left, right, gas, brake].

    ## Reset Arguments

    Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
    Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
    `domain_randomize` must be `True` on init for this argument to work.

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("CarRacing-v2", domain_randomize=True)

    # normal reset, this changes the colour scheme by default
    >>> obs, _ = env.reset()

    # reset with colour scheme change
    >>> randomize_obs, _ = env.reset(options={"randomize": True})

    # reset with no colour scheme change
    >>> non_random_obs, _ = env.reset(options={"randomize": False})

    ```

    ## Version History
    - v1: Change track completion logic and add domain randomization (0.24.0)
    - v0: Original version

    ## References
    - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.

    ## Credits
    Created by Oleg Klimov
    """

    metadata = {
        "render_modes": [
            "human",
            "ansi",
            "rgb_array",
            "state_pixels",
        ],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car: Optional[Car] = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(5)
            # do nothing, left, right, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        assert self.car is not None
        self.car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert (
            self.domain_randomize
        ), "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track(self):
        maze = Maze()
        roadpieces = self._generate_roadpieces(maze.edges, include_ends=True)
        secret_roadpieces = self._generate_roadpieces(
            maze.secret_edges, include_ends=False
        )
        self.track = sorted(
            [
                (x * TRACK_WIDTH, y * TRACK_WIDTH)
                for x, y in roadpieces + secret_roadpieces
            ]
        )
        self.road_poly = [
            self._roadpiece_to_poly(r, is_secret=False) for r in roadpieces
        ] + [self._roadpiece_to_poly(r, is_secret=True) for r in secret_roadpieces]

    def _generate_roadpieces(self, edges, include_ends=False):
        result = set()
        for edge in edges:
            idxs = range(EDGE_LEN + 1) if include_ends else range(1, EDGE_LEN)
            for d in idxs:
                dx = d if edge.direction == "E" else 0
                dy = d if edge.direction == "N" else 0
                result.add(
                    (
                        edge.x * EDGE_LEN + dx - GRASS_DIM * 3 / 4,
                        edge.y * EDGE_LEN + dy - GRASS_DIM * 3 / 4,
                    )
                )
        return list(result)

    def _roadpiece_to_poly(self, roadpiece, is_secret):
        x, y = roadpiece
        w = TRACK_WIDTH
        self.fd_tile.shape = polygonShape(box=(w / 2, w / 2, (x * w, y * w), 0))
        vertices = self.fd_tile.shape.vertices

        if is_secret:
            color = np.array(FLOWER_COLOR)
        else:
            color = self.road_color + 0.02 * (random.random()) * 255

        t = self.world.CreateStaticBody(fixtures=self.fd_tile)
        t.userData = t
        t.color = color
        t.road_visited = False
        t.road_friction = 1.0
        t.idx = 10
        t.fixtures[0].sensor = True

        return (vertices, color)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        self._create_track()

        self.car = Car(self.world, 0, init_x=self.track[0][0], init_y=self.track[0][1])

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(
        self, action: Union[np.ndarray, int], camera=None
    ) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.car is not None
        if action is not None:
            if self.continuous:
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                # Truncation due to finishing lap
                # This should not be treated as a failure
                # but like a timeout
                truncated = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward = -100

        if self.render_mode == "human":
            self.render(camera)
        return self.state, step_reward, terminated, truncated, {}

    def render(self, camera=None):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode, camera)

    def _render(self, mode: str, camera=None):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        if ZOOM_FOLLOW:
            angle = 0  # -self.car.hull.angle
            # Animating first second zoom.
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + 0.5 * ZOOM * SCALE * min(
                self.t, 1
            )
            scroll_x = -(self.car.hull.position[0]) * zoom
            scroll_y = -(self.car.hull.position[1]) * zoom
        else:
            angle = 0
            zoom = ZOOM * SCALE * 0.1
            # center the x coordinate on the middle of the maze
            if camera is None:
                scroll_x = -MAZE_W * TRACK_WIDTH / 2 * 2 * zoom
                scroll_y = 0
            else:
                scroll_x, scroll_y = camera

        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "ansi":
            line_len = 120
            full_block = "█"

            img = self._create_image_array(self.surf, (STATE_W, STATE_H))
            scaledown = 1 << (int(img.shape[0] / line_len) - 1).bit_length()
            img_pool = np.array(
                [
                    [
                        [
                            int(np.mean(img[i : i + scaledown, j : j + scaledown, k]))
                            for k in range(0, img.shape[2])
                        ]
                        for j in range(0, img.shape[1], scaledown)
                    ]
                    for i in range(0, img.shape[0], scaledown)
                ]
            )
            # use rgb_to_shell_code to convert each triple of rgb values in img_pool to a shell code
            shell_codes = "\n".join(
                [
                    "".join(
                        [
                            f"{rgb_to_shell_code(img_pool[i, j])}{full_block}"
                            for j in range(0, img_pool.shape[1])
                        ]
                    )
                    for i in range(0, img_pool.shape[0])
                ]
            ) + rgb_to_shell_code(None)
            print("scaledown:", scaledown)
            print(shell_codes)
            return img

        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )

        grass_texture = pygame.image.load(
            os.path.join(os.path.dirname(__file__), "grass.jpg")
        )
        # grass_texture = pygame.transform.scale(
        #     grass_texture, (int(2 * bounds * zoom), int(2 * bounds * zoom))
        # )
        # self.surf.blit(grass_texture, (2.4*bounds, 0.9*bounds))

        for poly in grass:
            poly = [(p[0], p[1]) for p in poly]
            assert len(poly) == 4
            poly_width = abs(poly[0][0] - poly[1][0])
            poly_height = abs(poly[0][1] - poly[3][1])
            grass_texture = pygame.transform.scale(
                grass_texture, (2 * int(poly_width * zoom), 2 * int(poly_height * zoom))
            )
            self.surf.blit(
                grass_texture,
                (
                    poly[0][0] * zoom + translation[0],
                    poly[0][1] * zoom + translation[1],
                ),
            )

            # self._draw_colored_polygon(
            #   self.surf, poly, self.grass_color, zoom, translation, angle
            # )

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert self.car is not None
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            self.car.wheels[0].omega,
            vertical_ind(7, 0.01 * self.car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[1].omega,
            vertical_ind(8, 0.01 * self.car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            self.car.wheels[2].omega,
            vertical_ind(9, 0.01 * self.car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            self.car.wheels[3].omega,
            vertical_ind(10, 0.01 * self.car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            self.car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            self.car.hull.angularVelocity,
            horiz_ind(30, -0.8 * self.car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_colored_polygon(
        self, surface, poly, color, zoom, translation, angle, clip=True
    ):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly
        ]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()


if __name__ == "__main__":
    main_loop(ObstacleCourse, "ansi")
