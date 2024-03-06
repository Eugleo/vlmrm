import numpy as np
import logging

try:
    import pygame
except ImportError as e:
    logging.warn(
        "pygame is not installed; to enable pygame features, run `pip install gymnasium[box2d]`"
    )

def parse_qsteps(stepstr):
    """Parse a string of 'wasd ' steps formatted like '10wd5a2 2s' into a list of steps"""
    if len(stepstr) == 1:
        return [stepstr]
    stepl = list(stepstr)[::-1]
    steps = []
    while stepl:
        num = ""
        while stepl and stepl[-1].isdigit():
            num += stepl.pop()
        if num == "":
            num = "0"
        if not stepl:  # ends in a number with no following step
            return steps
        st = ""
        while stepl and not stepl[-1].isdigit():
            st += stepl.pop()
        steps.extend([st] * int(num))
    return steps

def register_input(action, qsteps=None, render_mode="human"):
    quit = False
    restart = False
    if render_mode == "human":
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    action[0] = +1.0
                if event.key == pygame.K_UP:
                    action[1] = +1.0
                if event.key == pygame.K_DOWN:
                    action[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    action[0] = 0
                if event.key == pygame.K_RIGHT:
                    action[0] = 0
                if event.key == pygame.K_UP:
                    action[1] = 0
                if event.key == pygame.K_DOWN:
                    action[2] = 0
            if event.type == pygame.QUIT:
                quit = True
    elif render_mode == "ansi":
        if qsteps:
            input_str = qsteps.pop(0)
        else:
            # take wasd input
            qsteps.extend(parse_qsteps(input(">")))
            if qsteps:
                input_str = qsteps.pop(0)
            else:
                input_str = ""
        if "w" in input_str:
            action[1] = +1.0
        else:
            action[1] = 0
        if "a" in input_str:
            action[0] = -1.0
        else:
            action[0] = 0
        if "s" in input_str:
            action[2] = +0.8
        else:
            action[2] = 0
        if "d" in input_str:
            action[0] = +1.0
        else:
            action[0] = 0
        if "r" in input_str:
            restart = True
        if "q" in input_str:
            quit = True
    else:
        raise ValueError(f"Invalid render_mode: {render_mode} (must be 'human' or 'ansi')")
    return quit, restart
    
def main_loop(env_class, *args, **kwargs):
    env = env_class(*args, **kwargs)

    # check if the action space is discrete or continuous
    if hasattr(env.action_space, "n"):
        # if discrete, one-hot encode the action
        action = np.zeros(env.action_space.n)
    else:
        action = np.zeros(env.action_space.shape)
    if env.render_mode == "ansi":
        qsteps = []
    else:
        qsteps = None

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            quit, restart = register_input(action, qsteps=qsteps, render_mode=env.render_mode)
            if env.render_mode == "ansi" and not qsteps:
                env.render()
            s, r, terminated, truncated, info = env.step(action)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in action]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
