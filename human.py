import pygame
from hackatari import HackAtari

def main(all_black_cars: bool = False):
    pygame.init()

    #mods = ["all_black_cars"] if all_black_cars else []
    mods = []

    env = HackAtari(
        env_name="Freeway",
        modifs=mods,
        render_mode="human",
    )

    obs, info = env.reset()
    clock = pygame.time.Clock()
    running = True
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = 0  # NOOP

        # Freeway actions (HackAtari / ALE standard)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()
            total_reward = 0.0

        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main(all_black_cars=True)
