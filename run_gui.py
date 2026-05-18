import argparse
import os
import pickle
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame
from tkinter import Tk, filedialog

sys.path.insert(0, ".")

from agents.dqn_agent import DQNAgent
from agents.qlearning_agent import QLearningAgent
from environment.traffic_env import TrafficEnv

ACTION_LABELS = {
    0: "KEEP GREEN",
    1: "INCREASE GREEN (+5s)",
    2: "DECREASE GREEN (-5s)",
}

NEON_COLORS = [
    (56, 189, 248),   # Glowing Sky Blue
    (244, 63, 94),    # Glowing Pink
    (16, 185, 129),   # Glowing Emerald Green
    (245, 158, 11),   # Glowing Amber Orange
    (139, 92, 246),   # Glowing Violet
    (236, 72, 153),   # Glowing Hot Pink
    (20, 184, 166),   # Glowing Teal
]


class VisualCar:
    """
    Manages individual car particles in the GUI for realistic physical motion,
    queuing, deceleration, stopping, and crossing the intersection.
    """

    def __init__(self, id_val: int, direction: str, x: float, y: float, speed: float, color: Tuple[int, int, int]):
        self.id = id_val
        self.direction = direction
        self.x = x
        self.y = y
        self.speed = speed
        # Reduced max speed for a calm, ultra-safe urban school-zone driving pace
        self.max_speed = random.uniform(1.4, 2.0)
        self.color = color
        self.state = "approaching"  # "approaching", "waiting", "crossing", "departed"
        self.type = random.choice(["sedan", "suv", "sport"])

        if direction in ("N", "S"):
            self.w = 20
            self.h = 32
        else:
            self.w = 32
            self.h = 20

    def update(self, dt_sec: float, target_limit: Optional[float], has_green_light: bool):
        # Determine if we should slow down or stop
        if target_limit is not None:
            if self.direction == "N":
                dist = target_limit - self.y
            elif self.direction == "S":
                dist = self.y - target_limit
            elif self.direction == "W":
                dist = target_limit - self.x
            else:  # E
                dist = self.x - target_limit

            if dist < 4:
                self.speed = 0.0
                self.state = "waiting"
            elif dist < 70:
                # Smooth deceleration
                target_speed = (dist / 70.0) * self.max_speed
                if self.speed > target_speed:
                    self.speed = max(target_speed, self.speed - 0.20)
                else:
                    self.speed = min(target_speed, self.speed + 0.08)
                if self.speed < 0.1:
                    self.speed = 0.0
                    self.state = "waiting"
            else:
                self.speed = min(self.max_speed, self.speed + 0.08)
                self.state = "approaching"
        else:
            # Green light or crossing - accelerate freely!
            self.speed = min(self.max_speed, self.speed + 0.08)
            
            # Once we pass the stop line, we lock the state to crossing
            if self.direction == "N" and self.y > 228:
                self.state = "crossing"
            elif self.direction == "S" and self.y < 500:
                self.state = "crossing"
            elif self.direction == "W" and self.x > 448:
                self.state = "crossing"
            elif self.direction == "E" and self.x < 720:
                self.state = "crossing"

        # Apply movements
        if self.direction == "N":
            self.y += self.speed
            if self.y > 820:
                self.state = "departed"
        elif self.direction == "S":
            self.y -= self.speed
            if self.y < -60:
                self.state = "departed"
        elif self.direction == "W":
            self.x += self.speed
            if self.x > 1260:
                self.state = "departed"
        elif self.direction == "E":
            self.x -= self.speed
            if self.x < -60:
                self.state = "departed"

    def draw(self, screen: pygame.Surface):
        # Draw car shadow
        pygame.draw.rect(screen, (10, 15, 20), (self.x + 3, self.y + 3, self.w, self.h), border_radius=5)
        # Draw car body
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.w, self.h), border_radius=5)
        # Draw car body highlight/border
        pygame.draw.rect(screen, (255, 255, 255), (self.x, self.y, self.w, self.h), width=1, border_radius=5)

        # Draw roof and windows
        if self.direction in ("N", "S"):
            # Windshield (front)
            windshield_y = self.y + 8 if self.direction == "N" else self.y + self.h - 12
            pygame.draw.rect(screen, (30, 41, 59), (self.x + 3, windshield_y, self.w - 6, 4), border_radius=1)
            # Rear window
            rear_y = self.y + self.h - 10 if self.direction == "N" else self.y + 6
            pygame.draw.rect(screen, (30, 41, 59), (self.x + 3, rear_y, self.w - 6, 3), border_radius=1)
            # Roof
            pygame.draw.rect(screen, (20, 20, 25), (self.x + 3, min(windshield_y, rear_y) + 4, self.w - 6, self.h - 18), border_radius=2)
        else:
            # Windshield
            windshield_x = self.x + 8 if self.direction == "W" else self.x + self.w - 12
            pygame.draw.rect(screen, (30, 41, 59), (windshield_x, self.y + 3, 4, self.h - 6), border_radius=1)
            # Rear window
            rear_x = self.x + self.w - 10 if self.direction == "W" else self.x + 6
            pygame.draw.rect(screen, (30, 41, 59), (rear_x, self.y + 3, 3, self.h - 6), border_radius=1)
            # Roof
            pygame.draw.rect(screen, (20, 20, 25), (min(windshield_x, rear_x) + 4, self.y + 3, self.w - 18, self.h - 6), border_radius=2)

        # Sports stripes
        if self.type == "sport":
            stripe_color = (255, 255, 255) if self.color != (255, 255, 255) else (200, 30, 30)
            if self.direction in ("N", "S"):
                pygame.draw.line(screen, stripe_color, (self.x + self.w // 2 - 2, self.y), (self.x + self.w // 2 - 2, self.y + self.h), 1)
                pygame.draw.line(screen, stripe_color, (self.x + self.w // 2 + 1, self.y), (self.x + self.w // 2 + 1, self.y + self.h), 1)
            else:
                pygame.draw.line(screen, stripe_color, (self.x, self.y + self.h // 2 - 2), (self.x + self.w, self.y + self.h // 2 - 2), 1)
                pygame.draw.line(screen, stripe_color, (self.x, self.y + self.h // 2 + 1), (self.x + self.w, self.y + self.h // 2 + 1), 1)

    def draw_lights(self, screen: pygame.Surface, glow_surf: pygame.Surface):
        is_braking = (self.speed < 0.8 or self.state == "waiting")
        headlight_glow = (255, 253, 220, 50)
        brake_color = (239, 68, 68) if is_braking else (127, 29, 29)

        if self.direction == "N":
            h1, h2 = (self.x + 3, self.y + self.h - 2), (self.x + self.w - 3, self.y + self.h - 2)
            b1, b2 = (self.x + 3, self.y + 2), (self.x + self.w - 3, self.y + 2)
            beam1 = [h1, (self.x - 20, self.y + self.h + 80), (self.x + 15, self.y + self.h + 80)]
            beam2 = [h2, (self.x + self.w - 15, self.y + self.h + 80), (self.x + self.w + 20, self.y + self.h + 80)]
        elif self.direction == "S":
            h1, h2 = (self.x + 3, self.y + 2), (self.x + self.w - 3, self.y + 2)
            b1, b2 = (self.x + 3, self.y + self.h - 2), (self.x + self.w - 3, self.y + self.h - 2)
            beam1 = [h1, (self.x - 20, self.y - 80), (self.x + 15, self.y - 80)]
            beam2 = [h2, (self.x + self.w - 15, self.y - 80), (self.x + self.w + 20, self.y - 80)]
        elif self.direction == "W":
            h1, h2 = (self.x + self.w - 2, self.y + 3), (self.x + self.w - 2, self.y + self.h - 3)
            b1, b2 = (self.x + 2, self.y + 3), (self.x + 2, self.y + self.h - 3)
            beam1 = [h1, (self.x + self.w + 80, self.y - 20), (self.x + self.w + 80, self.y + 15)]
            beam2 = [h2, (self.x + self.w - 2, self.y + self.h - 3), (self.x + self.w + 80, self.y + self.h + 20)]
        else:  # E
            h1, h2 = (self.x + 2, self.y + 3), (self.x + 2, self.y + self.h - 3)
            b1, b2 = (self.x + self.w - 2, self.y + 3), (self.x + self.w - 2, self.y + self.h - 3)
            beam1 = [h1, (self.x - 80, self.y - 20), (self.x - 80, self.y + 15)]
            beam2 = [h2, (self.x - 80, self.y + self.h - 15), (self.x - 80, self.y + self.h + 20)]

        # Draw yellow light cones on glow surface
        pygame.draw.polygon(glow_surf, headlight_glow, beam1)
        pygame.draw.polygon(glow_surf, headlight_glow, beam2)

        # Draw headlights bulbs
        pygame.draw.circle(screen, (255, 255, 255), h1, 3)
        pygame.draw.circle(screen, (255, 255, 255), h2, 3)

        # Draw brake bulbs
        pygame.draw.circle(screen, brake_color, b1, 3)
        pygame.draw.circle(screen, brake_color, b2, 3)

        # Draw brake glow if braking
        if is_braking:
            pygame.draw.circle(glow_surf, (239, 68, 68, 100), b1, 8)
            pygame.draw.circle(glow_surf, (239, 68, 68, 100), b2, 8)


class TrafficSimulationGUI:
    def __init__(self, model_path: str, episodes: int, fps: int, step_interval_ms: int):
        self.model_path_dqn = model_path
        self.model_path_q = "models/q_table.pkl"
        self.episodes = episodes
        self.fps = fps
        self.step_interval_ms = step_interval_ms

        self.env = TrafficEnv(max_steps=200)
        self.agent = DQNAgent(
            state_size=4,
            action_size=3,
            learning_rate=0.0005,
            gamma=0.99,
            epsilon=0.0,
            epsilon_decay=1.0,
            use_double_dqn=True,
        )
        self.agent.epsilon = 0.0

        self.q_agent = QLearningAgent(epsilon=0.0, epsilon_decay=1.0, min_epsilon=0.0)
        self.q_table_loaded = self._try_load_q_table(self.model_path_q)
        self.dqn_loaded = self._try_load_dqn_model(self.model_path_dqn)

        self.controller_mode = "dqn" if self.dqn_loaded else "random"
        self.last_manual_action: Optional[int] = None
        self.paused = False

        self.episode = 1
        self.state = self.env.reset()
        self.done = False
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.last_action = 0
        self.global_step = 0
        self.last_step_ms = 0
        self.episode_history: List[float] = []
        self.status_message = "Modes: 1 DQN, 2 QL, 3 Random, 4 Manual, 5 Mixed."
        self.status_until_ms = 0

        # --- Premium visual variables ---
        self.cars_dict: Dict[str, List[VisualCar]] = {"N": [], "S": [], "W": [], "E": []}
        self.car_id_counter = 0

        self.pending_spawns = {"N": 0, "S": 0, "W": 0, "E": 0}
        self.spawn_timer = {"N": 0.0, "S": 0.0, "W": 0.0, "E": 0.0}

        self.step_elapsed_ms = 0.0
        self.light_ns = "RED"
        self.light_ew = "RED"

        # Adaptive Split Split State
        self.ns_split = 0.50

        self.served_to_release = {"N": 0, "S": 0, "W": 0, "E": 0}
        self.ns_release_triggered = False
        self.ew_release_triggered = False

        pygame.init()
        pygame.display.set_caption("Traffic RL - Live Premium Simulation")
        self.screen = pygame.display.set_mode((1200, 760))
        self.clock = pygame.time.Clock()

        # Premium sleek fonts
        self.font = pygame.font.SysFont("arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("arial", 12, bold=True)
        self.font_big = pygame.font.SysFont("arial", 22, bold=True)

        self._populate_initial_cars()

    def _try_load_dqn_model(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            self.agent.load(path)
            self.model_path_dqn = path
            return True
        except Exception:
            return False

    def _try_load_q_table(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                table = pickle.load(f)
            self.q_agent.Q.clear()
            for key, value in table.items():
                self.q_agent.Q[key] = np.array(value, dtype=np.float32)
            self.model_path_q = path
            return True
        except Exception:
            return False

    def _set_status(self, message: str, duration_ms: int = 4000):
        self.status_message = message
        self.status_until_ms = pygame.time.get_ticks() + duration_ms

    def _open_file_dialog(self, title: str, filters: List[Tuple[str, str]]) -> str:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=filters)
        root.destroy()
        return selected

    def _load_dqn_from_dialog(self):
        selected = self._open_file_dialog(
            "Choose DQN model file",
            [("PyTorch model", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not selected:
            self._set_status("DQN model selection cancelled.")
            return

        self.dqn_loaded = self._try_load_dqn_model(selected)
        if self.dqn_loaded:
            self.controller_mode = "dqn"
            self._set_status(f"DQN loaded: {os.path.basename(selected)}")
        else:
            self._set_status("Failed to load DQN model.")

    def _load_q_from_dialog(self):
        selected = self._open_file_dialog(
            "Choose Q-table file",
            [("Pickle file", "*.pkl"), ("All files", "*.*")],
        )
        if not selected:
            self._set_status("Q-table selection cancelled.")
            return

        self.q_table_loaded = self._try_load_q_table(selected)
        if self.q_table_loaded:
            self.controller_mode = "qlearning"
            self._set_status(f"Q-table loaded: {os.path.basename(selected)}")
        else:
            self._set_status("Failed to load Q-table.")

    def _set_mode(self, mode: str):
        if mode == "dqn" and not self.dqn_loaded:
            self._set_status("DQN not loaded. Press M to load it.")
            return
        if mode == "qlearning" and not self.q_table_loaded:
            self._set_status("Q-table not loaded. Press Q to load it.")
            return
        self.controller_mode = mode
        self._set_status(f"Controller mode: {mode.upper()}")

    def _choose_action(self) -> int:
        if self.controller_mode == "manual":
            if self.last_manual_action is None:
                return 0
            action = self.last_manual_action
            self.last_manual_action = None
            return action

        if self.controller_mode == "mixed":
            if self.last_manual_action is not None:
                action = self.last_manual_action
                self.last_manual_action = None
                return action
            pick = random.random()
            if pick < 0.6 and self.dqn_loaded:
                normalized_state = self.env.get_state_normalized()
                return self.agent.act(normalized_state)
            if pick < 0.85 and self.q_table_loaded:
                q_state = self.q_agent.get_state(self.env._get_state())
                return int(np.argmax(self.q_agent.Q[q_state]))
            return random.randint(0, 2)

        if self.controller_mode == "dqn" and self.dqn_loaded:
            normalized_state = self.env.get_state_normalized()
            return self.agent.act(normalized_state)

        if self.controller_mode == "qlearning" and self.q_table_loaded:
            q_state = self.q_agent.get_state(self.env._get_state())
            return int(np.argmax(self.q_agent.Q[q_state]))

        if self.controller_mode == "random":
            return random.randint(0, 2)

        if self.env.cars_ns > self.env.cars_ew + 3:
            return 1
        if self.env.cars_ew > self.env.cars_ns + 3:
            return 2
        return random.choice([0, 0, 1, 2])

    def _populate_initial_cars(self):
        """Populates the lanes with initial static cars according to env's random start"""
        self.cars_dict = {"N": [], "S": [], "W": [], "E": []}
        self.car_id_counter = 0

        # North-South lanes (Right-Hand Drive: N southbound drives on left side x=545, S northbound on right x=635)
        ns_count = self.env.cars_ns
        n_count = ns_count // 2 + (ns_count % 2)
        s_count = ns_count // 2

        # North southbound cars stack behind stop line at 228
        for i in range(n_count):
            y_pos = 228 - i * 45
            color = random.choice(NEON_COLORS)
            car = VisualCar(self.car_id_counter, "N", 545, y_pos, 0.0, color)
            car.state = "waiting"
            self.cars_dict["N"].append(car)
            self.car_id_counter += 1

        # South northbound cars stack behind stop line at 500
        for i in range(s_count):
            y_pos = 500 + i * 45
            color = random.choice(NEON_COLORS)
            car = VisualCar(self.car_id_counter, "S", 635, y_pos, 0.0, color)
            car.state = "waiting"
            self.cars_dict["S"].append(car)
            self.car_id_counter += 1

        # East-West lanes (Right-Hand Drive: W eastbound drives on bottom y=415, E westbound on top y=325)
        ew_count = self.env.cars_ew
        w_count = ew_count // 2 + (ew_count % 2)
        e_count = ew_count // 2

        # West eastbound cars stack behind stop line at 448
        for i in range(w_count):
            x_pos = 448 - i * 45
            color = random.choice(NEON_COLORS)
            car = VisualCar(self.car_id_counter, "W", x_pos, 415, 0.0, color)
            car.state = "waiting"
            self.cars_dict["W"].append(car)
            self.car_id_counter += 1

        # East westbound cars stack behind stop line at 720
        for i in range(e_count):
            x_pos = 720 + i * 45
            color = random.choice(NEON_COLORS)
            car = VisualCar(self.car_id_counter, "E", x_pos, 325, 0.0, color)
            car.state = "waiting"
            self.cars_dict["E"].append(car)
            self.car_id_counter += 1

    def _spawn_car(self, direction: str) -> bool:
        active_cars = self.cars_dict[direction]
        color = random.choice(NEON_COLORS)

        if direction == "N":
            if any(c.y < 15 for c in active_cars):
                return False
            car = VisualCar(self.car_id_counter, "N", 545, -45, 1.5, color)
        elif direction == "S":
            if any(c.y > 745 for c in active_cars):
                return False
            car = VisualCar(self.car_id_counter, "S", 635, 805, 1.5, color)
        elif direction == "W":
            if any(c.x < 15 for c in active_cars):
                return False
            car = VisualCar(self.car_id_counter, "W", -45, 415, 1.5, color)
        else:  # E
            if any(c.x > 1185 for c in active_cars):
                return False
            car = VisualCar(self.car_id_counter, "E", 1245, 325, 1.5, color)

        self.cars_dict[direction].append(car)
        self.car_id_counter += 1
        return True

    def _release_cars(self, direction: str, count: int):
        if count <= 0:
            return

        candidates = [c for c in self.cars_dict[direction] if c.state in ("waiting", "approaching")]
        if direction == "N":
            candidates.sort(key=lambda c: c.y, reverse=True)
        elif direction == "S":
            candidates.sort(key=lambda c: c.y, reverse=False)
        elif direction == "W":
            candidates.sort(key=lambda c: c.x, reverse=True)
        else:  # E
            candidates.sort(key=lambda c: c.x, reverse=False)

        for i in range(min(count, len(candidates))):
            candidates[i].state = "crossing"

    def _reset_episode(self):
        self.episode_history.append(self.total_reward)
        self.episode += 1
        self.state = self.env.reset()
        self.done = False
        self.last_reward = 0.0
        self.total_reward = 0.0

        # Reset visual components
        self._populate_initial_cars()
        self.pending_spawns = {"N": 0, "S": 0, "W": 0, "E": 0}
        self.spawn_timer = {"N": 0.0, "S": 0.0, "W": 0.0, "E": 0.0}
        self.step_elapsed_ms = 0.0
        self.ns_split = 0.50
        self.ns_release_triggered = False
        self.ew_release_triggered = False

    def _any_car_in_intersection(self, directions: List[str]) -> bool:
        """
        Safety Lock Reservation System: Returns True if any car from the specified
        directions is currently inside the actual intersection box [500-700 x 280-480].
        """
        for d in directions:
            for car in self.cars_dict[d]:
                overlap_x = (car.x + car.w > 500 and car.x < 700)
                overlap_y = (car.y + car.h > 280 and car.y < 480)
                if overlap_x and overlap_y:
                    return True
        return False

    def _update_physics(self, dt_ms: float):
        dt_sec = dt_ms / 1000.0

        for d in ("N", "S", "W", "E"):
            self.spawn_timer[d] = max(0.0, self.spawn_timer[d] - dt_ms)

        for d in ("N", "S", "W", "E"):
            cars = self.cars_dict[d]
            if d == "N":
                cars.sort(key=lambda c: c.y, reverse=True)
                stop_line = 228
                has_green = (self.light_ns == "GREEN")
                # Cross traffic reservation
                cross_active = self._any_car_in_intersection(["W", "E"])
            elif d == "S":
                cars.sort(key=lambda c: c.y, reverse=False)
                stop_line = 500
                has_green = (self.light_ns == "GREEN")
                cross_active = self._any_car_in_intersection(["W", "E"])
            elif d == "W":
                cars.sort(key=lambda c: c.x, reverse=True)
                stop_line = 448
                has_green = (self.light_ew == "GREEN")
                cross_active = self._any_car_in_intersection(["N", "S"])
            else:  # E
                cars.sort(key=lambda c: c.x, reverse=False)
                stop_line = 720
                has_green = (self.light_ew == "GREEN")
                cross_active = self._any_car_in_intersection(["N", "S"])

            for i, car in enumerate(cars):
                if i == 0:
                    # Front-most car stopping and crossing logic with safety reservation checks
                    if d == "N":
                        has_entered = (car.y > 228)
                    elif d == "S":
                        has_entered = (car.y < 500)
                    elif d == "W":
                        has_entered = (car.x > 448)
                    else:  # E
                        has_entered = (car.x < 720)

                    # DOUBLE-LAYERED SAFETY:
                    # Can cross ONLY if:
                    # 1. We have green light AND cross traffic is clear
                    # 2. OR we are already inside the intersection (must clear it)
                    can_cross = (has_green and not cross_active) or has_entered
                    limit = None if can_cross else stop_line
                else:
                    # Subsequent cars strictly obey queue spacings behind the bumper of the car in front
                    # Caps the limit to the stop_line if the light is Red/Yellow to prevent running the red light
                    front_car = cars[i - 1]
                    if d == "N":
                        limit = front_car.y - 45
                        if not has_green and car.y <= 228:
                            limit = min(limit, stop_line)
                    elif d == "S":
                        limit = front_car.y + 45
                        if not has_green and car.y >= 500:
                            limit = max(limit, stop_line)
                    elif d == "W":
                        limit = front_car.x - 45
                        if not has_green and car.x <= 448:
                            limit = min(limit, stop_line)
                    else:  # E
                        limit = front_car.x + 45
                        if not has_green and car.x >= 720:
                            limit = max(limit, stop_line)

                car.update(dt_sec, limit, has_green)

            self.cars_dict[d] = [c for c in cars if c.state != "departed"]

    def _draw_premium_traffic_lights(self, glow_surf: pygame.Surface):
        # Position signals exactly on the RIGHT-hand side of each lane approach
        signals = [
            ("NS", 515, 220),  # North Signal (for Southbound N lane at x=545, placed on right x=515)
            ("NS", 675, 540),  # South Signal (for Northbound S lane at x=635, placed on right x=675)
            ("EW", 450, 430),  # West Signal (for Eastbound W lane at y=415, placed on right y=430)
            ("EW", 750, 310),  # East Signal (for Westbound E lane at y=325, placed on right y=310)
        ]

        for group, cx, cy in signals:
            # Casing drop-shadow
            pygame.draw.rect(self.screen, (10, 15, 20), (cx - 11, cy - 29, 24, 62), border_radius=4)
            # Casing
            pygame.draw.rect(self.screen, (30, 41, 59), (cx - 12, cy - 30, 24, 62), border_radius=4)
            pygame.draw.rect(self.screen, (71, 85, 105), (cx - 12, cy - 30, 24, 62), width=1, border_radius=4)

            light_val = self.light_ns if group == "NS" else self.light_ew

            bulbs = [
                ("RED", cy - 18, (239, 68, 68), (100, 20, 20)),
                ("YELLOW", cy, (245, 158, 11), (100, 60, 10)),
                ("GREEN", cy + 18, (16, 185, 129), (10, 80, 40)),
            ]

            for name, y_coord, active_color, inactive_color in bulbs:
                is_active = (light_val == name)
                pygame.draw.circle(self.screen, (15, 23, 42), (cx, y_coord), 8, width=1)

                if is_active:
                    pygame.draw.circle(self.screen, active_color, (cx, y_coord), 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (cx, y_coord), 2)

                    # Dynamic glow effects
                    a_color = (active_color[0], active_color[1], active_color[2], 100)
                    pygame.draw.circle(glow_surf, a_color, (cx, y_coord), 16)
                    a_color_outer = (active_color[0], active_color[1], active_color[2], 30)
                    pygame.draw.circle(glow_surf, a_color_outer, (cx, y_coord), 28)
                else:
                    pygame.draw.circle(self.screen, inactive_color, (cx, y_coord), 6)

    def _draw_dashboard(self):
        panel = pygame.Rect(840, 20, 340, 720)
        pygame.draw.rect(self.screen, (9, 15, 29), panel, border_radius=12)
        pygame.draw.rect(self.screen, (71, 85, 105), panel, width=2, border_radius=12)

        title = self.font_big.render("Traffic RL Dashboard", True, (241, 245, 249))
        self.screen.blit(title, (865, 35))

        # Dynamic, ultra-responsive live visual car counters updating at 60 FPS
        ns_queue = len([c for c in self.cars_dict["N"] + self.cars_dict["S"] if c.state in ("waiting", "approaching")])
        ew_queue = len([c for c in self.cars_dict["W"] + self.cars_dict["E"] if c.state in ("waiting", "approaching")])
        in_intersection = len([c for d in ("N", "S", "W", "E") for c in self.cars_dict[d] if c.state == "crossing"])

        stats = [
            ("Episode / Step", f"Ep {self.episode}/{self.episodes} | Step {self.env.step_count}/{self.env.max_steps}"),
            ("Global Step", str(self.global_step)),
            ("Cars NS (Live Queue)", f"{ns_queue} vehicles"),
            ("Cars EW (Live Queue)", f"{ew_queue} vehicles"),
            ("Intersection Occupancy", f"{in_intersection} clearing"),
            ("Green Split Priority", f"NS: {self.ns_split*100:.0f}% | EW: {(1.0-self.ns_split)*100:.0f}%"),
            ("Green Duration", f"{self.env.green_duration}s"),
            ("Avg Wait Time", f"{self.env.avg_wait:.2f}s"),
            ("Last Step Reward", f"{self.last_reward:.2f}"),
            ("Total Reward", f"{self.total_reward:.2f}"),
            ("Last Controller Action", ACTION_LABELS[self.last_action]),
            ("Agent Mode", self.controller_mode.upper()),
            ("DQN / QL Status", f"DQN: {'READY' if self.dqn_loaded else 'NO'} | QL: {'READY' if self.q_table_loaded else 'NO'}") if self.global_step % 2 == 0 else ("Simulation Status", "PAUSED" if self.paused else "RUNNING"),
        ]

        y = 90
        for label, val in stats:
            lbl_surf = self.font_small.render(label, True, (148, 163, 184))
            val_surf = self.font.render(val, True, (248, 250, 252))

            pygame.draw.circle(self.screen, (56, 189, 248), (860, y + 10), 3)
            self.screen.blit(lbl_surf, (872, y))
            self.screen.blit(val_surf, (872, y + 18))
            y += 45

        # Reward Graph
        graph_rect = pygame.Rect(860, 615, 300, 90)
        pygame.draw.rect(self.screen, (15, 23, 42), graph_rect, border_radius=6)
        pygame.draw.rect(self.screen, (51, 65, 85), graph_rect, width=1, border_radius=6)

        graph_title = self.font_small.render("Rolling Reward History", True, (148, 163, 184))
        self.screen.blit(graph_title, (865, 595))

        if len(self.episode_history) > 1:
            history = self.episode_history[-20:]
            min_r = min(history)
            max_r = max(history)
            r_range = max_r - min_r if max_r != min_r else 1.0

            points = []
            for i, val in enumerate(history):
                gx = graph_rect.x + 10 + (i / max(1, len(history) - 1)) * (graph_rect.width - 20)
                gy = graph_rect.y + graph_rect.height - 10 - ((val - min_r) / r_range) * (graph_rect.height - 20)
                points.append((gx, gy))

            pygame.draw.lines(self.screen, (34, 211, 238), False, points, 2)
            for gx, gy in points:
                pygame.draw.circle(self.screen, (244, 63, 94), (int(gx), int(gy)), 3)

            min_surf = self.font_small.render(f"Min: {min_r:.0f}", True, (248, 113, 113))
            max_surf = self.font_small.render(f"Max: {max_r:.0f}", True, (74, 222, 128))
            self.screen.blit(min_surf, (865, graph_rect.y + graph_rect.height + 2))
            self.screen.blit(max_surf, (1080, graph_rect.y + graph_rect.height + 2))
        else:
            placeholder = self.font_small.render("Awaiting episode completions...", True, (71, 85, 105))
            self.screen.blit(placeholder, (graph_rect.x + 35, graph_rect.y + 35))

        if self.status_message and pygame.time.get_ticks() <= self.status_until_ms:
            status_surf = self.font_small.render(self.status_message, True, (253, 224, 71))
            self.screen.blit(status_surf, (850, 725))

    def _draw_scene(self):
        # 1. Clear screen and draw glowing cyber city grid
        self.screen.fill((15, 23, 42))

        grid_color = (23, 37, 84)
        grid_spacing = 40
        for x in range(0, 800, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, 760), 1)
        for y in range(0, 760, grid_spacing):
            pygame.draw.line(self.screen, grid_color, (0, y), (800, y), 1)

        # 2. Draw roads
        road_color = (30, 41, 59)
        pygame.draw.rect(self.screen, road_color, (0, 280, 800, 200))
        pygame.draw.rect(self.screen, road_color, (500, 0, 200, 760))

        # Sidewalk borders
        border_color = (56, 189, 248)
        pygame.draw.rect(self.screen, border_color, (0, 277, 800, 3))
        pygame.draw.rect(self.screen, border_color, (0, 480, 800, 3))
        pygame.draw.rect(self.screen, border_color, (497, 0, 3, 760))
        pygame.draw.rect(self.screen, border_color, (700, 0, 3, 760))

        # Exclude borders inside intersection
        pygame.draw.rect(self.screen, road_color, (500, 280, 200, 200))

        # 3. Draw Lane Markings & Double Solid Yellow Dividers
        lane_mark_color = (251, 191, 36)

        # Horizontal Double Solid Yellow Lines (separating y=325 top and y=415 bottom)
        pygame.draw.line(self.screen, lane_mark_color, (0, 378), (490, 378), 2)
        pygame.draw.line(self.screen, lane_mark_color, (0, 382), (490, 382), 2)
        pygame.draw.line(self.screen, lane_mark_color, (710, 378), (800, 378), 2)
        pygame.draw.line(self.screen, lane_mark_color, (710, 382), (800, 382), 2)

        # Vertical Double Solid Yellow Lines (separating x=545 left and x=635 right)
        pygame.draw.line(self.screen, lane_mark_color, (598, 0), (598, 270), 2)
        pygame.draw.line(self.screen, lane_mark_color, (602, 0), (602, 270), 2)
        pygame.draw.line(self.screen, lane_mark_color, (598, 490), (598, 760), 2)
        pygame.draw.line(self.screen, lane_mark_color, (602, 490), (602, 760), 2)

        # 4. Draw Crosswalks (Zebra stripes)
        for x in range(505, 700, 15):
            pygame.draw.rect(self.screen, (226, 232, 240), (x, 242, 6, 16), border_radius=1)
        for x in range(505, 700, 15):
            pygame.draw.rect(self.screen, (226, 232, 240), (x, 502, 6, 16), border_radius=1)
        for y in range(285, 480, 15):
            pygame.draw.rect(self.screen, (226, 232, 240), (462, y, 16, 6), border_radius=1)
        for y in range(285, 480, 15):
            pygame.draw.rect(self.screen, (226, 232, 240), (702, y, 16, 6), border_radius=1)

        # 5. Draw Lane Arrows (aligned for RHD lanes: N moves down x=550, S moves up x=645, W moves right y=425, E moves left y=335)
        arrow_color = (148, 163, 184)
        # North lane (down, x=550)
        pygame.draw.line(self.screen, arrow_color, (550, 100), (550, 120), 2)
        pygame.draw.line(self.screen, arrow_color, (546, 114), (550, 120), 2)
        pygame.draw.line(self.screen, arrow_color, (554, 114), (550, 120), 2)
        # South lane (up, x=645)
        pygame.draw.line(self.screen, arrow_color, (645, 660), (645, 640), 2)
        pygame.draw.line(self.screen, arrow_color, (641, 646), (645, 640), 2)
        pygame.draw.line(self.screen, arrow_color, (649, 646), (645, 640), 2)
        # West lane (right, y=425)
        pygame.draw.line(self.screen, arrow_color, (100, 425), (120, 425), 2)
        pygame.draw.line(self.screen, arrow_color, (114, 421), (120, 425), 2)
        pygame.draw.line(self.screen, arrow_color, (114, 429), (120, 425), 2)
        # East lane (left, y=335)
        pygame.draw.line(self.screen, arrow_color, (700, 335), (680, 335), 2)
        pygame.draw.line(self.screen, arrow_color, (686, 331), (680, 335), 2)
        pygame.draw.line(self.screen, arrow_color, (686, 339), (680, 335), 2)

        # 6. Draw Glowing Stop Lines
        cyan_glow = (34, 211, 238)
        red_glow = (248, 113, 113)

        pygame.draw.line(self.screen, cyan_glow if self.light_ns == "GREEN" else red_glow, (500, 260), (600, 260), 4)
        pygame.draw.line(self.screen, cyan_glow if self.light_ns == "GREEN" else red_glow, (600, 500), (700, 500), 4)
        pygame.draw.line(self.screen, cyan_glow if self.light_ew == "GREEN" else red_glow, (480, 380), (480, 480), 4)  # W stop line bottom
        pygame.draw.line(self.screen, cyan_glow if self.light_ew == "GREEN" else red_glow, (720, 280), (720, 380), 4)  # E stop line top

        # 7. Draw Cars and Glows
        glow_surf = pygame.Surface((1200, 760), pygame.SRCALPHA)

        for d in ("N", "S", "W", "E"):
            for car in self.cars_dict[d]:
                car.draw(self.screen)
                car.draw_lights(self.screen, glow_surf)

        # 8. Draw Premium LED Traffic Lights
        self._draw_premium_traffic_lights(glow_surf)

        self.screen.blit(glow_surf, (0, 0))

        # 9. Draw Dashboard Overlay
        self._draw_dashboard()
        pygame.display.flip()

    def run(self):
        running = True
        while running and self.episode <= self.episodes:
            dt = self.clock.tick(self.fps)  # ms since last frame
            dt_sim = 0.0 if self.paused else float(dt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        self._load_dqn_from_dialog()
                    elif event.key == pygame.K_q:
                        self._load_q_from_dialog()
                    elif event.key == pygame.K_1:
                        self._set_mode("dqn")
                    elif event.key == pygame.K_2:
                        self._set_mode("qlearning")
                    elif event.key == pygame.K_3:
                        self._set_mode("random")
                    elif event.key == pygame.K_4:
                        self._set_mode("manual")
                    elif event.key == pygame.K_5:
                        self._set_mode("mixed")
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                        self._set_status("Paused." if self.paused else "Resumed.")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.step_interval_ms = max(500, self.step_interval_ms - 200)
                        self._set_status(f"Speed up. Step every {self.step_interval_ms} ms")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.step_interval_ms = min(6000, self.step_interval_ms + 200)
                        self._set_status(f"Slow down. Step every {self.step_interval_ms} ms")
                    elif event.key == pygame.K_UP:
                        self.last_manual_action = 1
                        self._set_status("Manual action queued: INCREASE GREEN")
                    elif event.key == pygame.K_DOWN:
                        self.last_manual_action = 2
                        self._set_status("Manual action queued: DECREASE GREEN")
                    elif event.key == pygame.K_SPACE:
                        self.last_manual_action = 0
                        self._set_status("Manual action queued: KEEP GREEN")

            # --- Visual cycle and Step synchronization ---
            if not self.paused and not self.done:
                pct = self.step_elapsed_ms / self.step_interval_ms

                # Dynamic priority phase splits:
                # NS Green runs from 0.0 to t_ns_yellow
                # NS Yellow runs from t_ns_yellow to t_ew_green
                # EW Green runs from t_ew_green to t_ew_yellow
                # EW Yellow runs from t_ew_yellow to 1.0
                t_ns_yellow = self.ns_split - 0.05
                t_ew_green = self.ns_split
                t_ew_yellow = 0.95

                # SMART SIGNAL CLEARANCE SECURITY LOCKS:
                # 1. NS is finishing, EW about to open (pct >= t_ew_green)
                is_holding_ns_to_ew = (pct >= t_ew_green and pct < t_ew_green + 0.05 and self._any_car_in_intersection(["N", "S"]))
                # 2. EW is finishing, NS about to open (pct >= 0.98)
                is_holding_ew_to_ns = (pct >= 0.98 and self._any_car_in_intersection(["W", "E"]))

                if is_holding_ns_to_ew:
                    self.light_ns = "RED"
                    self.light_ew = "RED"
                    dt_applied = 0.0
                elif is_holding_ew_to_ns:
                    self.light_ns = "RED"
                    self.light_ew = "RED"
                    dt_applied = 0.0
                else:
                    dt_applied = dt_sim

                self.step_elapsed_ms += dt_applied

                # Advance environment step when step_interval expires
                if self.step_elapsed_ms >= self.step_interval_ms:
                    self.step_elapsed_ms = 0.0
                    self.ns_release_triggered = False
                    self.ew_release_triggered = False

                    # DYNAMIC PRIORITY GREEN LIGHT ALLOCATION:
                    # Check waiting queues to allocate proportional split ratio for the NEXT step
                    ns_queue = len([c for c in self.cars_dict["N"] + self.cars_dict["S"] if c.state in ("waiting", "approaching")])
                    ew_queue = len([c for c in self.cars_dict["W"] + self.cars_dict["E"] if c.state in ("waiting", "approaching")])
                    total_q = ns_queue + ew_queue

                    if total_q == 0:
                        self.ns_split = 0.50
                    else:
                        # Cap split ratio between 20% and 80% to maintain a minimum green duration for both flows
                        self.ns_split = min(0.80, max(0.20, ns_queue / total_q))

                    cars_ns_before = self.env.cars_ns
                    cars_ew_before = self.env.cars_ew

                    action = self._choose_action()
                    self.last_action = action

                    _, reward, self.done, _ = self.env.step(action)
                    self.last_reward = float(reward)
                    self.total_reward += float(reward)
                    self.global_step += 1
                    self.last_step_ms = pygame.time.get_ticks()

                    # Compute actual service and arrivals to sync visual simulation
                    capacity = self.env.green_duration // 5
                    served_ns = min(cars_ns_before, capacity)
                    arrived_ns = self.env.cars_ns - (cars_ns_before - served_ns)

                    served_ew = min(cars_ew_before, capacity)
                    arrived_ew = self.env.cars_ew - (cars_ew_before - served_ew)

                    self.served_to_release = {
                        "N": served_ns // 2 + (served_ns % 2),
                        "S": served_ns // 2,
                        "W": served_ew // 2 + (served_ew % 2),
                        "E": served_ew // 2,
                    }

                    self.pending_spawns = {
                        "N": arrived_ns // 2 + (arrived_ns % 2),
                        "S": arrived_ns // 2,
                        "W": arrived_ew // 2 + (arrived_ew % 2),
                        "E": arrived_ew // 2,
                    }

                    self.spawn_timer = {"N": 0.0, "S": 0.0, "W": 0.0, "E": 0.0}

                # Update pct after applying dt_applied to ensure smooth transitions
                pct = self.step_elapsed_ms / self.step_interval_ms

                if is_holding_ns_to_ew or is_holding_ew_to_ns:
                    # Light is already set to RED/RED above
                    pass
                else:
                    # Phase 1: NS Green, EW Red
                    if pct < t_ns_yellow:
                        self.light_ns = "GREEN"
                        self.light_ew = "RED"

                        if not self.ns_release_triggered:
                            self._release_cars("N", self.served_to_release["N"])
                            self._release_cars("S", self.served_to_release["S"])
                            self.ns_release_triggered = True

                        # Spawning pending North-South arrivals (spaced out at 6.0s step pacing)
                        for d in ("N", "S"):
                            if self.pending_spawns[d] > 0 and self.spawn_timer[d] <= 0:
                                if self._spawn_car(d):
                                    self.pending_spawns[d] -= 1
                                    self.spawn_timer[d] = random.uniform(800.0, 1500.0)

                    # Phase 2: NS Yellow, EW Red
                    elif pct < t_ew_green:
                        self.light_ns = "YELLOW"
                        self.light_ew = "RED"

                    # Phase 3: EW Green, NS Red
                    elif pct < t_ew_yellow:
                        self.light_ns = "RED"
                        self.light_ew = "GREEN"

                        if not self.ew_release_triggered:
                            self._release_cars("W", self.served_to_release["W"])
                            self._release_cars("E", self.served_to_release["E"])
                            self.ew_release_triggered = True

                        # Spawning pending East-West arrivals
                        for d in ("W", "E"):
                            if self.pending_spawns[d] > 0 and self.spawn_timer[d] <= 0:
                                if self._spawn_car(d):
                                    self.pending_spawns[d] -= 1
                                    self.spawn_timer[d] = random.uniform(800.0, 1500.0)

                    # Phase 4: EW Yellow, NS Red
                    else:
                        self.light_ns = "RED"
                        self.light_ew = "YELLOW"

                # Update physics & reservation queuing
                self._update_physics(dt_sim)

            if self.done:
                self._reset_episode()
                if self.episode > self.episodes:
                    break

            self._draw_scene()

        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(description="Live Premium GUI for traffic signal RL environment.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes.")
    parser.add_argument("--fps", type=int, default=60, help="GUI frame rate.")
    parser.add_argument(
        "--step-ms",
        type=int,
        # Default step interval set to a calm, highly organized 6000 ms (6.0 seconds)
        default=6000,
        help="Milliseconds between environment steps.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn_final.pt",
        help="Path to DQN model file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gui = TrafficSimulationGUI(
        model_path=args.model,
        episodes=args.episodes,
        fps=args.fps,
        step_interval_ms=args.step_ms,
    )
    gui.run()


if __name__ == "__main__":
    main()
