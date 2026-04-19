"""
Traffic Signal Control Environment
Author: Engineer 1
Description: Custom RL environment for a single intersection traffic signal control.
             No external RL libraries required.
"""

import random


class TrafficEnv:
    """
    A simple single-intersection traffic signal environment.

    State:
        - cars_ns   : number of cars waiting North-South
        - cars_ew   : number of cars waiting East-West
        - green_duration : current green phase duration (seconds)
        - avg_wait  : average waiting time of all cars (seconds)

    Actions (Discrete):
        0 -> keep green duration the same
        1 -> increase green duration by 5 seconds
        2 -> decrease green duration by 5 seconds

    Reward:
        Negative value proportional to average waiting time.
        reward = -avg_wait
        (The agent learns to minimise waiting time to maximise reward)
    """

    MIN_GREEN = 5    # seconds
    MAX_GREEN = 60   # seconds
    MAX_CARS  = 20   # max cars per direction per step

    def __init__(self, max_steps: int = 200):
        self.max_steps = max_steps
        self.reset()

    # ------------------------------------------------------------------
    # Core API (used by Engineer 2 & 3)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset the environment to an initial state. Returns the initial state."""
        self.step_count      = 0
        self.cars_ns         = random.randint(0, self.MAX_CARS)
        self.cars_ew         = random.randint(0, self.MAX_CARS)
        self.green_duration  = 30          # start at a neutral 30 s
        self.avg_wait        = 0.0
        self.total_wait      = 0.0
        self.cars_served     = 0
        return self._get_state()

    def step(self, action: int):
        """
        Apply an action and advance the environment by one time step.

        Args:
            action (int): 0 = keep, 1 = increase +5s, 2 = decrease -5s

        Returns:
            next_state (tuple): the new state
            reward     (float): immediate reward
            done       (bool) : whether the episode has ended
            info       (dict) : extra diagnostic information
        """
        assert action in (0, 1, 2), f"Invalid action {action}. Must be 0, 1, or 2."

        # --- Apply action ---
        if action == 1:
            self.green_duration = min(self.green_duration + 5, self.MAX_GREEN)
        elif action == 2:
            self.green_duration = max(self.green_duration - 5, self.MIN_GREEN)
        # action 0: no change

        # --- Simulate one time step of traffic ---
        self._simulate_traffic()

        # --- Reward ---
        reward = -self.avg_wait

        # --- Check termination ---
        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "step":           self.step_count,
            "cars_ns":        self.cars_ns,
            "cars_ew":        self.cars_ew,
            "green_duration": self.green_duration,
            "avg_wait":       round(self.avg_wait, 2),
        }

        return self._get_state(), reward, done, info

    def render(self):
        """Print a simple text representation of the current state."""
        print(
            f"Step {self.step_count:>3} | "
            f"NS cars: {self.cars_ns:>2} | "
            f"EW cars: {self.cars_ew:>2} | "
            f"Green: {self.green_duration:>2}s | "
            f"Avg wait: {self.avg_wait:.1f}s"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> tuple:
        """Return the current state as a tuple."""
        return (self.cars_ns, self.cars_ew, self.green_duration, round(self.avg_wait, 1))

    def _simulate_traffic(self):
        """
        Simplified traffic simulation for one time step.

        Logic:
        - Cars served during this step depend on the green duration.
        - New cars arrive randomly.
        - Average wait time is updated accordingly.
        """
        # Cars that pass through (proportional to green time, capped by waiting cars)
        capacity = self.green_duration // 5          # how many cars can pass per 5 s block
        served_ns = min(self.cars_ns, capacity)
        served_ew = min(self.cars_ew, capacity)
        total_served = served_ns + served_ew

        # Update waiting cars after service
        self.cars_ns -= served_ns
        self.cars_ew -= served_ew

        # New cars arrive
        self.cars_ns = min(self.cars_ns + random.randint(0, 5), self.MAX_CARS)
        self.cars_ew = min(self.cars_ew + random.randint(0, 5), self.MAX_CARS)

        # Update average wait time
        # Cars that were NOT served wait one more time unit
        waiting = self.cars_ns + self.cars_ew
        self.total_wait += waiting
        self.cars_served += total_served

        if (waiting + total_served) > 0:
            self.avg_wait = self.total_wait / max(self.step_count + 1, 1)
        else:
            self.avg_wait = 0.0


# ------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------

def run_tests():
    print("=" * 50)
    print("Running Unit Tests for TrafficEnv")
    print("=" * 50)

    env = TrafficEnv(max_steps=5)

    # Test 1: reset returns a 4-element tuple
    state = env.reset()
    assert isinstance(state, tuple) and len(state) == 4, "FAIL: reset() state shape"
    print("✓ Test 1 passed: reset() returns correct state shape")

    # Test 2: action 0 keeps green duration the same
    env.reset()
    env.green_duration = 30
    env.step(0)
    assert env.green_duration == 30, "FAIL: action 0 should not change green_duration"
    print("✓ Test 2 passed: action 0 keeps green_duration unchanged")

    # Test 3: action 1 increases green duration
    env.reset()
    env.green_duration = 30
    env.step(1)
    assert env.green_duration == 35, "FAIL: action 1 should increase green_duration by 5"
    print("✓ Test 3 passed: action 1 increases green_duration by 5")

    # Test 4: action 2 decreases green duration
    env.reset()
    env.green_duration = 30
    env.step(2)
    assert env.green_duration == 25, "FAIL: action 2 should decrease green_duration by 5"
    print("✓ Test 4 passed: action 2 decreases green_duration by 5")

    # Test 5: green_duration never exceeds MAX_GREEN
    env.reset()
    env.green_duration = 60
    env.step(1)
    assert env.green_duration == 60, "FAIL: green_duration exceeded MAX_GREEN"
    print("✓ Test 5 passed: green_duration capped at MAX_GREEN (60s)")

    # Test 6: green_duration never goes below MIN_GREEN
    env.reset()
    env.green_duration = 5
    env.step(2)
    assert env.green_duration == 5, "FAIL: green_duration went below MIN_GREEN"
    print("✓ Test 6 passed: green_duration floored at MIN_GREEN (5s)")

    # Test 7: episode ends after max_steps
    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(0)
    assert done, "FAIL: episode should be done after max_steps"
    print("✓ Test 7 passed: episode ends correctly after max_steps")

    # Test 8: invalid action raises AssertionError
    env.reset()
    try:
        env.step(99)
        print("✗ Test 8 FAILED: should have raised AssertionError for invalid action")
    except AssertionError:
        print("✓ Test 8 passed: invalid action raises AssertionError")

    print("=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)


if __name__ == "__main__":
    # Run tests
    run_tests()

    # Quick demo
    print("\nDemo: 10 random steps")
    print("-" * 50)
    env = TrafficEnv(max_steps=10)
    state = env.reset()
    env.render()

    for _ in range(10):
        action = random.choice([0, 1, 2])
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            break
