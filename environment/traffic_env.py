import random


class TrafficEnv:
    MIN_GREEN = 5
    MAX_GREEN = 60
    MAX_CARS = 20

    def __init__(
        self,
        max_steps: int = 200,
        reward_balance_weight: float = 0.0,
        throughput_bonus_weight: float = 0.0,
    ):
        self.max_steps = max_steps
        self.reward_balance_weight = float(reward_balance_weight)
        self.throughput_bonus_weight = float(throughput_bonus_weight)
        self._last_total_served = 0
        self.reset()

    def reset(self):
        self.step_count = 0
        self.cars_ns = random.randint(0, self.MAX_CARS)
        self.cars_ew = random.randint(0, self.MAX_CARS)
        self.green_duration = 30
        self.avg_wait = 0.0
        self.total_wait = 0.0
        self.cars_served = 0
        self._last_total_served = 0
        return self._get_state()

    def step(self, action: int):
        assert action in (0, 1, 2), f"Invalid action {action}. Must be 0, 1, or 2."

        if action == 1:
            self.green_duration = min(self.green_duration + 5, self.MAX_GREEN)
        elif action == 2:
            self.green_duration = max(self.green_duration - 5, self.MIN_GREEN)

        self._simulate_traffic()

        reward = -self.avg_wait
        if self.reward_balance_weight > 0.0:
            imbalance = abs(self.cars_ns - self.cars_ew) / max(self.MAX_CARS, 1)
            reward -= self.reward_balance_weight * imbalance
        if self.throughput_bonus_weight > 0.0:
            reward += self.throughput_bonus_weight * float(self._last_total_served)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "cars_ns": self.cars_ns,
            "cars_ew": self.cars_ew,
            "green_duration": self.green_duration,
            "avg_wait": round(self.avg_wait, 2),
            "served_last_step": int(self._last_total_served),
        }

        return self._get_state(), reward, done, info

    def render(self):
        print(
            f"Step {self.step_count:>3} | "
            f"NS cars: {self.cars_ns:>2} | "
            f"EW cars: {self.cars_ew:>2} | "
            f"Green: {self.green_duration:>2}s | "
            f"Avg wait: {self.avg_wait:.1f}s"
        )

    def _get_state(self) -> tuple:
        return (
            self.cars_ns,
            self.cars_ew,
            self.green_duration,
            round(self.avg_wait, 1),
        )

    def get_state_normalized(self) -> tuple:
        max_wait = max(float(self.max_steps) * self.MAX_CARS * 2, 1.0)
        return (
            self.cars_ns / self.MAX_CARS,
            self.cars_ew / self.MAX_CARS,
            (self.green_duration - self.MIN_GREEN)
            / max(self.MAX_GREEN - self.MIN_GREEN, 1),
            min(self.avg_wait / max_wait, 1.0),
        )

    def _simulate_traffic(self):
        capacity = self.green_duration // 5
        served_ns = min(self.cars_ns, capacity)
        served_ew = min(self.cars_ew, capacity)
        total_served = served_ns + served_ew
        self._last_total_served = int(total_served)

        self.cars_ns -= served_ns
        self.cars_ew -= served_ew

        self.cars_ns = min(self.cars_ns + random.randint(0, 5), self.MAX_CARS)
        self.cars_ew = min(self.cars_ew + random.randint(0, 5), self.MAX_CARS)

        waiting = self.cars_ns + self.cars_ew
        self.total_wait += waiting
        self.cars_served += total_served

        if (waiting + total_served) > 0:
            self.avg_wait = self.total_wait / max(self.step_count + 1, 1)
        else:
            self.avg_wait = 0.0


def run_tests():
    print("=" * 50)
    print("Running Unit Tests for TrafficEnv")
    print("=" * 50)

    env = TrafficEnv(max_steps=5)

    state = env.reset()
    assert isinstance(state, tuple) and len(state) == 4, "FAIL: reset() state shape"
    print("✓ Test 1 passed: reset() returns correct state shape")

    env.reset()
    env.green_duration = 30
    env.step(0)
    assert env.green_duration == 30, "FAIL: action 0 should not change green_duration"
    print("✓ Test 2 passed: action 0 keeps green_duration unchanged")

    env.reset()
    env.green_duration = 30
    env.step(1)
    assert (
        env.green_duration == 35
    ), "FAIL: action 1 should increase green_duration by 5"
    print("✓ Test 3 passed: action 1 increases green_duration by 5")

    env.reset()
    env.green_duration = 30
    env.step(2)
    assert (
        env.green_duration == 25
    ), "FAIL: action 2 should decrease green_duration by 5"
    print("✓ Test 4 passed: action 2 decreases green_duration by 5")

    env.reset()
    env.green_duration = 60
    env.step(1)
    assert env.green_duration == 60, "FAIL: green_duration exceeded MAX_GREEN"
    print("✓ Test 5 passed: green_duration capped at MAX_GREEN (60s)")

    env.reset()
    env.green_duration = 5
    env.step(2)
    assert env.green_duration == 5, "FAIL: green_duration went below MIN_GREEN"
    print("✓ Test 6 passed: green_duration floored at MIN_GREEN (5s)")

    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(0)
    assert done, "FAIL: episode should be done after max_steps"
    print("✓ Test 7 passed: episode ends correctly after max_steps")

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
    run_tests()

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
