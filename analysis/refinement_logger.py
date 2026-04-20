import csv
from datetime import datetime


class RefinementLogger:
    def __init__(self, log_file="results/dqn_refinement_log.csv"):
        self.log_file = log_file
        self.log_entries = []

        try:
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Step", "Date", "Observation", "Modification", "Result"]
                )
        except FileNotFoundError:
            pass

    def log_change(self, observation, modification, result):
        step = len(self.log_entries) + 1
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "step": step,
            "date": date,
            "observation": observation,
            "modification": modification,
            "result": result,
        }

        self.log_entries.append(entry)

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, date, observation, modification, result])

        print(f"✅ Logged: Step {step} - {observation}")

    def print_log(self):
        print("\n📋 Refinement Log Summary:")
        print("-" * 120)
        for entry in self.log_entries:
            print(f"\nStep {entry['step']} ({entry['date']})")
            print(f"  Observation: {entry['observation']}")
            print(f"  Modification: {entry['modification']}")
            print(f"  Result: {entry['result']}")
        print("-" * 120)
