import json
import random

# Load gaia_tasks
with open("data/gaia_tasks.json") as f:
    all_tasks = json.load(f)

# Filter to only tasks WITH reference answers
tasks_with_ref = [t for t in all_tasks if t.get("reference_answer")]

print(f"Total tasks: {len(all_tasks)}")
print(f"Tasks with reference_answer: {len(tasks_with_ref)}")

# Sample 30 (or however many you need)
random.seed(42)
pool = random.sample(tasks_with_ref, min(30, len(tasks_with_ref)))

# Save to human_study_pool.json
with open("data/human_study_pool.json", "w") as f:
    json.dump(pool, f, indent=2)

print(f"Created pool with {len(pool)} tasks (all have reference answers)")
