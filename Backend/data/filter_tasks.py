import json

with open("gaia_tasks.json") as f:
    tasks = json.load(f)

# Keep only tasks WITH reference answers
filtered = [t for t in tasks if t.get("reference_answer")]

print(f"Before: {len(tasks)} tasks")
print(f"After: {len(filtered)} tasks")

with open("gaia_tasks.json", "w") as f:
    json.dump(filtered, f, indent=2)

print("✅ Updated gaia_tasks.json - removed empty references")
