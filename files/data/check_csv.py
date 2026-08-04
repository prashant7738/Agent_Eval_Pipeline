import csv

with open("annotator1_sheet.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print(f"\nColumn names: {list(rows[0].keys()) if rows else 'NO DATA'}")

# Check reference_answer column
if rows:
    empty_ref = sum(1 for r in rows if not r.get("reference_answer") or r.get("reference_answer").strip() == "")
    print(f"\nEmpty 'reference_answer': {empty_ref} ({empty_ref/len(rows)*100:.1f}%)")
    
    # Show first 5 rows
    print(f"\nFirst 5 rows:")
    for i, row in enumerate(rows[:5]):
        print(f"  [{i+1}] Q: {row.get('question', 'N/A')[:40]}...")
        print(f"      Ref: {row.get('reference_answer', '[EMPTY]')[:40]}")
        print(f"      Response: {row.get('response', '[EMPTY]')[:40]}")
