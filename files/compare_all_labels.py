import json
import csv
from pathlib import Path
from kappa import cohens_kappa, interpret_kappa, per_class_prf, pabak, collapse_binary

DATA_DIR = Path('data')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

print('Loading data...\n')
anno1_csv = load_csv(DATA_DIR / 'annotator1_sheet.csv')
anno2_csv = load_csv(DATA_DIR / 'annotator2_sheet.csv')
judge_groq = load_json(DATA_DIR / 'judge_labels.json')
judge_gemini = load_json(DATA_DIR / 'judge_labels_gemini.json')
final_human = load_json(DATA_DIR / 'final_human_labels.json')

a1_dict = {r['item_id']: r['label'] for r in anno1_csv}
a2_dict = {r['item_id']: r['label'] for r in anno2_csv}

main_ids = sorted([k.replace('main_', '') for k in judge_groq.keys() if k.startswith('main_')])

print(f'Total main items: {len(main_ids)}\n')
print('=' * 150)
print(f"{'#':<3} {'Task ID':<37} {'Annotator1':<12} {'Annotator2':<12} {'Groq':<12} {'Gemini':<12} {'Final':<12}")
print('-' * 150)

rows = []
for i, tid in enumerate(main_ids, 1):
    mid = f'main_{tid}'
    a1_label = a1_dict.get(mid, 'MISSING')
    a2_label = a2_dict.get(mid, 'MISSING')
    g_label = judge_groq.get(mid, 'MISSING')
    gm_label = judge_gemini.get(mid, 'MISSING')
    f_label = final_human.get(mid, 'MISSING')
    
    rows.append({'tid': tid, 'a1': a1_label, 'a2': a2_label, 'groq': g_label, 'gemini': gm_label, 'final': f_label})
    print(f"{i:<3} {tid:<37} {a1_label:<12} {a2_label:<12} {g_label:<12} {gm_label:<12} {f_label:<12}")

print('\n' + '=' * 80)
print('CONFUSION MATRICES & KAPPA STATISTICS')
print('=' * 80)

def print_cm(confusion, labels):
    max_len = max(len(l) for l in labels)
    print(f"  {' ':<{max_len}} ", end='')
    for label in labels:
        print(f"{label:>10}", end=' ')
    print()
    for row_label in labels:
        print(f"  {row_label:<{max_len}} ", end='')
        for col_label in labels:
            print(f"{confusion[row_label][col_label]:>10}", end=' ')
        print()

print('\n1. ANNOTATOR1 vs ANNOTATOR2')
a1_labels = [r['a1'] for r in rows if r['a1'] != 'MISSING' and r['a2'] != 'MISSING']
a2_labels = [r['a2'] for r in rows if r['a1'] != 'MISSING' and r['a2'] != 'MISSING']

if a1_labels:
    result = cohens_kappa(a1_labels, a2_labels)
    print(f"   κ = {result['kappa']} ({interpret_kappa(result['kappa'])})")
    print(f"   n = {result['n']}, Observed agreement = {result['observed_agreement']}")
    print('   Confusion Matrix (rows=Annotator1, cols=Annotator2):')
    print_cm(result['confusion_matrix'], result['labels'])

print('\n2. GROQ JUDGE vs GEMINI JUDGE')
groq_aligned = [r['groq'] for r in rows if r['groq'] != 'MISSING' and r['gemini'] != 'MISSING']
gemini_aligned = [r['gemini'] for r in rows if r['groq'] != 'MISSING' and r['gemini'] != 'MISSING']

if groq_aligned:
    result = cohens_kappa(groq_aligned, gemini_aligned)
    print(f"   κ = {result['kappa']} ({interpret_kappa(result['kappa'])})")
    print(f"   n = {result['n']}, Observed agreement = {result['observed_agreement']}")
    print('   Confusion Matrix (rows=Groq, cols=Gemini):')
    print_cm(result['confusion_matrix'], result['labels'])
else:
    print('   No overlapping items')

print('\n3. GROQ JUDGE vs FINAL HUMAN')
groq_aligned = [r['groq'] for r in rows if r['groq'] != 'MISSING' and r['final'] != 'MISSING']
final_aligned = [r['final'] for r in rows if r['groq'] != 'MISSING' and r['final'] != 'MISSING']

if groq_aligned:
    result = cohens_kappa(groq_aligned, final_aligned)
    print(f"   κ = {result['kappa']} ({interpret_kappa(result['kappa'])})")
    print(f"   n = {result['n']}, Observed agreement = {result['observed_agreement']}")
    print('   Confusion Matrix (rows=Groq, cols=Final Human):')
    print_cm(result['confusion_matrix'], result['labels'])

print('\n4. GEMINI JUDGE vs FINAL HUMAN')
gemini_aligned = [r['gemini'] for r in rows if r['gemini'] != 'MISSING' and r['final'] != 'MISSING']
final_aligned = [r['final'] for r in rows if r['gemini'] != 'MISSING' and r['final'] != 'MISSING']

if gemini_aligned:
    result = cohens_kappa(gemini_aligned, final_aligned)
    print(f"   κ = {result['kappa']} ({interpret_kappa(result['kappa'])})")
    print(f"   n = {result['n']}, Observed agreement = {result['observed_agreement']}")
    print('   Confusion Matrix (rows=Gemini, cols=Final Human):')
    print_cm(result['confusion_matrix'], result['labels'])
else:
    print('   No overlapping items')

print('\n' + '=' * 80)
print('PRIMARY ANALYSIS: FULL n=30 POOL (main + selfcheck resamples), GROQ + HUMAN ONLY')
print('(Gemini judge only covers the 10 main_ items above, so it is reported separately)')
print('=' * 80)

final_human = load_json(DATA_DIR / 'final_human_labels.json')
all_ids = sorted(set(a1_dict) & set(a2_dict) & set(judge_groq) & set(final_human))
print(f'\nItems in full pool with all four labels present: {len(all_ids)}\n')

a1_full = [a1_dict[i] for i in all_ids]
a2_full = [a2_dict[i] for i in all_ids]
groq_full = [judge_groq[i] for i in all_ids]
human_full = [final_human[i] for i in all_ids]

print('5. ANNOTATOR1 vs ANNOTATOR2 (n=30)')
result_hh30 = cohens_kappa(a1_full, a2_full)
print(f"   κ = {result_hh30['kappa']} ({interpret_kappa(result_hh30['kappa'])})")
print(f"   n = {result_hh30['n']}, Observed agreement = {result_hh30['observed_agreement']}")
print('   Confusion Matrix (rows=Annotator1, cols=Annotator2):')
print_cm(result_hh30['confusion_matrix'], result_hh30['labels'])

print('\n6. GROQ JUDGE vs FINAL HUMAN (n=30, primary)')
result_gh30 = cohens_kappa(groq_full, human_full)
print(f"   κ = {result_gh30['kappa']} ({interpret_kappa(result_gh30['kappa'])})")
print(f"   n = {result_gh30['n']}, Observed agreement = {result_gh30['observed_agreement']}")
print('   Confusion Matrix (rows=Groq, cols=Final Human):')
print_cm(result_gh30['confusion_matrix'], result_gh30['labels'])

print('\n   Per-class precision/recall/F1 (Groq predicted vs. human gold):')
prf = per_class_prf(result_gh30['confusion_matrix'], result_gh30['labels'])
for label, m in prf.items():
    print(f"     {label:<14} precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}")

print('\n   Binary collapse (correct+partial=right vs incorrect+hallucinated=wrong):')
groq_bin = collapse_binary(groq_full)
human_bin = collapse_binary(human_full)
result_bin = cohens_kappa(groq_bin, human_bin)
print(f"     binary κ = {result_bin['kappa']} ({interpret_kappa(result_bin['kappa'])}), "
      f"observed agreement = {result_bin['observed_agreement']}")
print(f"     PABAK = {pabak(result_bin['observed_agreement'])}")

print('\n' + '=' * 80)
