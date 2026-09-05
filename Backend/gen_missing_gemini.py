import json
import os
from groq import Groq

# Load from .env
with open('.env', 'r') as f:
    for line in f:
        if '=' in line:
            key, val = line.strip().split('=', 1)
            os.environ[key.strip()] = val.strip()

groq_key = os.environ.get('GROQ_API_KEY')

# Load data
with open('data/gaia_tasks.json', encoding='utf-8') as f:
    all_tasks = json.load(f)

with open('data/responses_gemini.json', encoding='utf-8') as f:
    responses = json.load(f)

# All 10 main items
main_ids = [
    '0bb3b44a-ede5-4db5-a520-4e844b0079c5',
    '1dcc160f-c187-48c2-b68e-319bd4354f3d',
    '3f57289b-8c60-48be-bd80-01f8099ca449',
    '46719c30-f4c3-4cad-be07-d5cb21eee6bb',
    '676e5e31-a554-4acc-9286-b60d90a92d26',
    '7619a514-5fa8-43ef-9143-83b66a43d7a4',
    '7673d772-ef80-4f0f-a602-1bf4485c9b43',
    '8131e2c0-0083-4265-9ce7-78c2d568425d',
    '8b3379c0-0981-4f5b-8407-6444610cb212',
    'a0068077-79f4-461a-adfe-75c1a4148545'
]

tasks_dict = {t['task_id']: t for t in all_tasks}
client = Groq(api_key=groq_key)

print('Generating missing Gemini responses...\n')

for task_id in main_ids:
    if task_id in responses:
        print(f'{task_id[:8]}: already have')
        continue
    
    if task_id not in tasks_dict:
        print(f'{task_id}: task not found')
        continue
    
    task = tasks_dict[task_id]
    question = task['question']
    
    print(f'{task_id[:8]}: generating...', end=' ')
    
    try:
        result = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{'role': 'user', 'content': question}],
            max_tokens=512,
            temperature=0.7
        )
        
        response_text = result.choices[0].message.content.strip()
        responses[task_id] = response_text
        print(f'OK ({len(response_text)} chars)')
    
    except Exception as e:
        print(f'ERROR: {e}')

# Save all responses
with open('data/responses_gemini.json', 'w', encoding='utf-8') as f:
    json.dump(responses, f, indent=2)

print(f'\nTotal responses: {len(responses)}')
print('Saved: responses_gemini.json')
