import json

cnt = 1
with open('temp_v18_gpt-4-1106-preview_prompt_v7_system.jsonl', 'r') as f, open('question.jsonl', 'w') as g:
    for idx, line in enumerate(f):
        js = json.loads(line)
        field = js['field']
        law = js['law']
        policy = js['policy']
        for result in js['results']:
            q = result['question']
            id0 = result['id']
            try:
                category = result['category']
                options = '\n'.join(['%s:%s' % (k, v) for k, v in result['options'].items()])
                question_type = result['question_type']
            except KeyError as e:
                print(e, result)
                continue
            dd = {
                "question_id": cnt,
                "category": category,
                "turns": ["%s\n%s" % (q, options)],
                "reference_answer": result['reference_answer'],
                "question_type": result['question_type'],
                "field": field,
                "law": law
            }
            g.write(json.dumps(dd, ensure_ascii=False) + '\n')
            cnt += 1
        