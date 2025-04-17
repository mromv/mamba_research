import json
import argparse

def process_dictionaries(dict_list):
    result = []
    for d in dict_list:            
        d['answer'] = d['answer'].lower()
        
        if d['answer'] in ['yes', 'no']:
            result.append(d)
            continue

        if 'yes' in d['answer']:
            d['answer'] = 'yes'
            result.append(d)
        elif 'no' in d['answer']:
            d['answer'] = 'no'
            result.append(d)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    filepath = f'data/{args.filename}'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
        
    processed_lines = process_dictionaries(lines)
    lines_sorted = sorted(processed_lines, key=lambda x: x['idx'])
    
    with open(f'{filepath}_processed.jsonl', 'w', encoding='utf-8') as f:
        for item in lines_sorted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
