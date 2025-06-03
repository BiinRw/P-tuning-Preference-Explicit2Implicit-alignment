from datasets import load_dataset
import json

if __name__ == '__main__':
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    subset_name = 'train_prefs'
    output_file = f'/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/{subset_name}_ultrafeedback_binarized_1w.jsonl'

    with open(output_file, 'w', encoding='utf-8') as f:
        i = 0
        for data in ds[subset_name]:
            i += 1
            if i >=10000:
                break
            prompt = data['prompt']
            formatted_data = {
                'prompt': prompt,
                'chosen': prompt +" " + data['chosen'][1]['content'],
                'rejected': prompt + " "+ data['rejected'][1]['content']
            }
            
            f.write(json.dumps(formatted_data) + "\n")
    print(f"Data saved to {output_file}")
        