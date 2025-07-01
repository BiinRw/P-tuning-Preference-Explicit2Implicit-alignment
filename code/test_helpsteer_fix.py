#!/usr/bin/env python3
"""
Test script to verify HelpSteer data format fix
"""

import json
from collections import defaultdict

def get_helpsteer_fixed(file_path: str):
    """Fixed version of HelpSteer loading that properly handles prompt+response format"""
    print(f'Loading HelpSteer dataset from {file_path}')
    
    data = defaultdict(lambda: defaultdict(list))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            
            # HelpSteer format: chosen/rejected already contain full prompt+response
            # Extract only the response part by removing the prompt
            chosen_response = chosen[len(prompt):].strip() if chosen.startswith(prompt) else chosen
            rejected_response = rejected[len(prompt):].strip() if rejected.startswith(prompt) else rejected
            
            responses = [chosen_response, rejected_response]
            n_responses = len(data[prompt]['responses'])
            data[prompt]['pairs'].append((n_responses, n_responses + 1))
            data[prompt]['responses'].extend(responses)
            data[prompt]['sft_target'] = chosen_response
    
    print(f'HelpSteer dataset loading done.')
    return data

def test_format():
    """Test the fixed format"""
    # Load original data
    file_path = '/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed/train_prefs_helpsteer.jsonl'
    
    # Test original format
    with open(file_path, 'r') as f:
        original_example = json.loads(f.readline().strip())
    
    print("=== ORIGINAL DATA ===")
    print(f"Prompt: {original_example['prompt'][:100]}...")
    print(f"Chosen: {original_example['chosen'][:100]}...")
    print(f"Rejected: {original_example['rejected'][:100]}...")
    
    # Test fixed format
    fixed_data = get_helpsteer_fixed(file_path)
    first_prompt = list(fixed_data.keys())[0]
    first_data = fixed_data[first_prompt]
    
    print("\n=== FIXED DATA ===")
    print(f"Prompt: {first_prompt[:100]}...")
    print(f"Response 1: {first_data['responses'][0][:100]}...")
    print(f"Response 2: {first_data['responses'][1][:100]}...")
    
    # Test tokenization simulation
    print("\n=== TOKENIZATION SIMULATION ===")
    simulated_chosen = first_prompt + first_data['responses'][0]
    simulated_rejected = first_prompt + first_data['responses'][1]
    
    print(f"Simulated chosen: {simulated_chosen[:100]}...")
    print(f"Simulated rejected: {simulated_rejected[:100]}...")
    
    # Check for prompt duplication
    prompt_start = first_prompt[:50]
    chosen_count = simulated_chosen.count(prompt_start)
    rejected_count = simulated_rejected.count(prompt_start)
    
    print(f"\n=== DUPLICATION CHECK ===")
    print(f"Prompt appears in chosen: {chosen_count} times")
    print(f"Prompt appears in rejected: {rejected_count} times")
    
    if chosen_count == 1 and rejected_count == 1:
        print("✅ SUCCESS: No prompt duplication!")
    else:
        print("❌ ERROR: Prompt duplication detected!")

if __name__ == "__main__":
    test_format()
