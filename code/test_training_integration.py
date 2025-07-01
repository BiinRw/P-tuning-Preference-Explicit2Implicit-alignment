#!/usr/bin/env python3
"""
Test integration of fixed HelpSteer format with training pipeline
"""

import os
import sys
sys.path.append('/home/wangbinrui/research_projects/llama_rlhf/code')

from pro_utils.preference_datasets import get_dataset, tokenize_batch_element
from transformers import AutoTokenizer

def test_training_integration():
    """Test the fixed HelpSteer format with training tokenization"""
    
    print("🧪 Testing HelpSteer integration with training pipeline...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load HelpSteer dataset using the fixed function
    helpsteer_path = '/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed/train_prefs_helpsteer.jsonl'
    data = get_dataset('helpsteer', 'train', file_path=helpsteer_path)
    
    print(f"✅ Loaded {len(data)} prompts from HelpSteer dataset")
    
    # Test tokenization on first example
    first_prompt = list(data.keys())[0]
    first_data = data[first_prompt]
    
    # Get responses
    chosen_response = first_data['responses'][0]  # Should be response only
    rejected_response = first_data['responses'][1]  # Should be response only
    
    print(f"📝 Testing tokenization...")
    print(f"Prompt length: {len(first_prompt)} chars")
    print(f"Chosen response length: {len(chosen_response)} chars") 
    print(f"Rejected response length: {len(rejected_response)} chars")
    
    # Test tokenization
    batch_element = tokenize_batch_element(
        prompt=first_prompt,
        chosen=chosen_response, 
        rejected=rejected_response,
        truncation_mode='keep_start',
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=128
    )
    
    # Verify final results
    final_chosen = batch_element['chosen']
    final_rejected = batch_element['rejected']
    
    print(f"\n🔍 Final tokenized results:")
    print(f"Final chosen: {final_chosen[:100]}...")
    print(f"Final rejected: {final_rejected[:100]}...")
    
    # Check for duplication
    prompt_start = first_prompt[:50]
    chosen_count = final_chosen.count(prompt_start)
    rejected_count = final_rejected.count(prompt_start)
    
    print(f"\n✨ Duplication check:")
    print(f"Prompt appears in chosen: {chosen_count} times")
    print(f"Prompt appears in rejected: {rejected_count} times")
    
    if chosen_count == 1 and rejected_count == 1:
        print("✅ SUCCESS: Training pipeline integration works correctly!")
        print("✅ No prompt duplication in final tokenized output!")
        return True
    else:
        print("❌ ERROR: Prompt duplication detected in training pipeline!")
        return False

def test_comparison_with_ultrafeedback():
    """Compare HelpSteer and UltraFeedback processing"""
    
    print("\n🔄 Comparing HelpSteer vs UltraFeedback processing...")
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load both datasets
    helpsteer_path = '/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed/train_prefs_helpsteer.jsonl'
    ultrafeedback_path = '/home/wangbinrui/research_projects/llama_rlhf/datasets/ultrafeedback_binarized/train_prefs_ultrafeedback_binarized.jsonl'
    
    helpsteer_data = get_dataset('helpsteer', 'train', file_path=helpsteer_path)
    ultrafeedback_data = get_dataset('ufb', 'train', file_path=ultrafeedback_path)
    
    print(f"📊 HelpSteer: {len(helpsteer_data)} prompts")
    print(f"📊 UltraFeedback: {len(ultrafeedback_data)} prompts")
    
    # Test both on same tokenization parameters
    for dataset_name, data in [("HelpSteer", helpsteer_data), ("UltraFeedback", ultrafeedback_data)]:
        first_prompt = list(data.keys())[0]
        first_data = data[first_prompt]
        
        batch_element = tokenize_batch_element(
            prompt=first_prompt,
            chosen=first_data['responses'][0],
            rejected=first_data['responses'][1],
            truncation_mode='keep_start',
            tokenizer=tokenizer,
            max_length=512,
            max_prompt_length=128
        )
        
        # Check duplication
        prompt_start = first_prompt[:30]
        chosen_count = batch_element['chosen'].count(prompt_start)
        rejected_count = batch_element['rejected'].count(prompt_start)
        
        print(f"\n{dataset_name}:")
        print(f"  Prompt duplication in chosen: {chosen_count}")
        print(f"  Prompt duplication in rejected: {rejected_count}")
        
        if chosen_count == 1 and rejected_count == 1:
            print(f"  ✅ {dataset_name} processing: CORRECT")
        else:
            print(f"  ❌ {dataset_name} processing: ERROR")

if __name__ == "__main__":
    success = test_training_integration()
    test_comparison_with_ultrafeedback()
    
    if success:
        print("\n🎉 HelpSteer format fix is ready for training!")
        print("🚀 You can now use HelpSteer data with the training pipeline.")
    else:
        print("\n⚠️  Issues detected. Please review the fix.")
