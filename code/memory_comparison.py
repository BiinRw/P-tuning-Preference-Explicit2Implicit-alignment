#!/usr/bin/env python3
"""
Memory Usage Comparison Script

This script demonstrates the memory savings achieved by using the memory-optimized
dataset format compared to the standard format.
"""

import os
import psutil
import json
import time
from typing import Dict, List
import torch
from transformers import AutoTokenizer


def measure_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def load_standard_dataset(file_path: str, n_samples: int = 1000) -> List[Dict]:
    """Load standard format dataset."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data.append(json.loads(line.strip()))
    return data


def load_optimized_dataset(file_path: str, n_samples: int = 1000) -> List[Dict]:
    """Load memory-optimized format dataset."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            example = json.loads(line.strip())
            # Reconstruct full chosen/rejected on demand
            data.append({
                'prompt': example['prompt'],
                'chosen': example['prompt'] + example['chosen_response'],
                'rejected': example['prompt'] + example['rejected_response']
            })
    return data


def simulate_training_memory_usage(dataset: List[Dict], tokenizer, batch_size: int = 4):
    """Simulate memory usage during training."""
    memory_usage = []
    
    for i in range(0, min(len(dataset), 100), batch_size):  # Process first 100 samples
        batch = dataset[i:i + batch_size]
        
        # Tokenize batch (simulating training)
        for example in batch:
            chosen_tokens = tokenizer(example['chosen'], 
                                    return_tensors='pt', 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=512)
            rejected_tokens = tokenizer(example['rejected'], 
                                      return_tensors='pt', 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=512)
        
        # Measure memory after tokenization
        memory_usage.append(measure_memory_usage())
    
    return memory_usage


def analyze_file_sizes(standard_file: str, optimized_file: str) -> Dict:
    """Analyze file size differences."""
    standard_size = os.path.getsize(standard_file) / (1024 * 1024)  # MB
    optimized_size = os.path.getsize(optimized_file) / (1024 * 1024)  # MB
    
    return {
        'standard_size_mb': standard_size,
        'optimized_size_mb': optimized_size,
        'savings_mb': standard_size - optimized_size,
        'savings_percentage': ((standard_size - optimized_size) / standard_size) * 100
    }


def run_memory_comparison():
    """Run comprehensive memory comparison."""
    print("🧠 Memory Usage Comparison: Standard vs Optimized Format")
    print("=" * 60)
    
    # File paths
    standard_file = "/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed/train_prefs_helpsteer.jsonl"
    optimized_file = "/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed/train_prefs_helpsteer_optimized.jsonl"
    
    # Check if files exist
    if not os.path.exists(standard_file):
        print(f"❌ Standard file not found: {standard_file}")
        print("Please run HelpSteer processor first to generate standard format.")
        return
    
    if not os.path.exists(optimized_file):
        print(f"❌ Optimized file not found: {optimized_file}")
        print("Please run HelpSteer processor with --memory-optimized flag.")
        return
    
    # Analyze file sizes
    print("📁 File Size Analysis:")
    file_analysis = analyze_file_sizes(standard_file, optimized_file)
    print(f"   Standard format: {file_analysis['standard_size_mb']:.2f} MB")
    print(f"   Optimized format: {file_analysis['optimized_size_mb']:.2f} MB")
    print(f"   File size savings: {file_analysis['savings_mb']:.2f} MB ({file_analysis['savings_percentage']:.1f}%)")
    
    # Initialize tokenizer
    print("\\n🔤 Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Memory usage comparison
    print("\\n🧠 Memory Usage During Loading:")
    
    n_samples = 1000
    print(f"   Loading {n_samples} samples...")
    
    # Test standard format
    start_memory = measure_memory_usage()
    standard_data = load_standard_dataset(standard_file, n_samples)
    standard_memory = measure_memory_usage() - start_memory
    
    # Clear memory
    del standard_data
    time.sleep(1)
    
    # Test optimized format
    start_memory = measure_memory_usage()
    optimized_data = load_optimized_dataset(optimized_file, n_samples)
    optimized_memory = measure_memory_usage() - start_memory
    
    print(f"   Standard format memory: {standard_memory:.2f} MB")
    print(f"   Optimized format memory: {optimized_memory:.2f} MB")
    print(f"   Memory savings: {standard_memory - optimized_memory:.2f} MB ({((standard_memory - optimized_memory) / max(standard_memory, 1)) * 100:.1f}%)")
    
    # Simulate training memory usage
    print("\\n🏃‍♂️ Simulating Training Memory Usage:")
    
    # Reload for fair comparison
    standard_data = load_standard_dataset(standard_file, 100)
    optimized_data = load_optimized_dataset(optimized_file, 100)
    
    standard_training_memory = simulate_training_memory_usage(standard_data, tokenizer)
    optimized_training_memory = simulate_training_memory_usage(optimized_data, tokenizer)
    
    avg_standard = sum(standard_training_memory) / max(len(standard_training_memory), 1)
    avg_optimized = sum(optimized_training_memory) / max(len(optimized_training_memory), 1)
    
    print(f"   Average training memory (standard): {avg_standard:.2f} MB")
    print(f"   Average training memory (optimized): {avg_optimized:.2f} MB")
    print(f"   Training memory savings: {avg_standard - avg_optimized:.2f} MB ({((avg_standard - avg_optimized) / max(avg_standard, 1)) * 100:.1f}%)")
    
    # Summary
    print("\\n📊 Summary:")
    print(f"   File size reduction: {file_analysis['savings_percentage']:.1f}%")
    print(f"   Loading memory reduction: {((standard_memory - optimized_memory) / max(standard_memory, 1)) * 100:.1f}%")
    print(f"   Training memory reduction: {((avg_standard - avg_optimized) / max(avg_standard, 1)) * 100:.1f}%")
    
    # Recommendations
    print("\\n💡 Recommendations:")
    if file_analysis['savings_percentage'] > 10:
        print("   ✅ Significant file size savings - recommended for large datasets")
    if ((standard_memory - optimized_memory) / standard_memory) * 100 > 5:
        print("   ✅ Noticeable memory savings during loading")
    if ((avg_standard - avg_optimized) / avg_standard) * 100 > 5:
        print("   ✅ Memory efficient for training - can handle larger batch sizes")
    else:
        print("   ℹ️  Memory savings are modest - standard format is fine for small datasets")


def generate_optimized_dataset():
    """Generate optimized format dataset if it doesn't exist."""
    print("🚀 Generating memory-optimized dataset...")
    
    from helpsteer_processor import HelpSteerProcessor
    
    processor = HelpSteerProcessor()
    
    try:
        output_paths = processor.process_helpsteer(
            output_dir="/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed",
            min_score_diff=0.3,
            memory_optimized=True,
            local_data_dir="/home/wangbinrui/research_projects/llama_rlhf/datasets/HelpSteer"
        )
        print(f"✅ Generated optimized dataset: {output_paths['train']}")
        return True
    except Exception as e:
        print(f"❌ Error generating optimized dataset: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare memory usage between dataset formats")
    parser.add_argument("--generate-optimized", action="store_true", 
                       help="Generate optimized dataset if it doesn't exist")
    
    args = parser.parse_args()
    
    if args.generate_optimized:
        if generate_optimized_dataset():
            print("\\n" + "="*60)
            run_memory_comparison()
    else:
        run_memory_comparison()
