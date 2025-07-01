#!/usr/bin/env python3
"""
HelpSteer Dataset Processor

Converts NVIDIA HelpSteer dataset to ultrafeedback format for training.
HelpSteer format: {prompt, response, helpfulness, correctness, coherence, complexity, verbosity}
UltraFeedback format: {prompt, chosen, rejected}

Strategy:
1. Group responses by prompt
2. Calculate composite scores based on weighted combination of 5 metrics
3. Create pairwise comparisons where higher-scored responses are "chosen"
4. Export in ultrafeedback-compatible format
"""

import json
import argparse
from datasets import load_dataset
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import os


class HelpSteerProcessor:
    def __init__(self, 
                 helpfulness_weight: float = 0.35,
                 correctness_weight: float = 0.30,
                 coherence_weight: float = 0.20,
                 complexity_weight: float = 0.075,
                 verbosity_weight: float = 0.075):
        """
        Initialize the processor with weights for combining metrics.
        
        Args:
            helpfulness_weight: Weight for helpfulness score
            correctness_weight: Weight for correctness score  
            coherence_weight: Weight for coherence score
            complexity_weight: Weight for complexity score
            verbosity_weight: Weight for verbosity score
        """
        self.weights = {
            'helpfulness': helpfulness_weight,
            'correctness': correctness_weight,
            'coherence': coherence_weight,
            'complexity': complexity_weight,
            'verbosity': verbosity_weight
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total_weight

    def calculate_composite_score(self, example: Dict) -> float:
        """Calculate weighted composite score from the 5 metrics."""
        score = 0.0
        for metric, weight in self.weights.items():
            score += example[metric] * weight
        return score

    def load_helpsteer_data_from_local(self, data_dir: str) -> Dict[str, List[Dict]]:
        """
        Load HelpSteer dataset from local JSONL files.
        
        Args:
            data_dir: Directory containing train.jsonl.gz and validation.jsonl.gz
        
        Returns:
            Dict mapping prompts to lists of response examples
        """
        import gzip
        
        print(f"Loading HelpSteer dataset from local directory: {data_dir}")
        
        all_examples = []
        
        # Load train split
        train_file = os.path.join(data_dir, "train.jsonl.gz")
        if os.path.exists(train_file):
            print(f"Loading training data from {train_file}")
            with gzip.open(train_file, 'rt', encoding='utf-8') as f:
                train_examples = [json.loads(line.strip()) for line in f if line.strip()]
                all_examples.extend(train_examples)
                print(f"Loaded {len(train_examples)} training examples")
        
        # Load validation split
        val_file = os.path.join(data_dir, "validation.jsonl.gz")
        if os.path.exists(val_file):
            print(f"Loading validation data from {val_file}")
            with gzip.open(val_file, 'rt', encoding='utf-8') as f:
                val_examples = [json.loads(line.strip()) for line in f if line.strip()]
                all_examples.extend(val_examples)
                print(f"Loaded {len(val_examples)} validation examples")
        
        print(f"Total examples loaded: {len(all_examples)}")
        
        # Group by prompt
        prompt_groups = defaultdict(list)
        for example in tqdm(all_examples, desc="Grouping by prompt"):
            prompt = example['prompt'].strip()
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(example)
            
            # Add composite score to example
            example_with_score = dict(example)
            example_with_score['composite_score'] = composite_score
            
            prompt_groups[prompt].append(example_with_score)
        
        print(f"Found {len(prompt_groups)} unique prompts")
        
        # Print statistics
        response_counts = [len(responses) for responses in prompt_groups.values()]
        print(f"Responses per prompt - Mean: {np.mean(response_counts):.2f}, "
              f"Min: {min(response_counts)}, Max: {max(response_counts)}")
        
        return dict(prompt_groups)

    def load_helpsteer_data(self, cache_dir: str = None, local_data_dir: str = None) -> Dict[str, List[Dict]]:
        """
        Load HelpSteer dataset and group responses by prompt.
        
        Args:
            cache_dir: Cache directory for HuggingFace datasets
            local_data_dir: Local directory containing HelpSteer files
        
        Returns:
            Dict mapping prompts to lists of response examples
        """
        # Try local files first if provided
        if local_data_dir and os.path.exists(local_data_dir):
            return self.load_helpsteer_data_from_local(local_data_dir)
        
        print("Loading HelpSteer dataset from HuggingFace...")
        
        if local_data_dir is not None:
            # Load from local files
            return self.load_helpsteer_data_from_local(local_data_dir)
        
        try:
            dataset = load_dataset("nvidia/HelpSteer", cache_dir=cache_dir)
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
            print("Please ensure you have internet connection and datasets library installed.")
            raise
        
        # Combine train and validation splits
        all_examples = []
        if 'train' in dataset:
            all_examples.extend(list(dataset['train']))
            print(f"Loaded {len(dataset['train'])} training examples")
        if 'validation' in dataset:
            all_examples.extend(list(dataset['validation']))
            print(f"Loaded {len(dataset['validation'])} validation examples")
        
        print(f"Total examples loaded: {len(all_examples)}")
        
        # Group by prompt
        prompt_groups = defaultdict(list)
        for example in tqdm(all_examples, desc="Grouping by prompt"):
            prompt = example['prompt'].strip()
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(example)
            
            # Add composite score to example
            example_with_score = dict(example)
            example_with_score['composite_score'] = composite_score
            
            prompt_groups[prompt].append(example_with_score)
        
        print(f"Found {len(prompt_groups)} unique prompts")
        
        # Print statistics
        response_counts = [len(responses) for responses in prompt_groups.values()]
        print(f"Responses per prompt - Mean: {np.mean(response_counts):.2f}, "
              f"Min: {min(response_counts)}, Max: {max(response_counts)}")
        
        return dict(prompt_groups)

    def create_preference_pairs(self, prompt_groups: Dict[str, List[Dict]], 
                              min_score_diff: float = 0.5) -> List[Dict]:
        """
        Create preference pairs from grouped responses.
        
        Args:
            prompt_groups: Dict mapping prompts to response lists
            min_score_diff: Minimum score difference required for pair creation
            
        Returns:
            List of preference pairs in ultrafeedback format
        """
        preference_pairs = []
        
        print(f"Creating preference pairs with minimum score difference: {min_score_diff}")
        
        total_pairs = 0
        valid_pairs = 0
        
        for prompt, responses in tqdm(prompt_groups.items(), desc="Creating pairs"):
            if len(responses) < 2:
                continue
            
            # Sort responses by composite score (descending)
            responses_sorted = sorted(responses, key=lambda x: x['composite_score'], reverse=True)
            
            # Create pairs: higher-scored vs lower-scored
            for i in range(len(responses_sorted)):
                for j in range(i + 1, len(responses_sorted)):
                    total_pairs += 1
                    
                    chosen_response = responses_sorted[i]
                    rejected_response = responses_sorted[j]
                    
                    score_diff = chosen_response['composite_score'] - rejected_response['composite_score']
                    
                    # Only create pair if score difference is significant enough
                    if score_diff >= min_score_diff:
                        pair = {
                            'prompt': prompt,
                            'chosen': prompt + chosen_response['response'],
                            'rejected': prompt + rejected_response['response'],
                            'chosen_score': chosen_response['composite_score'],
                            'rejected_score': rejected_response['composite_score'],
                            'score_difference': score_diff,
                            'chosen_metrics': {
                                'helpfulness': chosen_response['helpfulness'],
                                'correctness': chosen_response['correctness'],
                                'coherence': chosen_response['coherence'],
                                'complexity': chosen_response['complexity'],
                                'verbosity': chosen_response['verbosity']
                            },
                            'rejected_metrics': {
                                'helpfulness': rejected_response['helpfulness'],
                                'correctness': rejected_response['correctness'],
                                'coherence': rejected_response['coherence'],
                                'complexity': rejected_response['complexity'],
                                'verbosity': rejected_response['verbosity']
                            }
                        }
                        preference_pairs.append(pair)
                        valid_pairs += 1
        
        print(f"Created {valid_pairs} valid pairs out of {total_pairs} total possible pairs")
        print(f"Filtering rate: {valid_pairs/max(total_pairs, 1)*100:.1f}%")
        
        return preference_pairs

    def create_ultrafeedback_format(self, preference_pairs: List[Dict]) -> List[Dict]:
        """
        Convert preference pairs to clean ultrafeedback format.
        
        Args:
            preference_pairs: List of preference pairs with scores and metrics
            
        Returns:
            List of examples in ultrafeedback format
        """
        ultrafeedback_data = []
        
        for pair in preference_pairs:
            # Create the basic ultrafeedback format
            ultrafeedback_example = {
                'prompt': pair['prompt'],
                'chosen': pair['chosen'],
                'rejected': pair['rejected']
            }
            ultrafeedback_data.append(ultrafeedback_example)
        
        return ultrafeedback_data

    def create_ultrafeedback_format_optimized(self, preference_pairs: List[Dict]) -> List[Dict]:
        """
        Convert preference pairs to memory-optimized ultrafeedback format.
        
        This format stores prompt and responses separately to reduce memory usage.
        
        Args:
            preference_pairs: List of preference pairs with scores and metrics
            
        Returns:
            List of examples in memory-optimized format
        """
        optimized_data = []
        
        for pair in preference_pairs:
            # Extract response parts (remove prompt from chosen/rejected)
            prompt = pair['prompt']
            chosen_full = pair['chosen']
            rejected_full = pair['rejected']
            
            # Remove prompt from responses to avoid duplication
            if chosen_full.startswith(prompt):
                chosen_response = chosen_full[len(prompt):]
            else:
                chosen_response = chosen_full
                
            if rejected_full.startswith(prompt):
                rejected_response = rejected_full[len(prompt):]
            else:
                rejected_response = rejected_full
            
            # Create memory-optimized format
            optimized_example = {
                'prompt': prompt,
                'chosen_response': chosen_response,
                'rejected_response': rejected_response
            }
            optimized_data.append(optimized_example)
        
        return optimized_data

    def save_dataset(self, data: List[Dict], output_path: str, include_metadata: bool = False):
        """
        Save dataset to JSONL file.
        
        Args:
            data: List of examples to save
            output_path: Output file path
            include_metadata: Whether to include score metadata in output
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                if include_metadata:
                    # Save with all metadata for analysis
                    f.write(json.dumps(example) + '\n')
                else:
                    # Save only ultrafeedback format
                    ultrafeedback_example = {
                        'prompt': example['prompt'],
                        'chosen': example['chosen'],
                        'rejected': example['rejected']
                    }
                    f.write(json.dumps(ultrafeedback_example) + '\n')
        
        print(f"Saved {len(data)} examples to {output_path}")

    def save_dataset_optimized(self, data: List[Dict], output_path: str, format_type: str = 'standard'):
        """
        Save dataset to JSONL file with optional memory optimization.
        
        Args:
            data: List of examples to save
            output_path: Output file path
            format_type: 'standard' for regular format, 'optimized' for memory-optimized format
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                if format_type == 'optimized':
                    # Save memory-optimized format
                    optimized_example = {
                        'prompt': example['prompt'],
                        'chosen_response': example.get('chosen_response', example.get('chosen', '')),
                        'rejected_response': example.get('rejected_response', example.get('rejected', ''))
                    }
                    # Preserve any additional metadata
                    for key, value in example.items():
                        if key not in ['prompt', 'chosen', 'rejected', 'chosen_response', 'rejected_response']:
                            optimized_example[key] = value
                    f.write(json.dumps(optimized_example) + '\n')
                else:
                    # Save standard ultrafeedback format
                    ultrafeedback_example = {
                        'prompt': example['prompt'],
                        'chosen': example['chosen'],
                        'rejected': example['rejected']
                    }
                    f.write(json.dumps(ultrafeedback_example) + '\n')
        
        print(f"Saved {len(data)} examples to {output_path} ({format_type} format)")

    def train_test_split(self, data: List[Dict], test_size: float = 0.1, 
                        random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train and test sets.
        
        Args:
            data: Full dataset
            test_size: Fraction for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        np.random.seed(random_seed)
        
        # Shuffle data
        indices = np.random.permutation(len(data))
        split_idx = int(len(data) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        print(f"Split: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data

    def process_helpsteer(self, 
                         output_dir: str = "./datasets/helpsteer_processed",
                         min_score_diff: float = 0.5,
                         test_size: float = 0.1,
                         save_metadata: bool = True,
                         cache_dir: str = None,
                         local_data_dir: str = None,
                         memory_optimized: bool = False) -> Dict[str, str]:
        """
        Complete processing pipeline for HelpSteer dataset.
        
        Args:
            output_dir: Directory to save processed files
            min_score_diff: Minimum score difference for pair creation
            test_size: Fraction for test set
            save_metadata: Whether to save metadata files
            cache_dir: Cache directory for HuggingFace datasets
            local_data_dir: Local directory containing HelpSteer files
            memory_optimized: Whether to use memory-optimized format
            
        Returns:
            Dict with paths to created files
        """
        print("="*60)
        print("🚀 HelpSteer Dataset Processing Pipeline")
        print("="*60)
        
        # Step 1: Load data
        prompt_groups = self.load_helpsteer_data(cache_dir=cache_dir, local_data_dir=local_data_dir)
        
        # Step 2: Create preference pairs
        preference_pairs = self.create_preference_pairs(prompt_groups, min_score_diff=min_score_diff)
        
        if len(preference_pairs) == 0:
            raise ValueError("No valid preference pairs created. Try reducing min_score_diff.")
        
        # Step 3: Train/test split
        train_data, test_data = self.train_test_split(preference_pairs, test_size=test_size)
        
        # Step 4: Convert to appropriate format
        if memory_optimized:
            print("🚀 Using memory-optimized format")
            train_data_formatted = self.create_ultrafeedback_format_optimized(train_data)
            test_data_formatted = self.create_ultrafeedback_format_optimized(test_data)
            format_suffix = "_optimized"
        else:
            print("📝 Using standard ultrafeedback format")
            train_data_formatted = self.create_ultrafeedback_format(train_data)
            test_data_formatted = self.create_ultrafeedback_format(test_data)
            format_suffix = ""
        
        # Step 5: Save files
        output_paths = {}
        
        # Save main training files
        train_path = os.path.join(output_dir, f"train_prefs_helpsteer{format_suffix}.jsonl")
        test_path = os.path.join(output_dir, f"test_prefs_helpsteer{format_suffix}.jsonl")
        
        format_type = 'optimized' if memory_optimized else 'standard'
        self.save_dataset_optimized(train_data_formatted, train_path, format_type=format_type)
        self.save_dataset_optimized(test_data_formatted, test_path, format_type=format_type)
        
        output_paths['train'] = train_path
        output_paths['test'] = test_path
        
        # Save metadata files if requested
        if save_metadata:
            train_meta_path = os.path.join(output_dir, "train_prefs_helpsteer_with_metadata.jsonl")
            test_meta_path = os.path.join(output_dir, "test_prefs_helpsteer_with_metadata.jsonl")
            
            self.save_dataset(train_data, train_meta_path, include_metadata=True)
            self.save_dataset(test_data, test_meta_path, include_metadata=True)
            
            output_paths['train_metadata'] = train_meta_path
            output_paths['test_metadata'] = test_meta_path
        
        # Save processing statistics
        stats_path = os.path.join(output_dir, "processing_stats.json")
        stats = {
            'total_prompts': len(prompt_groups),
            'total_preference_pairs': len(preference_pairs),
            'train_pairs': len(train_data),
            'test_pairs': len(test_data),
            'min_score_diff': min_score_diff,
            'weights': self.weights,
            'test_size': test_size
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        output_paths['stats'] = stats_path
        
        print("\n" + "="*60)
        print("✅ Processing Complete!")
        print("="*60)
        print(f"📊 Statistics:")
        print(f"   • Total prompts: {len(prompt_groups)}")
        print(f"   • Total preference pairs: {len(preference_pairs)}")
        print(f"   • Training pairs: {len(train_data)}")
        print(f"   • Test pairs: {len(test_data)}")
        print(f"   • Score weights: {self.weights}")
        print(f"\n📁 Output files:")
        for name, path in output_paths.items():
            print(f"   • {name}: {path}")
        print("="*60)
        
        return output_paths


def main():
    parser = argparse.ArgumentParser(description="Process HelpSteer dataset to ultrafeedback format")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="/home/wangbinrui/research_projects/llama_rlhf/datasets/helpsteer_processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--min-score-diff", 
        type=float, 
        default=0.5,
        help="Minimum composite score difference for creating preference pairs"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.1,
        help="Fraction of data to use for test set"
    )
    parser.add_argument(
        "--helpfulness-weight", 
        type=float, 
        default=0.35,
        help="Weight for helpfulness score"
    )
    parser.add_argument(
        "--correctness-weight", 
        type=float, 
        default=0.30,
        help="Weight for correctness score"
    )
    parser.add_argument(
        "--coherence-weight", 
        type=float, 
        default=0.20,
        help="Weight for coherence score"
    )
    parser.add_argument(
        "--complexity-weight", 
        type=float, 
        default=0.075,
        help="Weight for complexity score"
    )
    parser.add_argument(
        "--verbosity-weight", 
        type=float, 
        default=0.075,
        help="Weight for verbosity score"
    )
    parser.add_argument(
        "--no-metadata", 
        action="store_true",
        help="Skip saving metadata files"
    )
    parser.add_argument(
        "--memory-optimized", 
        action="store_true",
        help="Use memory-optimized format (stores prompt and responses separately)"
    )
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Local directory containing HelpSteer dataset files"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=None,
        help="Cache directory for HuggingFace datasets"
    )
    
    args = parser.parse_args()
    
    # Initialize processor with custom weights
    processor = HelpSteerProcessor(
        helpfulness_weight=args.helpfulness_weight,
        correctness_weight=args.correctness_weight,
        coherence_weight=args.coherence_weight,
        complexity_weight=args.complexity_weight,
        verbosity_weight=args.verbosity_weight
    )
    
    # Process dataset
    try:
        output_paths = processor.process_helpsteer(
            output_dir=args.output_dir,
            min_score_diff=args.min_score_diff,
            test_size=args.test_size,
            save_metadata=not args.no_metadata,
            cache_dir=args.cache_dir,
            local_data_dir=args.local_data_dir,
            memory_optimized=args.memory_optimized
        )
        
        print(f"\n🎉 Successfully processed HelpSteer dataset!")
        print(f"📁 Training file: {output_paths['train']}")
        print(f"📁 Test file: {output_paths['test']}")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    main()
