#!/usr/bin/env python3
"""
Test script to verify P-tuning integration in gen_hh_answers.py
"""

import sys
import os
import torch

# Add the FastChat directory to the path
sys.path.append('/home/wangbinrui/research_projects/llama_rlhf/FastChat')

def test_prompt_embedding_functions():
    """Test the P-tuning helper functions"""
    try:
        from fastchat.llm_judge.gen_hh_answers import load_prompt_embeddings, apply_prompt_embeddings
        
        print("✓ Successfully imported P-tuning functions")
        
        # Test load_prompt_embeddings with non-existent file
        result = load_prompt_embeddings(None, 'cpu')
        assert result is None, "Should return None for None path"
        print("✓ load_prompt_embeddings handles None path correctly")
        
        result = load_prompt_embeddings('/non/existent/path.pt', 'cpu')
        assert result is None, "Should return None for non-existent path"
        print("✓ load_prompt_embeddings handles non-existent path correctly")
        
        # Create a mock prompt embedding for testing
        mock_prompt_embedding = torch.randn(10, 768)  # 10 prompt tokens, 768 hidden size
        
        # Test apply_prompt_embeddings
        batch_size = 2
        seq_len = 20
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
        
        tokenizer = MockTokenizer()
        
        new_input_ids, new_attention_mask, prompt_emb = apply_prompt_embeddings(
            None, tokenizer, input_ids, attention_mask, mock_prompt_embedding
        )
        
        expected_seq_len = seq_len + mock_prompt_embedding.shape[0]
        assert new_input_ids.shape == (batch_size, expected_seq_len), f"Expected shape {(batch_size, expected_seq_len)}, got {new_input_ids.shape}"
        assert new_attention_mask.shape == (batch_size, expected_seq_len), f"Expected shape {(batch_size, expected_seq_len)}, got {new_attention_mask.shape}"
        assert prompt_emb.shape == (batch_size, mock_prompt_embedding.shape[0], mock_prompt_embedding.shape[1])
        
        print("✓ apply_prompt_embeddings works correctly")
        print("✓ All P-tuning function tests passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing P-tuning functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_prompt_embedding():
    """Create a sample prompt embedding file for testing"""
    sample_embedding = torch.randn(5, 4096)  # 5 prompt tokens, 4096 hidden size (typical for LLaMA)
    
    # Save as different formats to test loading
    torch.save(sample_embedding, '/tmp/test_prompt_embedding.pt')
    torch.save({'prompt_embeddings': sample_embedding}, '/tmp/test_prompt_embedding_dict.pt')
    
    print("✓ Created sample prompt embedding files:")
    print("  - /tmp/test_prompt_embedding.pt (direct tensor)")
    print("  - /tmp/test_prompt_embedding_dict.pt (dictionary format)")

def test_command_line_args():
    """Test that the new command line argument works"""
    try:
        import subprocess
        
        # Test help output includes the new argument
        result = subprocess.run([
            'python3', 
            '/home/wangbinrui/research_projects/llama_rlhf/FastChat/fastchat/llm_judge/gen_hh_answers.py',
            '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--prompt-embedding-path' in result.stdout:
            print("✓ New --prompt-embedding-path argument is available")
            return True
        else:
            print("✗ --prompt-embedding-path argument not found in help output")
            return False
            
    except Exception as e:
        print(f"✗ Error testing command line args: {e}")
        return False

if __name__ == "__main__":
    print("Testing P-tuning integration...")
    print("=" * 50)
    
    # Test 1: Function imports and basic functionality
    test1_passed = test_prompt_embedding_functions()
    
    # Test 2: Create sample files
    create_sample_prompt_embedding()
    
    # Test 3: Command line arguments
    test3_passed = test_command_line_args()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"✓ Function tests: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✓ Command line args: {'PASSED' if test3_passed else 'FAILED'}")
    
    if test1_passed and test3_passed:
        print("\n🎉 All tests PASSED! P-tuning integration is ready to use.")
        print("\nUsage example:")
        print("python3 gen_hh_answers.py --model-path /path/to/model --model-id my-model --prompt-embedding-path /tmp/test_prompt_embedding.pt")
    else:
        print("\n❌ Some tests FAILED. Please check the implementation.")
        sys.exit(1)
