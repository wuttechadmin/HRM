#!/usr/bin/env python3
import numpy as np
import json
import os
import argparse
import shutil
from datetime import datetime

def repair_dataset(data_path, output_path=None, sample_size=1000):
    """
    Repair Sudoku dataset by ensuring input puzzles match solutions at non-empty positions.
    
    Args:
        data_path: Path to the original dataset directory
        output_path: Path where the repaired dataset will be saved. If None, a new directory will be created.
        sample_size: Number of samples to check before assuming the entire dataset has the same pattern
    """
    if output_path is None:
        # Create a default output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_path + f"_repaired_{timestamp}"
    
    print(f"Input dataset: {data_path}")
    print(f"Output dataset: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process train and test splits
    for split in ['train', 'test']:
        split_path = os.path.join(data_path, split)
        output_split_path = os.path.join(output_path, split)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping")
            continue
        
        os.makedirs(output_split_path, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        
        # Check if the dataset metadata exists
        metadata_path = os.path.join(split_path, 'dataset.json')
        if os.path.exists(metadata_path):
            # Copy metadata file
            shutil.copy2(metadata_path, os.path.join(output_split_path, 'dataset.json'))
        
        # Load the data in batches to save memory
        inputs_path = os.path.join(split_path, 'all__inputs.npy')
        labels_path = os.path.join(split_path, 'all__labels.npy')
        
        # Get the total size first
        inputs_mmap = np.load(inputs_path, mmap_mode='r')
        total_puzzles = len(inputs_mmap)
        print(f"Total puzzles: {total_puzzles}")
        
        # Check a sample to determine if repairs are needed
        sample_size = min(sample_size, total_puzzles)
        print(f"Checking {sample_size} sample puzzles...")
        
        puzzles_with_mismatches = 0
        total_mismatches = 0
        
        # Load a portion of the dataset to check
        sample_inputs = np.array(inputs_mmap[:sample_size])
        sample_labels = np.load(labels_path, mmap_mode='r')[:sample_size]
        
        for i in range(sample_size):
            # Get mask of non-empty cells in input
            mask = sample_inputs[i] != 0
            
            # Find mismatches
            mismatches = (sample_inputs[i][mask] != sample_labels[i][mask])
            num_mismatches = np.sum(mismatches)
            
            if num_mismatches > 0:
                puzzles_with_mismatches += 1
                total_mismatches += num_mismatches
        
        print(f"Found {puzzles_with_mismatches}/{sample_size} sample puzzles with mismatches ({puzzles_with_mismatches/sample_size*100:.2f}%)")
        print(f"Total sample mismatches: {total_mismatches}")
        
        # If no mismatches found in the sample, assume the dataset is correct
        if puzzles_with_mismatches == 0:
            print(f"No mismatches found in sample. Assuming entire dataset is correct.")
            print(f"Copying files without modification...")
            
            # Copy all files directly
            for filename in os.listdir(split_path):
                src_file = os.path.join(split_path, filename)
                dst_file = os.path.join(output_split_path, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
            
            continue
        
        # If mismatches were found, process the entire dataset in chunks
        print(f"Processing full dataset in chunks...")
        
        # Create an output file
        repaired_inputs_path = os.path.join(output_split_path, 'all__inputs.npy')
        shutil.copy2(inputs_path, repaired_inputs_path)
        repaired_inputs = np.load(repaired_inputs_path, mmap_mode='r+')
        
        # Process in chunks to save memory
        chunk_size = 10000
        total_fixed = 0
        
        for chunk_start in range(0, total_puzzles, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_puzzles)
            chunk_size_actual = chunk_end - chunk_start
            
            print(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_puzzles+chunk_size-1)//chunk_size}: "
                  f"puzzles {chunk_start} to {chunk_end-1}")
            
            # Load chunk
            inputs_chunk = np.array(inputs_mmap[chunk_start:chunk_end])
            labels_chunk = np.load(labels_path, mmap_mode='r')[chunk_start:chunk_end]
            
            # Process each puzzle in the chunk
            chunk_fixed = 0
            
            for i in range(chunk_size_actual):
                # Get mask of non-empty cells in input
                mask = inputs_chunk[i] != 0
                
                # Find mismatches
                mismatches = (inputs_chunk[i][mask] != labels_chunk[i][mask])
                num_mismatches = np.sum(mismatches)
                
                if num_mismatches > 0:
                    # Fix mismatches by copying from solution to input
                    mismatch_indices = np.where(mask & (inputs_chunk[i] != labels_chunk[i]))[0]
                    for idx in mismatch_indices:
                        repaired_inputs[chunk_start + i][idx] = labels_chunk[i][idx]
                    
                    chunk_fixed += 1
            
            total_fixed += chunk_fixed
            print(f"  Fixed {chunk_fixed} puzzles in this chunk")
        
        print(f"Total puzzles fixed: {total_fixed}/{total_puzzles} ({total_fixed/total_puzzles*100:.2f}%)")
        
        # Copy other files unchanged
        for filename in os.listdir(split_path):
            if filename != 'all__inputs.npy':  # Skip the file we already processed
                src_file = os.path.join(split_path, filename)
                dst_file = os.path.join(output_split_path, filename)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
    
    print(f"\nDataset repair completed. Repaired dataset saved to: {output_path}")
    return output_path

def validate_repair(original_path, repaired_path, sample_size=1000):
    """Validate that the repair fixed the mismatches."""
    
    validation_results = {}
    
    for split in ['train', 'test']:
        original_split = os.path.join(original_path, split)
        repaired_split = os.path.join(repaired_path, split)
        
        if not (os.path.exists(original_split) and os.path.exists(repaired_split)):
            continue
        
        # Load original data
        orig_inputs_mmap = np.load(os.path.join(original_split, 'all__inputs.npy'), mmap_mode='r')
        
        # Get total size
        total_puzzles = len(orig_inputs_mmap)
        
        # Use a sample of the dataset to validate
        sample_size = min(sample_size, total_puzzles)
        print(f"\nValidating {split} split using {sample_size} sample puzzles...")
        
        # Load samples
        orig_inputs = np.array(orig_inputs_mmap[:sample_size])
        orig_labels = np.load(os.path.join(original_split, 'all__labels.npy'), mmap_mode='r')[:sample_size]
        
        rep_inputs_mmap = np.load(os.path.join(repaired_split, 'all__inputs.npy'), mmap_mode='r')
        rep_inputs = np.array(rep_inputs_mmap[:sample_size])
        rep_labels = np.load(os.path.join(repaired_split, 'all__labels.npy'), mmap_mode='r')[:sample_size]
        
        # Check that labels are unchanged
        labels_unchanged = np.array_equal(orig_labels, rep_labels)
        
        # Check original dataset for mismatches
        orig_puzzles_with_mismatches = 0
        
        for i in range(sample_size):
            # Get mask of non-empty cells in original input
            mask = orig_inputs[i] != 0
            
            # Find mismatches
            mismatches = (orig_inputs[i][mask] != orig_labels[i][mask])
            if np.any(mismatches):
                orig_puzzles_with_mismatches += 1
        
        # Check repaired dataset for mismatches
        rep_puzzles_with_mismatches = 0
        
        for i in range(sample_size):
            # Get mask of non-empty cells in repaired input
            mask = rep_inputs[i] != 0
            
            # Find mismatches
            mismatches = (rep_inputs[i][mask] != rep_labels[i][mask])
            if np.any(mismatches):
                rep_puzzles_with_mismatches += 1
        
        validation_results[split] = {
            'labels_unchanged': labels_unchanged,
            'total_puzzles': total_puzzles,
            'sample_size': sample_size,
            'orig_puzzles_with_mismatches': orig_puzzles_with_mismatches,
            'orig_mismatch_percentage': orig_puzzles_with_mismatches / sample_size * 100,
            'rep_puzzles_with_mismatches': rep_puzzles_with_mismatches,
            'rep_mismatch_percentage': rep_puzzles_with_mismatches / sample_size * 100
        }
    
    print("\nValidation Results:")
    print("=" * 40)
    
    for split, results in validation_results.items():
        print(f"\n{split.upper()} Split:")
        print(f"Solution labels unchanged: {'✅' if results['labels_unchanged'] else '❌'}")
        print(f"Original dataset - Puzzles with mismatches: {results['orig_puzzles_with_mismatches']}/{results['sample_size']} ({results['orig_mismatch_percentage']:.2f}%)")
        print(f"Repaired dataset - Puzzles with mismatches: {results['rep_puzzles_with_mismatches']}/{results['sample_size']} ({results['rep_mismatch_percentage']:.2f}%)")
        
        if results['rep_puzzles_with_mismatches'] < results['orig_puzzles_with_mismatches']:
            improvement = results['orig_puzzles_with_mismatches'] - results['rep_puzzles_with_mismatches']
            print(f"✅ Improvement: Fixed {improvement} puzzles ({improvement/results['sample_size']*100:.2f}% of sample)")
        elif results['rep_puzzles_with_mismatches'] == results['orig_puzzles_with_mismatches'] == 0:
            print(f"✅ No mismatches in either dataset")
        else:
            print(f"❌ No improvement: Original had {results['orig_puzzles_with_mismatches']} mismatches, Repaired has {results['rep_puzzles_with_mismatches']}")
    
    all_fixed = all(r['rep_puzzles_with_mismatches'] == 0 for r in validation_results.values())
    print("\n" + "=" * 40)
    print(f"{'✅' if all_fixed else '❌'} Overall: {'All mismatches fixed' if all_fixed else 'Some mismatches remain'}")
    
    return all_fixed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair Sudoku dataset to ensure inputs match solutions")
    parser.add_argument("--data_path", type=str, default="/Users/robertburkhall/Development/HRM/data/sudoku-extreme-1k-aug-1000", 
                        help="Path to the dataset directory")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path where repaired dataset will be saved. Default: original_path_repaired_timestamp")
    args = parser.parse_args()
    
    repaired_path = repair_dataset(args.data_path, args.output_path)
    validate_repair(args.data_path, repaired_path)
