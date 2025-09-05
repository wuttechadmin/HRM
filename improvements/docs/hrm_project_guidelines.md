# HRM Project Guidelines and System Prompt

## CRITICAL SAFETY GUIDELINES

### ⚠️ NEVER MODIFY RUNNING NOTEBOOKS

- **HIGHEST PRIORITY RULE**: NEVER suggest modifications to a notebook that is currently executing training.
- **IDENTIFICATION**: Always check for indicators of active training (iteration counters, progress reports, etc.)
- **CONSEQUENCES**: Modifying a running notebook can crash training, corrupt model state, and waste hours/days of computation.
- **ALTERNATIVES**: Create separate files for suggestions or wait until training completes.

## Project Overview

This repository contains a Hierarchical Relational Model (HRM) implementation for solving puzzles like Sudoku using deep learning techniques.

### Key Components

- **Model Architecture**: Transformer-based HRM with relational understanding capabilities
- **Training Process**: Progressive complexity training with PyTorch on MacOS using MPS acceleration
- **Dataset**: Puzzle datasets with varying difficulty levels, primarily focused on Sudoku

## Interaction Guidelines

### When Assisting with Training

1. **OBSERVE FIRST**: Always check if training is in progress before suggesting code changes
2. **DOCUMENTATION ONLY**: If training is running, only provide documentation or analysis
3. **SEPARATE FILES**: Create separate files for suggestions rather than modifying notebooks
4. **PATIENCE**: Wait for natural training pauses before suggesting modifications

### Safe Modification Practices

1. Only suggest notebook modifications when:
   - The notebook is not currently executing
   - The user explicitly requests modifications
   - You have confirmed training is not in progress

2. For running notebooks:
   - Focus on analyzing outputs and metrics
   - Suggest improvements for future training runs
   - Create external utility scripts if needed

### Memory and Performance Optimization

When suggesting optimizations for MPS-based training:

1. **Batch Size Tuning**: Suggest powers of 2 for memory boundary alignment
2. **Model Dimensions**: Recommend dimensions that are also powers of 2
3. **Mixed Precision**: Consider float16 where appropriate for Apple Silicon
4. **Memory Monitoring**: Add utilities that don't interfere with running processes

## Technical Guidelines

### MacOS-Specific Considerations

- **MPS Acceleration**: Optimize for Metal Performance Shaders
- **Memory Management**: Be mindful of memory constraints on Apple Silicon
- **Checkpointing**: Ensure robust checkpoint systems before starting long runs

### Training Stability

- **Interrupt Handling**: Implement proper KeyboardInterrupt handling in training loops
- **Regular Checkpoints**: Save state at regular intervals, not just on improvements
- **Resumable Training**: Ensure training can be stopped and resumed cleanly

## Communication Protocol

- **Clear Status Checks**: "Is training currently running?" before suggesting code changes
- **Non-Disruptive Analysis**: Provide insights without suggesting disruptive actions
- **Future-Oriented Advice**: Frame suggestions as "for your next training run"

---

By following these guidelines, we can ensure productive collaboration without risking disruption to valuable training processes. The safety and integrity of running computations must always be the highest priority.
