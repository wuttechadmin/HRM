# HRM Project Structure

This document outlines the organized project structure for the Hierarchical Relational Model (HRM) codebase.

## Directory Structure

```
HRM/
├── assets/              # Images and static assets
│   └── hrm.png
│   └── npyjs.js
├── config/              # Configuration files
│   ├── cfg_pretrain.yaml
│   └── arch/
│       └── hrm_v1.yaml
├── data/                # Dataset files
│   └── sudoku-extreme-1k-aug-1000/
│       ├── identifiers.json
│       ├── test/
│       └── train/
├── dataset/             # Dataset generation scripts
│   ├── build_arc_dataset.py
│   ├── build_maze_dataset.py
│   ├── build_sudoku_dataset.py
│   └── common.py
├── docs/                # Documentation
│   ├── LICENSE
│   ├── MACOS_GUIDE.md
│   ├── README.md
│   ├── colab_dataset_comparison.md
│   ├── hrm_project_guidelines.md
│   ├── mps_optimization_strategies.md
│   └── project_structure.md
├── models/              # Model architecture definitions
│   ├── common.py
│   ├── layers.py
│   ├── losses.py
│   ├── sparse_embedding.py
│   └── hrm/
│       └── hrm_act_v1.py
├── notebooks/           # Jupyter notebooks (DO NOT MODIFY WHILE TRAINING)
│   ├── arc_eval.ipynb
│   ├── dataset_verification.ipynb
│   └── colab/
│       └── HRM_Sudoku_1k_T4.ipynb
├── results/             # Output and evaluation results
│   ├── dataset_check_results.txt
│   ├── training_history.png
│   ├── training_results.png
│   └── validation_results.txt
├── scripts/             # Utility scripts
│   ├── apply_mps_patch.py
│   ├── check_dataset.py
│   ├── check_dataset_quality.py
│   ├── create_macos_version.py
│   ├── evaluate.py
│   ├── examine_dataset.py
│   ├── inspect_model.py
│   ├── manual_checkpoint_cell.txt
│   ├── pretrain.py
│   ├── pretrain_macos.py
│   ├── pretrain.py.bak
│   ├── puzzle_dataset.py
│   ├── quick_dataset_check.py
│   ├── repair_dataset.py
│   ├── run_macos.py
│   ├── run_macos_simple.py
│   ├── save_current_model.py
│   ├── simple_test.py
│   ├── simple_validation.py
│   ├── train_sudoku_enhanced.py
│   ├── validate_data.py
│   ├── validate_data_new.py
│   ├── validate_sudoku.py
│   └── verify_dataset.py
├── tests/               # Test files
│   ├── test_colab_compatibility.ipynb
│   ├── test_device.py
│   ├── test_mps_training.py
│   ├── test_mps_training_file.py
│   ├── test_notebook_cell.py
│   ├── test_results.txt
│   ├── test_sudoku_training.py
│   └── test_training.py
├── utils/               # Utility functions
│   └── functions.py
├── web/                 # Web visualizations
│   └── puzzle_visualizer.html
├── requirements.txt     # Project dependencies
└── HRM_Sudoku_MPS.ipynb # Active training notebook (DO NOT MODIFY)
```

## Important Notes

1. **DO NOT MODIFY any notebooks while training is in progress**
2. The root notebook `HRM_Sudoku_MPS.ipynb` is actively used for training and should not be moved
3. Virtual environments (`hrm_venv` and `hrm_venv_new`) are kept at the root level
4. Files have been copied to their respective directories but original files remain in place to prevent disrupting ongoing training

## Usage Guidelines

- Use the organized folders for any new development
- Add new notebooks to the `notebooks` folder
- Add new scripts to the `scripts` folder
- Add new documentation to the `docs` folder
- Add new test files to the `tests` folder

This structure improves organization while maintaining compatibility with the current training process.
