# Student Needs Inference
Inference of Student Needs in STEM self-study using facial expression recognition

This repository contains the implementation and data analysis for a research project that infers student needs in STEM self-study environments using facial expression recognition (FACS) and machine learning.

## Key Features
- MLP neural network with 91.2% accuracy in need classification
- Synchronized multimodal data collection (facial AUs, surveys, mouse logs)
- Browser-based experiment interface for STEM self-study simulation
- OpenFace-powered facial action unit extraction
- Frameworks for handling data imbalance in educational contexts

## Repository Structure

├── data/ # Processed datasets
│ ├── alldata_17.csv # Final 17-feature dataset (normalized AU intensities)
│ └── combined_participant_data.csv # Aggregated participant data
│
├── notebooks/ # Analysis and processing workflows
│ ├── preprocessing_1.ipynb → Raw data synchronization & labeling
│ ├── preprocessing_2.ipynb → Feature engineering & normalization
│ ├── factor_analysis.ipynb → AU correlation studies
│ ├── distribution.ipynb → Label distribution analysis
│ ├── train.ipynb → Primary MLP training (91.2% accuracy)
│ ├── train_2.ipynb → Alternative training approaches
│ └── loocv.ipynb → Leave-One-Out Cross-Validation tests
│
├── thesis/ # Thesis document and resources
│ └── GR_ThesisFinal_26002105151.pdf
│
└── README.md # This document


## Key Findings
1. Achieved **91.2% accuracy** classifying 5 student need categories
2. Best performance on "I need an easier question" (96% precision, 94.5% recall)
3. Identified limitations in LOOCV (21.1% accuracy) due to:
   - Small homogeneous sample (5 engineering students)
   - Survey trigger frequency imbalances
   - Cross-participant expression variability
