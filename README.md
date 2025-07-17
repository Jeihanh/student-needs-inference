# Inference of Student Needs in STEM Self-Study using Facial Expression Recognition

## Project Description
This repository contains the implementation and data analysis for a research project that infers student needs in STEM self-study environments using facial expression recognition (FACS) and machine learning. The system integrates facial action unit analysis with self-reported needs surveys to classify student cognitive and emotional states during learning sessions.

## Files Included
- 'alldata_17.csv' - Final dataset with 17 normalized AU intensity features
- combined_participant_data.csv - Aggregated participant data from all sessions
- distribution.ipynb - Analysis of label distribution and data imbalances
- factor_analysis.ipynb - Action Unit correlation studies
- loocv.ipynb - Leave-One-Out Cross-Validation implementation and tests
- preprocessing_1.ipynb - Raw data synchronization and labeling workflow
- preprocessing_2.ipynb - Feature engineering and normalization pipeline
- train.ipynb - Primary MLP training achieving 91.2% accuracy
- train_2.ipynb - Alternative training approaches and model variations

## Key Features
- MLP neural network with 91.2% accuracy in classifying 5 student need categories
- Synchronized multimodal data collection (facial AUs, surveys, mouse logs)
- Browser-based experiment interface for STEM self-study simulation
- OpenFace-powered facial action unit extraction
- Frameworks for handling data imbalance in educational contexts

## Key Findings
- Achieved 91.2% accuracy classifying 5 student need categories
- Best performance on "I need an easier question" (96% precision, 94.5% recall)
- LOOCV revealed performance drop to 21.1% due to:
  - Small homogeneous participant pool (5 male engineering students)
  - Survey trigger frequency imbalances
  - Cross-participant expression variability
- Most effective Action Units for detection: AU1, AU2, AU4, AU6, AU12

## Usage Instructions
1. Data preprocessing:
   - Run preprocessing_1.ipynb for data synchronization and labeling
   - Run preprocessing_2.ipynb for feature engineering
2. Data analysis:
   - distribution.ipynb for label distribution
   - factor_analysis.ipynb for AU correlations
3. Model training:
   - train.ipynb for primary MLP implementation
   - train_2.ipynb for alternative approaches
   - loocv.ipynb for cross-validation tests

## Requirements
- Python 3.8+
- TensorFlow 2.12
- OpenFace 2.0
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
