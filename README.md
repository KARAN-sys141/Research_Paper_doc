# Context-Aware Hybrid Deep Learning for E-Commerce Intent Prediction

## Overview
A three-tower deep learning architecture that predicts purchasing intent
from e-commerce behavioral data. Built on the Retailrocket dataset
(2.7M events), the model fuses collaborative, sequential, and contextual
signals to output a real-time purchase probability score.

## Problem
96% of e-commerce interactions are passive browsing (24:1 class
imbalance). Traditional models develop majority-class bias and fail to
identify genuine purchase signals.

## Architecture
Three independent towers process different behavioral modalities:

Tower 1 — Collaborative Domain
  Inputs: visitor_id, target_item_id
  Method: Embedding layers → Concatenate → Dense(ReLU)
  Captures: Long-term user-item preference history

Tower 2 — Sequential Domain
  Inputs: [item_t-3, item_t-2, item_t-1]
  Method: Shared Embedding → GlobalAveragePooling1D → Dense(ReLU)
  Captures: Short-term session micro-intent

Tower 3 — Contextual Domain
  Inputs: hour_sin, hour_cos, time_of_day (embedded), is_weekend
  Method: Concatenate → Dense(ReLU)
  Captures: Temporal and environmental shopping context

Fusion Layer
  All three tower outputs concatenated → Dense layers with
  L2 regularization + BatchNorm + Dropout → Sigmoid output

## Dataset
Retailrocket E-Commerce Dataset
Source: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
Files: events.csv, item_properties_part1.csv,
       item_properties_part2.csv, category_tree.csv
Size: 2,756,101 events | 18,827,550 item property records

## Key Features
- Cyclical time encoding: sin(2π×hour/24), cos(2π×hour/24)
  Preserves temporal proximity across midnight boundary
- Sliding window sequences via pandas groupby shift operations
- Focal Loss (gamma=2.0, alpha=0.25) to handle class imbalance
- L2 Weight Decay (lambda=0.001) to prevent overfitting
- Shared item embedding weights between Tower 1 and Tower 2
- Keras Tuner Hyperband for automated hyperparameter search

## Training Configuration
Framework:     TensorFlow / Keras
Optimizer:     Adam
Loss:          BinaryFocalCrossentropy (gamma=2.0, alpha=0.25)
Batch Size:    2048
Callbacks:     EarlyStopping (patience=2), ReduceLROnPlateau (factor=0.5),
               ModelCheckpoint (monitor=val_auc)
Metric:        AUC-ROC (accuracy avoided due to class imbalance)
Best Epoch:    2 (val_auc = 0.8814)

## Results
Model                  | Val AUC
-----------------------|---------
BCE + class_weight     | 0.9057
Focal Loss + L2        | 0.8862
Balanced Dataset       | 0.8801
Master (Full Data)     | 0.8814

Ablation Study (Visitor 1330668, Evening Weekday):
  Content Only  (Tower 2) : 31.58%
  Collab Only   (Tower 1) : 29.31%
  Hybrid Engine (All)     : 36.83%  (+5.25% uplift)

## Notebook Execution Order
1. Install dependencies (ydata-profiling, pdfkit, keras_tuner)
2. Load and concatenate CSV files
3. Run YData ProfileReport for EDA
4. Feature engineering (target, time features, sequences)
5. Build Tower architecture (Collaborative, Sequential, Contextual)
6. Train Model V1 with BCE and class weights
7. Train Model V2 with Focal Loss + L2 regularization
8. Train Model V3 on balanced dataset (3:1 negative sampling)
9. Train Master Model on full dataset
10. Run ablation study (zero-masking per tower)
11. Run 5-user X-Ray validation
12. Run GrandMaster Top-10 recommendation engine
13. Generate paper visualizations

## Dependencies
tensorflow>=2.x
keras-tuner
pandas
numpy
scikit-learn
matplotlib
ydata-profiling
pdfkit
wkhtmltopdf (system package)

## Files Saved
best_context_model.keras    — BCE model best checkpoint
robust_context_model.keras  — Focal Loss + L2 best checkpoint
master_context_model.keras  — Final production model

## Future Work
- FastAPI backend for real-time inference (<100ms target)
- MLOps pipeline with model versioning
- Extend sequence window beyond 3 items
- Address cold-start problem for new users/items
- LSTM/Transformer comparison for sequence modeling

## Authors
Karan Darji — Architecture, implementation, experiments
Jal Patel   — Data analysis, validation
Prof. Karan P. Bhatt — Research supervision

## Institution
Department of Computer Science and Engineering (Data Science)
Vishwakarma Government Engineering College (VGEC)
Gujarat Technological University (GTU), Ahmedabad, India
