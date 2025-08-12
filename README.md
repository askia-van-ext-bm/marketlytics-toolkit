# Marketlytics Toolkit

Welcome to the **Marketlytics Toolkit**, a centralized repository designed for Back Market's Marketing Analytics team. This library contains reusable SQL queries, Python functions, and modeling tools to support efficient, scalable, and insightful marketing analyses.

## Purpose

This repository aims to:
- Centralize commonly used analytics tools and queries
- Promote code reuse and consistency across marketing projects
- Accelerate insight generation through pre-built models and utilities

## Contents

- `bigquery/`: Parameterized BQ queries for campaign performance, customer segmentation, and more
- `functions/`: Python utilities for data cleaning, feature engineering, and visualization
- `models/`: Predefined modeling workflows for uplift modeling, etc.

## How to Import in Google Colab

To use this library in Google Colab:

### 1. Clone the repository

If the repository is **public**:
```python
!git clone https://github.com/backmarket/marketlytics-toolkit.git
```

If the repo is private, generate a GitHub personal access token and use the following command:
```python
!git clone https://<YOUR_TOKEN>@github.com/askia-van-ext-bm/marketlytics-toolkit.git
```
Replace <YOUR_TOKEN> with your actual token.

### 2. Add the repo to your Python path

```
import sys
sys.path.append('/content/marketlytics-toolkit')
```

### 3. Import modules (exemple)

```
from functions.data_cleaning import clean_campaign_data
from models.uplift_model import run_uplift_model
```

### 4. (Optional) Enable autoreload for development

```
%load_ext autoreload
%autoreload 2
```

## Contribution Guidelines

 - Keep functions modular and well-documented
 - Use consistent naming conventions
 - Submit pull requests with clear descriptions

## Contact
For questions or suggestions, reach out to the Marketing Analytics team via Slack or email.
