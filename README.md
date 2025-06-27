# Enhancing Collaborative Filtering based Course Recommendations by Exploiting Time-to-Event Information with Survival Analysis

This repository contains the code to reproduce the experiments for Enhancing Collaborative Filtering based Course Recommendations by Exploiting Time-to-Event Information with Survival Analysis.

## Abstract 
> Massive Open Online Courses (MOOCs) are emerging as a popular alternative to traditional education, offering learners the flexibility to access a wide range of courses from various disciplines, anytime and anywhere. Despite this accessibility, a significant number of enrollments in MOOCs result in dropouts. To enhance learner engagement, it is crucial to recommend courses that align with their preferences and needs. Course Recommender Systems (RSs) can play an important role in this by modeling learners' preferences based on their previous interactions within the MOOC platform. Time-to-dropout and time-to-completion in MOOCs, like other time-to-event prediction tasks, can be effectively modeled using survival analysis (SA) methods. In this study, we apply SA methods to improve collaborative filtering recommendation performance by considering time-to-event in the context of MOOCs. Our proposed approach demonstrates superior performance compared to collaborative filtering methods trained based on learners' interactions with MOOCs, as evidenced by two performance measures on three publicly available datasets. The findings underscore the potential of integrating SA methods with RSs to enhance personalization in MOOCs.

## Reproducability

The experiments can be reproduced using the `run_all_pca` function in `experiments.py`. This function will perform tuning for each survival model on both completion and dropout as the target. The `requirements.txt` file contains the package dependencies for these experiments.

#### Parameters:

- **`dataset`** *(required, options=['Canvas','X','KDD'])*:  
  A string for the dataset to perform experiments on. Loads the respective files and generates features

- **`split_count`** *(optional, default=3)*:  
  The number of hidden courses per user for the test set

- **`min_completed`** *(optional, default=1)*:  
  The minimum number of completed courses per user in the test set

- **`normalize_time`** *(optional, default=True)*:  
  Should time be normalized between 0 and 1 within each course

- **`tune_models`** *(optional, default=False)* 
  Should the time-to-event and baseline recommender models be re-tuned. If false, the values reported in the appendix of the table will be used for each dataset and method. If true, the survival models will be re-tuned using Bayesian hyperparameter optimization using the `hyperopt` package while the baseline recommender are tuned using the `skopt` package.

## Repository structure

```plaintext
mocc_cf_sa/
├── Base/                   # helper functions for baseline recommenders 
├── Utils/                  # util functions 
├── baselines/              # implementation of baseline recommenders
├── helpers/                # helper functions 
├── README.md               # Project README
├── canvas_preprocessed     # processed cavas dataset for experiments
├── experiments.py          # main python file to reproduce experiments
├── kddcup_preprocessed.csv # processed kddcup dataset for experiments
├── xuentengx_preprocessed  # processed xuentengx dataset for experiments
├── requirements.txt        # package versions
```





