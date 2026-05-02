# Image-Only Experiments for Predicting Mode of Delivery After IOL

This folder presents the code used for experiments with only image data. 

## Run Image-Only Experiments 

Use `train_validate_multi_seeds.py` to train and test all the Vision Models used as image encoders in the multimodal approach. To aggregate the results from each model use `aggregate_metrics.py`. For each image encoder, you must clone the corresponding model repository and use its associated environment configuration file.