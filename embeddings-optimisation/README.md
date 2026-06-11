# EDCA – An Evolutionary Data-Centric AutoML Framework for Efficient Pipelines

EDCA is a low-cost AutoML framework capable of creating simpler but efficient ML solutions.

## What is EDCA?

EDCA is a Python library for Automated Machine Learning (AutoML). It optimizes the entire ML pipeline. Given a classification dataset, EDCA starts by making an analysis of the features types and characteristics. This analysis serves to define the data transformations required for the data in question. Then, with the pipeline steps required, it starts the search for its bests estimators and models for each step of the pipeline. The search relies on a Genetic Algorithm. In the end, the user receives the best pipeline found ready to make predictions over unseen data.

![image info](docs/images/edca/edca-process-overview.png)

## Installation

### Conda

Install the conda environment from the yml file with all the dependencies.

    conda env create -f environment.yml

    conda activate edca

### pip

Install from the pip requirements (not recommended)

    conda create --name edca python=3.11.13

    conda activate edca 

    pip install -r requirements.txt

## Getting Started


Start by importing it

```python

from edca.evodata import DataCentricAutoML


To run the benchmark using Docker, change the code and configuration files, accordingly, and then run:

- for the benchmark

      docker compose -f docker-compose.benchmark.yml -p edca-benchmark up

## Repository Structure

- `*EDCA/analysis*`: scripts for making a statistical analysis of the benchmarks
- `*EDCA/benchmarks*`: source code for making the benchmarks
  
  - `*EDCA/benchmarks/configs*`: configuration files to use on the benchmarks
  - `*EDCA/benchmarks/src*`: source code for the benchmarks

- `*EDCA/data*`: contains information about the datasets used.
- `*EDCA/edca*`: contains EDCA implementation
- `*EDCA/ensemble*`: contains the code for ensemble experiments with clinical practice.

Note: Inside most directories there is a *README.md* detailing its content.


