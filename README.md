# Replication Package: Higher Moments and Efficiency Gains in Recursive Structural Vector Autoregressions

This repository contains the code to replicate the results of the paper "Higher Moments and Efficiency Gains in Recursive Structural Vector Autoregressions".

## Citation

Keweloh, S. A., & Wang, S. (2025). Higher Moments and Efficiency Gains in Recursive Structural Vector Autoregressions. *Oxford Bulletin of Economics and Statistics*. https://doi.org/10.1111/obes.70008

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to have Anaconda or Miniconda installed on your system. You can download it from the [official Anaconda website](https://www.anaconda.com/products/distribution).

### Installing

1. Clone the repository to your local machine:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd Replication-Higher-Moments-and--Efficiency-Gains-in-Recursive-Structural-Vector-Autoregressions
   ```
3. Create the conda environment from the `environment.yml` file. This will install all the necessary dependencies.
   ```sh
   conda env create -f environment.yml
   ```
4. Activate the newly created environment:
   ```sh
   conda activate svarpy
   ```

## Running the Simulation and Analysis

The analysis is split into three main scripts:

1.  **Run the Monte Carlo Simulation**

    To replicate the Monte Carlo simulation, run `MC_Efficiency_Sim1.py`. This script creates a folder `MCResults/MC_Efficiency_Sim1` where all the results of the individual Monte Carlo runs are saved.

    ```sh
    python MC_Efficiency_Sim1.py
    ```

2.  **Collect Simulation Data**

    After the simulations are complete, the `aMC_CollectData.py` script collects all the individual result files from the `MCResults` directory and consolidates them into a single file for easier analysis.

    ```sh
    python aMC_CollectData.py
    ```

3.  **Evaluate Results and Generate Plots**

    Finally, the `aMC_Evaluation_Efficiency.py` script evaluates the collected data and creates the two figures `MCMeanPlot.pdf` and `MCRejectionPlot.pdf` which are presented in the paper.

    ```sh
    python aMC_Evaluation_Efficiency.py
    ```

## Author

*   **Sascha A. Keweloh**
