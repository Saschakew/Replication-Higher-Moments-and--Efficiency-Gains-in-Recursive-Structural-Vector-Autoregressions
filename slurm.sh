#!/bin/sh -l

#SBATCH --time=28-00:00:00
#SBATCH --mem=200G
#SBATCH --nodes=1 --exclusive
#SBATCH --partition=ext_vwl_prio,ultralong,long,med,short
#SBATCH --job-name="MC_Efficiency_Sim1"
#SBATCH --output=MC_Efficiency_Sim1-job%j.out
#SBATCH --error=MC_Efficiency_Sim1-job%j.err

module purge
module add python/3.7.4
cd "/work/smsakewe/Replication-Higher-Moments-and--Efficiency-Gains-in-Recursive-Structural-Vector-Autoregressions"
pip install statsmodels
python3 MC_Efficiency_Sim1.py
