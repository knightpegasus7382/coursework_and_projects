************** CS6046: MULTI-ARMED BANDITS PROJECT ***************

TEAM MEMBERS: Dhruv Gopalakrishnan (AE17B004), Akash Reddy (EE17B001), Vishwajit Prakash Hedge (ME17B039)

******************************************************************

Some steps to run our project properly:

1. Paper 1 algorithms run directly by running p1_if1_if2.py

2. Paper 2 algorithms runs using all 3 code files. 

	- Use the last segment of code in p1_if1_if2.py to obtain a dict called "Regret_dict_" using Pickle. Store the "Regret_dict_.pickle" file - the regret values for IF2 for 50 runs. This information is used in the comparison of algorithms in paper 2. After loading the pickle file into a program as a dict, we can use dict[run number]['link function type'] to obtain the per-timestep regret list. For e.g., "regret_dict[0]['linear']". Use np.cumsum to plot cumulative regret.

	- In p2_doubler_multi.ipynb and p2_sparring_plots.ipynb, uncomment and comment lines of code where required to select among 1good, 2good, 3good, arith, geom. And among linear, natural, logit link functions.

	- Run p2_doubler_multi.ipynb for Doubler and MultiSBM algorithms by uncommenting the required expected values of arms, and link function. This code is used to write the regret values for Doubler and MultiSBM into csv files whose name can be changed depending on expected value and link function setting. For e.g. "multisbm_linear_3.csv" or "doubler_linear_arith.csv". All files have been included in this directory.

	- Load these .csv files and the .pickle file into arrays and dict respectively on p2_sparring_plots.ipynb (already done). This file runs Sparring algorithm after uncommenting the chosen expected arm values, and changing the function call within bt(ut, vt) to the desired link function. It also uses the loaded regret values for Doubler, MultiSBM, and IF2 in order to produce the final required regret plots.
