# Budget Sharing for Multi-Analyst Differential Privacy

The arxiv version of this paper [1] is at https://arxiv.org/abs/2011.01192.

The codes in the paper are in the **master** branch. Please only use the **master** branch. Other branches are used for developments beyond this paper.

All codes are in Python 3. (We use 3.6.9 for experiments and 3.7.7 Jupyter Notebooks.)

The core codes are in the **src** directory, including the **hdmm** package from the HDMM paper [2], which is implemented based on the Ektelo package [3]. All the codes are implemented based on the hdmm package.

The codes for calculating strategies and errors are in **src/fair_experiments.py**.

The codes for water-filling algorithm is in **src/bucket_filling.py** and the function we use in our paper is **query_sd**.

For the variables, *n* is the size of data, *k* is the number of analysts, *eps* is the total privacy budget, *rep* is the number of iterations for optimizing the strategies. *tol* is the tolerance in the Water-filling Mechanism.

*modes* is a list of names of the mechanisms used in the experiment. They can be chosen from 'ind' (Independent), 'iden' (Identity), 'uni' (Utilitarian), 'fsum' (Weighted Utilitarian), 'buc_qsd' (Water-filling).

*Ws* is the workload set, *Wr* is the workload set to be evaluated (may be same as or different from *Ws*). *As* and *Ar* are the strategy matrices for Ws and Wr, respectively. *Ws*, *Wr*, *As*, and *Ar* are all 1-d numpy arrays of Ektelo matrices.

*qr*, *x*, *ans*, *sample* are only used for data-dependent queries. *qr* is the name of queries and can be chosen from 'Linear', 'Median', 'Mode', 'Mean', 'Per_x' (x percentile). None and any other value is the same as 'Linear'. *x* is the data. *ans* is the true answer. *sample* is the number of iterations for calculating the expected errors.

We use *eps*=1.0, *rep*=10, *sample*=10000, *tol*=1e-3 as the default settings in our experiments. Note that the values may be different from the default values in the Python scripts.

The Python scripts and the Slurm job scripts for the experiments shown in the paper are as the follows:

Practical Settings (Section 5.3): **practical_interference/race_vary.py**, **practical_interference/array_workload.sh**

Marginal Workloads (Section 5.4): **practical_interference/oneway_marginal_vary.py**, **practical_interference/array_marginal.sh**

Tolerance for Water-filling Mechanism (Section 5.5): **practical_tol/oneway_marginal_vary.py**, **practical_tol/array_marginal.sh**

Data-dependent Non-linear Queries (Section 5.6): **data_interference/mm_per_vary.py**, **data_interference/array_workload.sh**

The Slurm job scripts show the exact experiment settings. The experiments ran on the Duke Computer Science Cluster.

### Reference
[1] Pujol, D., Wu, Y., Fain, B., & Machanavajjhala, A. (2020). Budget Sharing for Multi-Analyst Differential Privacy. arXiv preprint arXiv:2011.01192.

[2] McKenna, R., Miklau, G., Hay, M., & Machanavajjhala, A. (2018). Optimizing error of high-dimensional statistical queries under differential privacy. Proceedings of the VLDB Endowment, 11(10), 1206-1219.

[3] Zhang, D., McKenna, R., Kotsogiannis, I., Hay, M., Machanavajjhala, A., & Miklau, G. (2018, May). Ektelo: A framework for defining differentially-private computations. In Proceedings of the 2018 International Conference on Management of Data (pp. 115-130).
