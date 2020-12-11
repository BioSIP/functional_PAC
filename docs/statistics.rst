
Statistics
===================================
Traditional, parametric approaches to statistical testing have revealed unuseful in mass univariate problems due to the Multiple Comparison Problem (MCP). Massive univariate testing yields unnaceptable levels of false significant regions, so a procedure for controlling the problem must be used. Family-wise error rate (FWER) such as the Bonferroni correction arise as overconservative thresholds, proving useless for this purpose. On the other hand, nonparametric approaches have lately gained ground, especially with the emergence of massive computational power.  


Permutation testing approaches have flourished among this plethora of testing methodologies, providing controlled estimation of significance levels at any given :math:`\alpha` -the statistical significance threshold-. The procedure that underlies this category is the approximation of the nonparametric distribution of the data via the Monte Carlo (MC) method. The simplest approach to permutation testing would work as follows. 

Given a dataset comprising pairs :math:`(\mathbf{x}_i, y_i), i\in \mathbb{N}`, respectively the data and labels of the :math:`i^{th}` sample, let us call :math:`X` the matrix containing all data from the :math:`k` samples :math:`X = \{\mathbf{x}_0, \mathbf{x}_1, \dots \mathbf{x}_k\}`, and :math:`\mathbf{y}` the vector containing :math:`k` labels for each sample. First, compute a statistical on the data using the real :math:`\mathbf{y}` vector, and save it as the reference :math:`z`. Afterwards, generate a surrogate label vector, :math:`\hat{\mathbf{y}}`, and re-estimate the statistical using :math:`\hat{\mathbf{y}}`, which which yields :math:`\hat{z}`. Repeat this surrogate procedure :math:`N` times, 

.. automodule:: statistics

Permutation Testing
-----------------------------------
.. autoclass:: PermutationTesting
	:members: permutation_testing, fit
	
Cluster-Based Permutation Testing
-----------------------------------

.. autoclass:: ClusterBasedPermutationTesting
	:members: mwu_vectorized, matrix_testing, permutation_testing, supra_threshold, threshold, fit
