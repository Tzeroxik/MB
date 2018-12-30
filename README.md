# 2nd Project for the curricular unit "Modelação Bayesiana"

## Implemented functions

correlation_matrix(mat) 
* **mat**: the original dataset matrix which is formated as samples in rows and variables in columns.
* Returns the respective correlation matrix. 

### covariance_matrix(**corrmat**)

 * **corrmat**: the correlated matrix resulting from calling the function **corellation_matrix** on the dataset matrix. 
* Returns the normalized covariance matrix.

pca(mat) 
* **mat**: the original dataset matrix which is formated as samples in rows and variables in columns.

* Returns an SVD object (from Julia's LinearAlgebra package) stemming from the single value decomposition of the correlation matrix of **mat**.