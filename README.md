# RecSys-Project

The aim of this project is to explore if the introduction of a probabilistic constraint  during  the  learning  phase  of  the  SLIM  and  Matrix  Factorization algorithms can lead to an improvement in algorithm performances. 

Our contribution has been to extend the Cython code available in order to introduce the constraint during the learning phase of the algorithm. We dealt with the modification of derivatives used by the gradient descent mechanism in order to make the resulting weigth matrix respect our constrained optimization conditions. 

We tested the constrained optimization algorithms on three classical Recommender Systems Dataset: BookCrossing, Epinions and Movielens 1M/10M considering various possibile data selection techniques.

Interesting conclusions we derived from our work is that our version of the SLIM RMSE algorithm is able to outperform the classical SLIM RMSE implementation if we consider a dense dataset, in particular we obtained +30% in Mean Average Precision. 
