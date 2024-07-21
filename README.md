# Fair-Link-Prediction
Project for [Data Mining, Basic Course (ID2211)](https://www.kth.se/student/kurser/kurs/ID2211?l=en) at KTH.

In the project we explore if MovieLens-100K data has biases wrt to gender, by trainining and evaluating different Classical and Neural Link Prediction methods.

Methods used for Link-Prediction:
- GNNs
- Network based methods

## Fairness De-biasing Methods
Three post-processing methods were used for de-biasing the rating differences between male and female movie watchers.

1. **Distribution Weights:** Weights of rate of movie-ratings (edge-rate between male & female users) for re-weighing the pdf of softmax predictions.
2. **Naive Optimized Weights:** Similar but weights are found by reducing the difference between SP and EO in a naive grid-search like fashion.
3. **Linear Optimized Weights(from scipy): **A linear cost-fn that finds weights by balancing SP,EO and F1.


| GNN                                 | MAP        | F1         | AUC         | Pr         | Rec        | SP(Overall)  | EO(Overall) |
| ----------------------------------- | ---------- | :--------- | :---------- | ---------- | ---------- | ------------ | ----------- |
| GCN-3L + Dist. Weights              | 0.2849     | 0.4593     | 0.8590      | 0.31       | 0.8181     | **0.00945**  | 0.001238    |
| GCN-3L +  Naive Optimized Weighting | 0.5232     | 0.6769     | **0.8862**  | 0.60       | 0.777      | 0.0174       | 0.0114      |
| <mark>GCN-3L + Linear Opt. Weights</mark>    | **0.6736** | **0.74**   | 0.8849      | **0.8033** | **0.6859** | 0.0270       | **0.001**       |
| GAT-5L +Dist. Weights               | 0.12       | 0.2151     | 0.7911      | 0.1206     | **0.9926** | **0.000626** | 0.033827    |
| GAT-5L + Naive Optimized Weighting  | 0.5289     | 0.6873     | 0.8201      | 0.5933     | 0.8168     | 0.0126       | 0.0128      |
| <mark>GAT-5L + Linear Opt. Weights</mark>    | **0.6349** | **0.7643** | **0.87921** | **0.7051** | **0.8334** | 0.0219       | 0.0034      |



