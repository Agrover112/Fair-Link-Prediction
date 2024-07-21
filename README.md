# Fair-Link-Prediction
Project for [Data Mining, Basic Course (ID2211)](https://www.kth.se/student/kurser/kurs/ID2211?l=en) at KTH.

In the project we explore if MovieLens-100K data has biases wrt to gender, by trainining and evaluating different Classical and Neural Link Prediction methods.

Methods used for Link-Prediction:
- GNNs
- Network based methods

### Performance metrics for different feature ablations

SAGEConv and other models were tested with features which would balance accuracy and fairness metric performance.

| **Feature Ablation**            | **AUC**   | **Accuracy** | **F1**   | **SP**     | **EO**     |
|---------------------------------|-----------|--------------|----------|------------|------------|
| Age + Occ + Gender, Movie       | **0.8643**| **0.7998**   | 0.6809   | 0.036829   | 0.047949   |
| <mark>Age + Occ, Movie</mark>                | 0.8635    | 0.7971       | **0.6837**| 0.026296   | 0.030179   |
| Age, Movie                      | 0.8061    | 0.7310       | 0.4290   | **0.020913**| **0.021649**|


### Performance measures for the different models tested

| **Model**                 |     | **AUC**   | **Accuracy** | **F1-score** | **Precision** | **Recall** | **SP**     | **EO**     |
|---------------------------|-----|-----------|--------------|--------------|---------------|------------|------------|------------|
| **Bi-common-neighbors**   |     | 0.8584    | 0.7424       | 0.7001       | 0.5888        | 0.8632     |            |            |
| **Bi-Adamic-Adar**        |     | 0.8599    | 0.7436       | 0.7017       | **0.5907**    | 0.8642     |            |            |
| **Bi-Jaccard**            |     | **0.8695**| **0.7448**   | **0.7027**   | 0.5905        | **0.8674** |            |            |
| **Preferential Attachment** |   | 0.8450    | 0.7273       | 0.6781       | 0.5653        | 0.8472     |            |            |
| **Architecture**          | **Layers** | **AUC** | **Accuracy** | **F1-score** | **Precision** | **Recall** | **SP**     | **EO**     |
| **GATConv**               | 3   | 0.8013    | 0.7439       | 0.5956       | 0.5656        | 0.6288     | 0.007911   | **0.007694**|
|                           | 4   | 0.8306    | 0.7609       | 0.6587       | 0.6923        | 0.6283     | 0.009295   | 0.008364   |
|                           | 5   | **0.9215**| **0.8504**   | **0.7801**   | **0.7963**    | 0.7647     | 0.024848   | 0.018533   |
|                           | 6   | 0.7706    | 0.6751       | 0.0635       | 0.0331        | **0.8097** | **0.004536**| 0.013637   |
| **GINConv**               | 3   | 0.7118    | 0.7025       | 0.5692       | 0.5895        | 0.5502     | 0.026767   | 0.001744   |
|                           | 4   | **0.8606**| **0.8041**   | **0.7365**   | 0.8214        | **0.6675** | 0.016521   | 0.006719   |
|                           | 5   | 0.7960    | 0.5924       | 0.5766       | **0.8324**    | 0.4410     | **0.004849**| **0.000996**|
|                           | 6   | 0.5927    | 0.6921       | 0.5363       | 0.5342        | 0.5384     | 0.012549   | 0.004277   |
| **GraphConv**             | 3   | 0.8822    | 0.8142       | 0.7047       | 0.6650        | 0.7494     | 0.018200   | 0.006651   |
|                           | 4   | 0.8853    | 0.8144       | 0.6977       | 0.6426        | **0.7632** | 0.021984   | 0.023273   |
|                           | 5   | 0.8822    | 0.8130       | 0.7288       | 0.7538        | 0.7053     | **0.011697**| 0.005238   |
|                           | 6   | **0.8914**| **0.8187**   | **0.7355**   | **0.7564**    | 0.7158     | 0.030041   | **0.000958**|
| **SAGEConv**              | 3   | 0.8767    | 0.8100       | 0.6994       | 0.6632        | 0.7399     | **0.009929**| 0.016642   |
|                           | 4   | 0.8782    | **0.8116**   | **0.7193**   | 0.7244        | 0.7143     | 0.028708   | 0.008952   |
|                           | 5   | **0.8804**| 0.8101       | 0.6992       | 0.6621        | **0.7406** | 0.016706   | 0.003372   |
|                           | 6   | 0.8747    | 0.8009       | 0.7135       | **0.7439**    | 0.6855     | 0.017205   | **0.001091**|
| **TransformerConv**       | 3   | 0.8843    | 0.8130       | 0.7243       | **0.7372**    | 0.7119     | **0.010989**| 0.012458   |
|                           | 4   | 0.8855    | 0.8149       | 0.7249       | 0.7316        | 0.7183     | 0.015846   | 0.011875   |
|                           | 5   | 0.8780    | 0.8056       | 0.7044       | 0.6948        | 0.7143     | 0.023165   | **0.010082**|
|                           | 6   | **0.8901**|**0.8193**    | **0.7279**   | 0.7249        | **0.7309** | 0.026621   | 0.012776   |


## Fairness De-biasing Methods
Three post-processing methods were used for de-biasing the rating differences between male and female movie watchers. These methods were only applied to the chosen models as they showcased a good balance over both performance and fairness metric before de-biasing.

1. **Distribution Weights:** Weights of rate of movie-ratings (edge-rate between male & female users) for re-weighing the pdf of softmax predictions.
2. **Naive Optimized Weights:** Similar but weights are found by reducing the difference between SP and EO in a naive grid-search like fashion.
3. **Linear Optimized Weights(from scipy):** A linear cost-fn that finds weights by balancing SP,EO and F1.


| GNN                                 | MAP        | F1         | AUC         | Pr         | Rec        | SP(Overall)  | EO(Overall) |
| ----------------------------------- | ---------- | :--------- | :---------- | ---------- | ---------- | ------------ | ----------- |
| GCN-3L + Dist. Weights              | 0.2849     | 0.4593     | 0.8590      | 0.31       | 0.8181     | **0.00945**  | 0.001238    |
| GCN-3L +  Naive Optimized Weighting | 0.5232     | 0.6769     | **0.8862**  | 0.60       | 0.777      | 0.0174       | 0.0114      |
| <mark>GCN-3L + Linear Opt. Weights</mark>    | **0.6736** | **0.74**   | 0.8849      | **0.8033** | **0.6859** | 0.0270       | **0.001**       |
| GAT-5L +Dist. Weights               | 0.12       | 0.2151     | 0.7911      | 0.1206     | **0.9926** | **0.000626** | 0.033827    |
| GAT-5L + Naive Optimized Weighting  | 0.5289     | 0.6873     | 0.8201      | 0.5933     | 0.8168     | 0.0126       | 0.0128      |
| <mark>GAT-5L + Linear Opt. Weights</mark>    | **0.6349** | **0.7643** | **0.87921** | **0.7051** | **0.8334** | 0.0219       | 0.0034      |

Graph Showcases how probability distribution of male and female ratings are shifted in GATConv-5L. One can notice that re-weighting via linear optimization shifts the density of female probability ratings , almost like a lifter in Speech processing and reduces the width of probability distribution of male rating distributions. For GraphConv-3L we observe something similar with density but given better parity scores we notice that shifting the pdf of female group towards right and making the width of male pdf smaller results in somewhat better parity between the two groups.
The bi-modal nature of the predictions is due to the experimental choice to split ratings into 0/1 labels.

![871315be-48c2-41f0-bd44-e3c2854a2aa1 height="250"](https://github.com/user-attachments/assets/8575c99f-25ee-4437-9767-51eec8237a97)

![590bae3b-6c42-4d38-94e9-592b374c6474 height="250"](https://github.com/user-attachments/assets/c0e8bc75-5f56-4a86-9df0-8a50a193ed70)

# Conclusion
We notice that although there might be slight loss in performance metrics, there is quite an improvement over the Fairness metrics. Simple post-processing methods seem quite useful however, further fine-tuning, other methods such as changes to latent distribution might help. Model parameters,layers doesn't seem to show any correlation with the fairness metrics, however aggregation schemes of different GNNs could in-theory along with our network be a cause of influence.

Authors: refer report.


