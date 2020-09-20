# Recommendation of merchants to bank's customers via latent semantic search
## Model
Latent semantic model based on singular value decomposition (SVD)
## Metric
**Precision@topK**

For each person in training set we know merchants that he likes. For each person in test set we know only 5 merchants that he likes, and try to predict K (in our case, 3) other merchants he might also like. **Precision@topK** for this person is a fraction of correctly predicted merchants among all K. Precision@topK for the whole dataset is an average of these values.

**Score: 0.57** (loosely speaking, among 3 recommendations more than a half works well on average)

## To reproduce the result:
* Clone this repo;
* Execute `python3 training.py`
## Relevant literature
<a href="https://dl.acm.org/citation.cfm?id=1864721">Performance of recommender algorithms on top-n recommendation tasks</a>
