# Fast STAN for Large Scale Session-Based Recommendation
Fast STAN is a multi-processing variant of the [Sequence and Time Aware Neighborhood (STAN)](https://dl.acm.org/doi/abs/10.1145/3331184.3331322) algorithm [(code)](https://github.com/rn5l/session-rec/blob/master/algorithms/knn/stan.py), an excellent $k$-NN-based algorithm for session-based recommendation tasks. It modified some data storage and result inference mechanisms of STAN by replacing incremental training codes with paralleled modules, sacrificing its online learning ability to achieve extremely fast training speed.

## How to Run It
``` bash
python ./run.py --train "train_*_20.parquet" --test "test_*_5.parquet" --ref "ref.pkl" --save "results.csv" -j 30
```
You can find the demo dataset at [kaggle](https://www.kaggle.com/competitions/otto-recommender-system/overview).

## Fast STAN v.s. STAN
### Running Time
How much faster Fast STAN is than STAN depends on the number of processes. Theoritically, one more process can double the speed. But it is still not the total improvement of Fast STAN. The results in [this notebook](./comparison.ipynb) shows the single-processing parts of Fast STAN are also faster than the original STAN by **5 times**.

### Recall / Precision / Accuracy
I tested Fast STAN on a [kaggle competition](https://www.kaggle.com/competitions/otto-recommender-system/overview), and the recall score is 0.512 (ranked 1000+). Here, I must point out that the STAN performed pooly because this competition was a multi-objective session-based recommendation task, but the STAN was design for single-objective tasks. It underestimated STAN's performance.

I did not compare their performance on general session-based recommendation tasks. I will be really appreciated it if you could feedback on their performance on benchmark datasets.

## Improvements
Despite the original STAN algorithm being closer to the realistic task requirements that recommended items are inferred in an incremental training manner (online learning), experimental analysis shows this inference process takes much more time than paralleled training (offline learning). Besides, it has been a few years since the release of STAN's source code, where some data operations are neither efficient nor concise.

Therefore, I accelerate STAN's recommendation process by:
1. Multi-processing inference. I parallelize the inference process so as to accelerate the process of learning sessions' relationship within a large dataset. Some non-process-safe designs (e.g., the incremental test data cache in inference) are modified in order to enable inference of the next item in a multi-process manner.
2. Third-party data operation module. I replace some build-in data operations with `pandas` because making large-scale inferences by `pandas` is significantly faster and more concise than the build-in methods.

In addition, the original STAN requires the input data ordered by time stamp. Some data will miss if the input data is not ordered by session id. It takes an additional step to sort the input data to support out-of-order input and make the model more robust. Although it sacrifices training efficiency, it is acceptable compared to the cost savings of inference, let alone the flexibility and concisen