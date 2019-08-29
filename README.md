# Image Search Ranking

## Product context

Canva has over 50 million images for a user to choose from when creating a new design. Image search is important to Canva so that users can find the right image conveniently at any time during the design process. We have more than 10 billion user query logs to learn from in over 100 languages.

We want to predict a ranked list of results for each query so that users always find great stock imagery for their designs.

## Data

We've supplied a small sample of anonymized data from our image search in the file `data.csv.gz`.
```
media_id,query,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,clicks
97fd5eb5e8,stars,True,,31,False,False,False,False,True,False,1413,0,0,0.0,0.0,True,0,0,0.0,0.0,143.62915,72.13346,0,0,0.0,0.0,0.0,0.0,4.0,-1.0,16,3,22,0,0
97fd5eb5e8,star,True,,15,False,False,True,False,False,False,975,0,0,0.0,0.0,True,0,0,0.0,0.0,162.12103000000002,68.08842,0,0,0.0,0.0,0.0,0.0,4.0,-1.0,16,3,22,0,0
c94faef102,beach,True,,30,False,False,True,False,True,False,15548,0,2,0.0,0.0,True,0,2,0.0,0.0,5.5240545,11.842522,0,0,0.0,0.0,0.0,0.0,4.0,-1.0,5,3,30,0,2
```
To decompress the data, in a terminal run
```
>> gunzip data.csv.gz
```

The columns are:
* `media_id` a reference to the image that was returned by the search (each image is given a unique ID),
* `query` the query that the user typed (only English),
* `f1` to `f33` features extracted from search logs, all with integer, float or boolean values,
* `clicks` the number of times this particular `media_id` was clicked for this query.

## Task

One way to solve ranking problems is via a **simple linear regression**. Specifically, we want to learn a single model that predicts a _score_ between `0` and `1` that approximates how much a user would click on a `media_id` result for a `query`.

The following subtasks will help you come up with this solution.

### 1. Explore the data.

This is an analysis task. You will need to:
* Identify a _score_. Consider how we need to preprocess this to be suitable for learning.
* Identify _features_ that we will use to train our model.

### 2. Data preprocessing, model training and prediction

This is a programming task. Write a short program that:
* Applies any necessary preprocessing you identified in step 1 to the data.
* Splits the data into a training and validation set.
* Trains a **linear regression** model to predict a score between `0` and `1` (inclusive of both) for a data row.
* Run that model on the validation data and print output to standard out.

For example:
```bash
python3 solution.py /path/to/data.csv > predictions.tsv
```

The expected output format tab-separated text, one line for each instance in your validation data.
```
query\treal_score\tpredicted_score
```

For example:
```
star	0.0	0.1553365446460433
beach	1.0	0.32203332713335575
beach	0.0	0.5234502323427924
education	1.0	0.843759720543486
education	1.0	0.9702476627505279
```

You should be able to use the included `evaluate_predictions.py` script to print the metric result for your output. For example:
```bash
python3 evaluate_predictions.py output.tsv
```

Prints (for a sample solution)
```
0.9853599280765144
Running tests
🎉	test_dcg
```

### 3. Understanding the metric

This is an analysis and programming task. You will need to:
* Read the `evaluate_predictions.py` script to understand how we implement Normalised Discounted Cumulative Gain (NDCG). This is a common relevance metric that we're reusing for this task.
* Write three more unittest-style functions following the `test_dcg()` example in `evaluate_predictions.py`.

### 4. Discussion

This is a short writing task. You will need to write maximum 300 words where you:
* Suggest a few ways in which you might extend or improve the work.
* Discuss how you could test that your solution works, both during development and as a deployed system.

## Resources

* Please use the data provided in `data.csv`.
* Spend no more than 6 hours on the task. 
* We care more about your approach than your model's performance (accuracy or inference speed), so don't feel you need to get a world-beating score.
* Choose a language and libraries you feel comfortable in. Internally, we use Python and Scala, but please contact us if you wish to use another language and we can try to find a reviewer. Python with `scikit-learn` and `pandas` is recommended for these tasks.
* The code does not need to be production ready, so as long as the approach and code structure is clear.

## Submission

Please submit:
* The source code for your program for subtask 2. Please include any instructions on dependencies or compilation as a comment at the top of the file.
* Your edited copy of `evaluate_predictions.py` from subtask 3 that includes the extra unittest functions.
* Your discussion from subtask 4.

