# Image Search Ranking

## Context

Image search is important to companies so that users can find the right image conveniently at any time during the design process. We want to predict a ranked list of results for each query so that users always find great stock imagery for their designs.

## Data

Over 110,000 rows of anonymized data from images searched in the file `image_summary.csv`.
```
media_id,query,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,clicks
```
The columns are:
* `media_id` a reference to the image that was returned by the search (each image is given a unique ID),
* `query` the query that the user typed (only English),
* `f1` to `f33` features extracted from search logs, all with integer, float or boolean values,
* `clicks` the number of times this particular `media_id` was clicked for this query.

### Model Preparation, Training and Validation
* Utilize a simple linear regression model to assign a score to each image
* We can map the `clicks` to a  _score_ by min-max scaling each clicks value with respect to the minimum and maximum clicks for each query. For example, if image 1 for a query 'stars' had 5 clicks, and the minimum for 'stars' was 0, and maximum was '10', then the score would be 0.5.
* We need to identify _features_ that we will use to train our model. In order to do this, we can begin by removing redundant features such as `media_id`, and ones which have 0 variance across the dataset. We then proceed to use PCA analysis to remove collinear features. The idea here is to produce a correlation matrix and its eigenvalues/eigenvectors. Extremely low value eigenvectors point to collinearties in the model, which effectively mean some features are linear functions of other features, and therefore redundant.
* Finally we can utilize a standard sci-kit learn linear regression model to train the data. Due to the large number of rows available, we keep aside 10% of the data for final validation. For training, K-fold validation is implemented with 10 folds.

### Evaluating Predictions
* For image search ranking, we aren't so much interested in the accuracy of our linear model, but rather if our scores effectively rank the images according to how much they were clicked. 
* We utilize the discounted cumulative gain, and normalized discounted cumulative gain as measures of our ranking. For more information on these metrics, see https://machinelearningmedium.com/2017/07/24/discounted-cumulative-gain/. 

### Running the program
The program can be run using the following command line prompts:

```bash
python3 solution.py /path/to/data.csv > predictions.tsv
```

The output format is tab-separated text, one line for each instance in the validation data.
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

The `evaluate_predictions.py` script can then be run on the predictions file to produce the normalized discounted cumulative gain:
```bash
python3 evaluate_predictions.py output.tsv
```

Prints (for a sample solution)
```
0.9853599280765144
Running tests
🎉	test_dcg
🎉	test_dcg2
🎉	test_ncdg
🎉	test_ncdg2
```

