import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Result:
    query: str
    real_score: float
    predicted_score: float


def read_data(path: str) -> Dict[str, List[Result]]:
    # Collate by query.
    by_query = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            query, real_score, predicted_score = row
            by_query.setdefault(query, []).append(
                Result(
                    query=query,
                    real_score=float(real_score),
                    predicted_score=float(predicted_score),
                )
            )

    # Sort by predicted_score.
    for query in by_query.keys():
        by_query[query].sort(key=lambda r: r.predicted_score, reverse=True)
    return by_query


def calculate_dcg(results: List[Result]) -> float:
    dcg = 0
    for rank, result in enumerate(results, 1):
        dcg += result.real_score / math.log(rank + 1, 2)
    return dcg


def calculate_ncdg(data_by_query: Dict[str, List[Result]]) -> float:
    ndcgs = []
    for query, results in data_by_query.items():
        dcg = calculate_dcg(results)
        optimal = calculate_dcg(
            [r for r in sorted(results, key=lambda r: r.real_score, reverse=True)]
        )
        ndcgs.append(dcg / optimal)
    return statistics.mean(ndcgs)


# Unit testing of dcg/ncdg
def test_dcg():
    assert calculate_dcg([Result('apple', 1, 1)]) == 1

# test dcg with two results (in correct ranking, and incorrect ranking)
# NOTE: result list should always be in descending order w.r.t predicted_score (as defined in read_data)
def test_dcg2():
    assert calculate_dcg([Result('apple', 1, 1), Result('apple', 0, 0)]) == 1
    assert calculate_dcg([Result('apple', 0, 1), Result('apple', 1, 0)]) == 1 / math.log(3,2)
    
# test ncdg with one query
def test_ncdg():
    assert calculate_ncdg({'apple': [Result('apple',1,1)]}) == 1 # base case
    assert calculate_ncdg({'apple': [Result('apple', 1, 1), Result('apple', 0, 0)]}) == 1  # dcg is 1, and optimal is 1
    assert calculate_ncdg({'apple': [Result('apple', 0, 1), Result('apple', 1, 0)]}) == 1 / math.log(3,2) # dcg is 1/math.log(3,2), and optimal is 1

# test two queries
def test_ncdg2():
    assert calculate_ncdg({'apple': [Result('apple',1,1), Result('apple',0,0)],
                           'pear': [Result('pear',1,1), Result('pear',0,0)]}) == 1 # both dcgs are 1, and optimals are 1
    assert calculate_ncdg({'apple': [Result('apple',1,1), Result('apple',0,0)],
                           'pear': [Result('pear',0,1), Result('pear',1,0)]}) == (1 + 1 / math.log(3,2))/2 # dcgs (1,1/math.log(3,2)), optimals(1,1), mean of dcgs./optimals


def test_all():
    print("Running tests")
    # TODO Add your functions to this list!
    for func in [test_dcg, test_dcg2, test_ncdg, test_ncdg2]:
        try:
            func()
        except AssertionError:
            status = "😫"
        else:
            status = "🎉"
        print(f"{status}\t{func.__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=1, help="Predictions in .TSV format")
    args = parser.parse_args()

    path = args.path[0]
    if not os.path.exists(path):
        parser.error(f"Invalid path {path}")

    data_by_query = read_data(path)
    mean_ndcg = calculate_ncdg(data_by_query)
    print(mean_ndcg)

    test_all()


if __name__ == "__main__":
    main()
