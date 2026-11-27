import numpy as np


def find_top_k(query, database, k=5):
    sim = np.dot(database, query)
    sim = np.argsort(sim)[::-1][:k]
    return sim


def recall_at_k(predictions, positives, k):
    results = []
    for i in range(len(predictions)):
        results.append(np.any(positives[i][predictions[i][:k]]))
    return np.mean(results)
