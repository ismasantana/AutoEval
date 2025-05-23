from sklearn.metrics import accuracy_score

def compare_real_estimated_cross_accuracy(train_X, train_y, test_X, test_y, evaluator):
    evaluator.fit(train_X, train_y)
    y_pred_real = evaluator.estimator.predict(test_X)
    real_acc = accuracy_score(test_y, y_pred_real)

    est_result = evaluator.estimate(test_X)
    estimated_acc = list(est_result.values())[0] if est_result else 0.0

    evaluator.fit(test_X, test_y)
    y_pred_cross = evaluator.estimator.predict(train_X)
    cross_acc = accuracy_score(train_y, y_pred_cross)
    if evaluator.verbose:
        print(f"Real Accuracy:     {real_acc:.4f}")
        print(f"Estimated Accuracy:{estimated_acc:.4f}")
        print(f"Cross Accuracy:    {cross_acc:.4f}")
        print("-" * 50)

    return {
        "real_acc": real_acc,
        "estimated_acc": estimated_acc,
        "cross_acc": cross_acc
    }
