def Classifiers(names, classifiers, x_train, x_test, y_train, y_test):
    import matplotlib.pyplot as plt

    scores = []
    best_score = 0
    best_model = None
    best_name = ""

    # Clear or create the results file
    with open('ClassificationResults.txt', 'w') as f:
        f.write("Classifier Performance Comparison\n\n")

    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)
        scores.append(test_score)

        # Save to results file
        with open('ClassificationResults.txt', 'a') as f:
            f.write(f"{name}\n")
            f.write(f"  Train Accuracy: {train_score:.4f}\n")
            f.write(f"  Test Accuracy:  {test_score:.4f}\n\n")

        # Track best
        if test_score > best_score:
            best_score = test_score
            best_model = clf
            best_name = name

    # Plot classifier performance
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, scores, color='skyblue')
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{score:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=45)
    plt.ylabel('Test Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Print best result
    print(f"\nBest Classifier: {best_name} with Accuracy: {best_score:.4f}")
