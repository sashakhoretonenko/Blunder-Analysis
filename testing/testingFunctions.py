import torch
import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
#-----------------------------------------------------------------------
def evaluate_per_piece_accuracy(model, test_loader, player, device):
    model.eval() 
    class_correct = [0] * 6
    class_total = [0] * 6
    all_labels = []
    all_predictions = []

    classes = ['P', 'N', 'B', 'R', 'Q', 'K']



    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            # Collect all labels and predictions for metrics
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # Calculate per-class accuracy
            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if predicted[i] == y[i]:
                    class_correct[label] += 1

    # Calculate metrics for each piece
    correct_counts = []
    incorrect_counts = []
    accuracies = []
    recalls = []
    f1_scores = []

    for i in range(6):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            recall = recall_score(all_labels, all_predictions, labels=[i], average='macro')
            f1 = f1_score(all_labels, all_predictions, labels=[i], average='macro')
            correct_counts.append(class_correct[i])
            incorrect_counts.append(class_total[i] - class_correct[i])
            accuracies.append(accuracy)
            recalls.append(recall)
            f1_scores.append(f1)
            print(f"{classes[i]}: Accuracy: {accuracy:.4f}%")
        else:
            print(f"{classes[i]}: No samples available.")
            correct_counts.append(0)
            incorrect_counts.append(0)
            accuracies.append(0)
            recalls.append(0)
            f1_scores.append(0)

    # Save this as an excel file
    df = pd.DataFrame({
        'Piece Type': classes,
        'Correct Predictions': correct_counts,
        'Incorrect Predictions': incorrect_counts
    })

    df.to_excel(f'{player}_per_piece_accuracy.xlsx', index=False)

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(classes))

    plt.bar(index, correct_counts, bar_width, label='Correct')
    plt.bar([i + bar_width for i in index], incorrect_counts, bar_width, label='Incorrect')

    plt.xlabel('Piece Type')
    plt.ylabel('Count')
    plt.title('Correct vs Incorrect Predictions by Piece Type')
    plt.xticks([i + bar_width / 2 for i in index], classes)
    plt.legend()
    plt.show()

    df = pd.DataFrame({
        'Piece Type': classes,
        'Accuracy (%)': accuracies,
        'Recall (%)': recalls,
        'F1 Score (%)': f1_scores
    })

    # Calculate weighted metrics
    weighted_accuracy = accuracy_score(all_labels, all_predictions) * 100
    weighted_recall = recall_score(all_labels, all_predictions, average='weighted') * 100
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted') * 100
    df.loc['Weighted'] = ['Weighted', weighted_accuracy, weighted_recall, weighted_f1]

    print("\nOverall Accuracy Metrics:")
    print(df)

    # Save df to excel
    df.to_excel(f'{player}_overall_accuracy_metrics.xlsx', index=False)

    return df