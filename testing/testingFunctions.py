import torch
import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, brier_score_loss
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F



def evaluate_per_piece_metrics(model, test_loader, player, device, train_name, test_name, model_name):
    model.eval()
    class_correct = [0] * 6
    class_total = [0] * 6
    all_labels = []
    all_predictions = []
    all_probs = []  # For storing probability distributions

    classes = ['K', 'Q', 'R', 'B', 'N', 'P']

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = model(X)
            probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)

            # Collect all labels, predictions, and probabilities
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Calculate per-class accuracy
            for i in range(len(y)):
                label = y[i].item()
                class_total[label] += 1
                if predicted[i] == y[i]:
                    class_correct[label] += 1

    # Initialize metrics
    correct_counts = []
    incorrect_counts = []
    accuracies = []
    recalls = []
    f1_scores = []
    brier_scores = []
    normalized_brier_scores = []
    cross_entropies = []

    # Compute metrics per piece type
    for i in range(6):
        if class_total[i] > 0:
            # Accuracy-based metrics
            accuracy = class_correct[i] / class_total[i]
            recall = recall_score(all_labels, all_predictions, labels=[i], average='macro')
            f1 = f1_score(all_labels, all_predictions, labels=[i], average='macro')
            
            # Compute Brier Score and Cross Entropy Loss
            labels_binary = [1 if label == i else 0 for label in all_labels]
            predicted_probs = [prob[i] for prob in all_probs]

            brier = brier_score_loss(labels_binary, predicted_probs)
            normalized_brier = 6 * (len(labels_binary)) * brier / class_total[i] if class_total[i] > 0 else 0

            cross_entropy = F.cross_entropy(
                torch.tensor(all_probs), 
                torch.tensor(all_labels), 
                reduction='mean'
            ).item()

            # Storing results
            correct_counts.append(class_correct[i])
            incorrect_counts.append(class_total[i] - class_correct[i])
            accuracies.append(accuracy)
            recalls.append(recall)
            f1_scores.append(f1)
            brier_scores.append(brier)
            normalized_brier_scores.append(normalized_brier)
            cross_entropies.append(cross_entropy)

            print(f"{classes[i]}: Accuracy: {accuracy:.4f} | Brier Score: {brier:.4f} | Cross-Entropy Loss: {cross_entropy:.4f}")

        else:
            print(f"{classes[i]}: No samples available.")
            correct_counts.append(0)
            incorrect_counts.append(0)
            accuracies.append(0)
            recalls.append(0)
            f1_scores.append(0)
            brier_scores.append(0)
            cross_entropies.append(0)

    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    total_accuracy = total_correct / total_samples
    weighted_accuracy = sum([accuracies[i] * class_total[i] for i in range(6)]) / total_samples

    print(f"\nTotal Accuracy: {total_accuracy:.4f}")
    print(f"Weighted Accuracy: {weighted_accuracy:.4f}")

    # Save this as an Excel file
    df = pd.DataFrame({
        'Piece Type': classes,
        'Correct Predictions': correct_counts,
        'Incorrect Predictions': incorrect_counts,
        'Cross-Entropy Loss': cross_entropies
    })

    df.to_excel(f'{player}_per_piece_metrics.xlsx', index=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(classes))

    plt.bar(index, correct_counts, bar_width, label='Correct', color='darkblue')
    plt.bar([i + bar_width for i in index], incorrect_counts, bar_width, label='Incorrect', color='lightblue')

    plt.xlabel('Piece Type')
    plt.ylabel('Count')
    plt.title(f'{model_name} Correct vs Incorrect Predictions by Piece Type   Train: {train_name}, Test: {test_name}')
    plt.xticks([i + bar_width / 2 for i in index], classes)
    plt.legend()
    plt.show()

    # Summary DataFrame for metrics
    df_summary = pd.DataFrame({
        'Piece Type': classes,
        'Accuracy': accuracies,
        'Recall': recalls,
        'F1 Score': f1_scores,
        'Vanilla Brier Score': brier_scores,
        'Normalized Brier Score': normalized_brier_scores,
        # 'Cross-Entropy Loss': cross_entropies,
        'Number of Samples': class_total
    })

    # Save the summary to Excel
    df_summary.to_excel(f'{player}_overall_metrics.xlsx', index=False)
    print("\nOverall Metrics Summary:")
    print(df_summary)

    return df_summary