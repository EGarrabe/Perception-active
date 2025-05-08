import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import joblib

# Configuration
CSV_FILE_PATH = "2-suffix_probas_240img_llava7b.csv"
MODEL_SAVE_PATH = "classifier_llava7b.joblib"
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200 # Number of trees in the Random Forest

# Correct labels
RAW_LABELS = {
    ("01.jpg", "02.jpg", "03.jpg", "04.jpg", "49.jpg"): ["mug", "cup"],
    ("05.jpg", "06.jpg", "07.jpg", "08.jpg", "51.jpg"): ["sponge"],
    ("09.jpg", "10.jpg", "11.jpg", "12.jpg", "50.jpg"): ["glove", "gloves"],
    ("13.jpg", "14.jpg", "15.jpg", "16.jpg", "52.jpg"): ["plate", "dish", "tray"],
    ("17.jpg", "18.jpg", "19.jpg", "20.jpg", "53.jpg"): ["fan"],
    ("21.jpg", "22.jpg", "23.jpg", "24.jpg", "54.jpg"): ["soap", "hand sanitizer", "cleaner", "cleaning solution"],
    ("25.jpg", "26.jpg", "27.jpg", "28.jpg", "56.jpg"): ["controller"],
    ("29.jpg", "30.jpg", "31.jpg", "32.jpg", "55.jpg"): ["cloth", "rag", "napkin"],
    ("33.jpg", "34.jpg", "35.jpg", "36.jpg", "59.jpg"): ["mouse"],
    ("37.jpg", "38.jpg", "39.jpg", "40.jpg", "57.jpg"): ["screwdriver", "screw driver"],
    ("41.jpg", "42.jpg", "43.jpg", "44.jpg", "58.jpg"): ["circuit board", "computer board", "electronic board", "electronics board", "microcontroller"],
    ("45.jpg", "46.jpg", "47.jpg", "48.jpg", "60.jpg"): ["cleaning foam", "cleaning solution", "cleaning product", "whiteboard cleaner", "whiteboard foam"],
}

# for augmented dataset (60->240 images). Comment by default.
# RAW_LABELS.update({
#     tuple(f"{i:03}.jpg" for i in range(61, 73)) + ("205.jpg", "206.jpg", "207.jpg"): ["mug", "cup"],
#     tuple(f"{i:03}.jpg" for i in range(73, 85)) + ("211.jpg", "212.jpg", "213.jpg"): ["sponge"],
#     tuple(f"{i:03}.jpg" for i in range(85, 97)) + ("208.jpg", "209.jpg", "210.jpg"): ["glove", "gloves"],
#     tuple(f"{i:03}.jpg" for i in range(97, 109)) + ("214.jpg", "215.jpg", "216.jpg"): ["plate", "dish", "tray"],
#     tuple(f"{i:03}.jpg" for i in range(109, 121)) + ("217.jpg", "218.jpg", "219.jpg"): ["fan"],
#     tuple(f"{i:03}.jpg" for i in range(121, 133)) + ("220.jpg", "221.jpg", "222.jpg"): ["soap", "hand sanitizer", "cleaner", "cleaning solution"],
#     tuple(f"{i:03}.jpg" for i in range(133, 145)) + ("226.jpg", "227.jpg", "228.jpg"): ["controller"],
#     tuple(f"{i:03}.jpg" for i in range(145, 157)) + ("223.jpg", "224.jpg", "225.jpg"): ["cloth", "rag", "napkin"],
#     tuple(f"{i:03}.jpg" for i in range(157, 169)) + ("235.jpg", "236.jpg", "237.jpg"): ["mouse"],
#     tuple(f"{i:03}.jpg" for i in range(169, 181)) + ("229.jpg", "230.jpg", "231.jpg"): ["screwdriver", "screw driver"],
#     tuple(f"{i:03}.jpg" for i in range(181, 193)) + ("232.jpg", "233.jpg", "234.jpg"): ["circuit board", "computer board", "electronic board", "electronics board", "microcontroller"],
#     tuple(f"{i:03}.jpg" for i in range(193, 205)) + ("238.jpg", "239.jpg", "240.jpg"): ["cleaning foam", "cleaning solution", "cleaning product", "whiteboard cleaner", "whiteboard foam"],
# })


# Preprocessing RAW_LABELS
CORRECT_LABELS_MAP = {}
IMAGE_TO_OBJECT_ID = {}
RANKS_TO_INCLUDE = [1] # 1 = Only first greedy path

for image_tuple, labels in RAW_LABELS.items():
    first_label_full = labels[0].lower().strip()
    object_id = first_label_full.split()[0] if first_label_full else "unknown_object"
    for img_name in image_tuple:
        clean_img_name = img_name.strip()
        CORRECT_LABELS_MAP[clean_img_name] = [label.lower().strip() for label in labels]
        IMAGE_TO_OBJECT_ID[clean_img_name] = object_id

# Feature columns
BASE_FEATURE_COLUMNS = [
    'mean_prob', 'min_prob', 'max_prob', 'std_dev_prob', 'prob_first_token', 'prob_last_token', 'suffix_len',
]

def parse_probabilities(prob_string):
    probabilities = []
    if not isinstance(prob_string, str) or prob_string == "[ERROR]" or not prob_string:
        return []
    try:
        pairs = prob_string.split(';')
        for pair in pairs:
            if ':' in pair:
                parts = pair.split(':')
                if len(parts) == 2:
                    prob = float(parts[1])
                    probabilities.append(prob)
    except (ValueError, IndexError):
        return []
    return probabilities

def get_ground_truth(image_name, suffix_text):
    suffix_clean = re.sub(r"^[.,!?;'\"]+|[.,!?;'\"]+$", "", str(suffix_text).lower().strip()).strip()
    for keyword in CORRECT_LABELS_MAP.get(image_name, []):
        if re.search(r'\b' + re.escape(keyword) + r'\b', suffix_clean):
            return 1
    return 0

def extract_features(prob_list):
    if not prob_list: # Handles empty list from parse_probabilities
        return pd.Series({col: 0.0 for col in BASE_FEATURE_COLUMNS}, index=BASE_FEATURE_COLUMNS)
    probs = np.array(prob_list)
    features = {
        'mean_prob': np.mean(probs), 'min_prob': np.min(probs), 'max_prob': np.max(probs),
        'std_dev_prob': np.std(probs) if len(probs) > 1 else 0.0,
        'prob_first_token': probs[0], 'prob_last_token': probs[-1],
        'suffix_len': float(len(probs)),
    }
    return pd.Series(features, index=BASE_FEATURE_COLUMNS)

def get_probs_class1(model, X_scaled_set):
    pred_prob_all = model.predict_proba(X_scaled_set)
    if pred_prob_all.shape[1] == 2:
        # Common case: model learned [class0, class1]
        if model.classes_[1] == 1:
            return pred_prob_all[:, 1]
        else: # Should be model.classes_[0] == 1
            return pred_prob_all[:, 0]
    else:
        # Model only predicted one class
        if model.classes_[0] == 1: # Model only knows/predicts class 1
            return pred_prob_all[:, 0]
    # Model only knows/predicts class 0
    return np.zeros(X_scaled_set.shape[0]) # Prob of class 1 is 0


# 1. Load Data
try:
    df_full = pd.read_csv(CSV_FILE_PATH)
    print(f"Loaded {len(df_full)} rows from {CSV_FILE_PATH}")
    if df_full.empty:
        exit(f"Error: No data found in {CSV_FILE_PATH}")
except Exception as e:
    exit(f"Error loading CSV {CSV_FILE_PATH}: {e}")


# Filter for known images and specified ranks
known_image_names = list(IMAGE_TO_OBJECT_ID.keys())
df_full = df_full[df_full['image_name'].isin(known_image_names)].copy()
df_full = df_full[df_full['path_rank'].isin(RANKS_TO_INCLUDE)].copy()
print(f"Filtered to {len(df_full)} rows with known image names and specified ranks.")

# 2. Preprocessing and Feature Extraction
df_full['probabilities'] = df_full['suffix_token_id_probs'].apply(parse_probabilities)
features_df = df_full['probabilities'].apply(extract_features)
df_full = pd.concat([df_full, features_df], axis=1)
df_full['is_correct'] = df_full.apply(lambda row: get_ground_truth(row['image_name'], row['suffix_text']), axis=1)
df_full['object_id'] = df_full['image_name'].map(IMAGE_TO_OBJECT_ID)

# 3. Prepare Data for Splitting
X = df_full[BASE_FEATURE_COLUMNS]
y = df_full['is_correct']
object_ids_for_stratify = df_full['object_id']

print("\nOverall Label Distribution before split:")
print(y.value_counts(normalize=True))

# 4. Train-Test Split
print(f"\nSplitting data: {1-TEST_SET_SIZE:.0%} train, {TEST_SET_SIZE:.0%} test")
try:
    train_indices, test_indices = train_test_split(
        df_full.index,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE,
        stratify=object_ids_for_stratify[df_full.index]
    )
    df_train = df_full.loc[train_indices].copy()
    df_test = df_full.loc[test_indices].copy()

except ValueError as e:
    print(f"Warning: Stratified split failed ({e}) -> non-stratified split fallback")
    train_indices, test_indices = train_test_split(
        df_full.index,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE
        # No stratify here
    )
    df_train = df_full.loc[train_indices].copy()
    df_test = df_full.loc[test_indices].copy()


X_train, y_train = df_train[BASE_FEATURE_COLUMNS], df_train['is_correct']
X_test, y_test = df_test[BASE_FEATURE_COLUMNS], df_test['is_correct']

print(f"Training Set ({len(df_train)} samples) Label Distribution:")
print(y_train.value_counts(normalize=True))

print(f"Test Set ({len(df_test)} samples) Label Distribution:")
print(y_test.value_counts(normalize=True))

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Training
print(f"RandomForestClassifier :{N_ESTIMATORS} trees, class_weight='balanced', min_samples_leaf=2")
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, class_weight='balanced', min_samples_leaf=2)
model.fit(X_train_scaled, y_train)

# 7. Threshold Calculation (based on Training Data)
train_probs_class1 = get_probs_class1(model, X_train_scaled)

# 7.1 Accuracy-Optimal Threshold
print("\nCalculating Accuracy-Optimal threshold based on training data...")
optimal_threshold = 0.5
max_accuracy_on_train = 0.0
if len(np.unique(y_train)) > 1:
    thresholds_opt = np.linspace(0.0, 1.0, 201)
    accuracies_opt = [accuracy_score(y_train, (train_probs_class1 >= thresh).astype(int)) for thresh in thresholds_opt]
    max_accuracy_on_train = np.max(accuracies_opt)
    best_threshold_indices = np.where(accuracies_opt == max_accuracy_on_train)[0]
    mid_best_index = best_threshold_indices[len(best_threshold_indices) // 2]
    optimal_threshold = thresholds_opt[mid_best_index]
    print(f"  Max Accuracy on Train: {max_accuracy_on_train:.4f} with threshold {optimal_threshold:.4f}")
else:
    pred_opt_default = (train_probs_class1 >= 0.5).astype(int)
    max_accuracy_on_train = accuracy_score(y_train, pred_opt_default)
    print(f"  Warning: Only one class in y_train or model predicts trivially. Using default threshold 0.5.\n  Accuracy on Train: {max_accuracy_on_train:.4f}")


# 7.2 Zero FP Threshold
print("\nCalculating Zero FP threshold based on training data...")
threshold_zero_fp = 1.0
max_accuracy_in_zero_fp_range_train = 0.0

if len(np.unique(y_train)) > 1:
    neg_probs_train = train_probs_class1[y_train == 0]
    min_required_threshold_zero_fp = 1.0
    if len(neg_probs_train) > 0:
        min_required_threshold_zero_fp = np.max(neg_probs_train) + 1e-9
        min_required_threshold_zero_fp = min(min_required_threshold_zero_fp, 1.0 - 1e-9)
    else:
        min_required_threshold_zero_fp = 1e-9

    print(f"  Min Required Threshold for Zero FP on Train: {min_required_threshold_zero_fp:.6f}")
    
    thresholds_to_check_zero_fp = np.linspace(min_required_threshold_zero_fp, 1.0, 101)
    accuracies_zero_fp = []
    actual_thresholds_checked_zero_fp = []

    for thresh in thresholds_to_check_zero_fp:
        y_pred_thresh = (train_probs_class1 >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred_thresh, labels=[0, 1]).ravel()
        if fp == 0:
            accuracies_zero_fp.append(accuracy_score(y_train, y_pred_thresh))
            actual_thresholds_checked_zero_fp.append(thresh)

    if accuracies_zero_fp:
        max_accuracy_in_zero_fp_range_train = np.max(accuracies_zero_fp)
        best_indices_zero_fp = np.where(np.array(accuracies_zero_fp) == max_accuracy_in_zero_fp_range_train)[0]
        mid_best_idx_zero_fp = best_indices_zero_fp[len(best_indices_zero_fp) // 2]
        threshold_zero_fp = actual_thresholds_checked_zero_fp[mid_best_idx_zero_fp]
        print(f"  Max Accuracy within Zero FP Range (Train): {max_accuracy_in_zero_fp_range_train:.4f}")
        print(f"  Chosen Zero FP Threshold (Middle of Best, Train): {threshold_zero_fp:.6f}")
    else:
        print("  Warning: Could not find a threshold in the checked range that yields Zero FP and has accuracy data. Using min_required or 1.0.")
        threshold_zero_fp = min_required_threshold_zero_fp if min_required_threshold_zero_fp <= 1.0 else 1.0
        y_pred_final_zero_fp = (train_probs_class1 >= threshold_zero_fp).astype(int)
        max_accuracy_in_zero_fp_range_train = accuracy_score(y_train, y_pred_final_zero_fp)

else:
    print("  Warning: Only one class in y_train or model predicts trivially. Zero FP threshold might not be meaningful.")
    pred_default_zero_fp = (train_probs_class1 >= 1.0).astype(int)
    max_accuracy_in_zero_fp_range_train = accuracy_score(y_train, pred_default_zero_fp)


# 8. Plotting Threshold Performance
def plot_threshold_performance(probs_class1_set, y_true_set, opt_thresh, zfp_thresh, title_suffix, plot_filename_suffix):
    print(f"\nGenerating Threshold vs Accuracy/FP Plot for {title_suffix}...")
    if len(np.unique(y_true_set)) < 2 or len(probs_class1_set) == 0:
        print(f"  Skipping plot for {title_suffix}: not enough class diversity or no probabilities.")
        return

    thresholds_to_plot = np.linspace(0.2, 0.8, 201)
    accuracies_plot = []
    false_positives_plot = []

    for thresh_plot in thresholds_to_plot:
        y_pred_plot = (probs_class1_set >= thresh_plot).astype(int)
        accuracies_plot.append(accuracy_score(y_true_set, y_pred_plot))
        tn, fp, fn, tp = confusion_matrix(y_true_set, y_pred_plot, labels=[0, 1]).ravel()
        false_positives_plot.append(fp)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(thresholds_to_plot, accuracies_plot, color=color, linestyle='-', marker='.', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':')
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Number of False Positives', color=color)
    ax2.plot(thresholds_to_plot, false_positives_plot, color=color, linestyle='--', marker='x', markersize=4, label='False Positives')
    ax2.tick_params(axis='y', labelcolor=color)
    max_fp_plot = np.max(false_positives_plot) if false_positives_plot else 0
    if max_fp_plot < 15 and max_fp_plot >= 0 : ax2.set_yticks(np.arange(0, int(max_fp_plot) + 2, 1))


    ax1.axvline(opt_thresh, color='tab:green', linestyle='--', linewidth=1.5, label=f'Optimal Threshold ({opt_thresh:.3f})')
    ax1.axvline(zfp_thresh, color='tab:orange', linestyle='-.', linewidth=1.5, label=f'Zero FP Threshold ({zfp_thresh:.3f})')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title(f'Accuracy and False Positives vs. Threshold ({title_suffix})')
    fig.tight_layout()
    plot_filename = f"threshold_performance_plot_{plot_filename_suffix}.png"
    plt.savefig(plot_filename)
    print(f"Saved threshold plot to {plot_filename}")
    plt.show()

plot_threshold_performance(train_probs_class1, y_train, optimal_threshold, threshold_zero_fp, "Training Data", "train")
test_probs_class1 = get_probs_class1(model, X_test_scaled)
plot_threshold_performance(test_probs_class1, y_test, optimal_threshold, threshold_zero_fp, "Test Data", "test")


# 9. Save Model Package
print(f"\n--- Saving Model Package to {MODEL_SAVE_PATH} ---")
save_data = {
    'model': model,
    'scaler': scaler,
    'optimal_threshold': optimal_threshold,
    'zero_fp_threshold': threshold_zero_fp,
    'max_accuracy_on_train_at_optimal_thresh': max_accuracy_on_train,
    'max_accuracy_on_train_at_zero_fp_thresh': max_accuracy_in_zero_fp_range_train,
    'feature_names': BASE_FEATURE_COLUMNS,
}
joblib.dump(save_data, MODEL_SAVE_PATH)
print("Model, scaler, thresholds, and feature names saved successfully.")


# 10. Evaluate and Report on Samples
def evaluate_report_and_collect_preds(df_eval_set, X_eval_scaled, y_eval_true, set_name, opt_thresh, zfp_thresh):
    print(f"\n--- Evaluating Trained Model on Each Input Line ({set_name}) ---")
    print("-" * 110)
    print(f"{'Object':<12} | {'Image':<7} | {'Rank':<4} | {'Ground Truth':<13} | {'Opt Predict':<12} | {'ZeroFP Predict':<15} | {'Prob (C=1)':<11} | {'Suffix Text'}")
    print("-" * 110)

    df_report = df_eval_set.copy()
    df_report['prob_class1'] = get_probs_class1(model, X_eval_scaled)
    df_report['pred_opt'] = (df_report['prob_class1'] >= opt_thresh).astype(int)
    df_report['pred_zero_fp'] = (df_report['prob_class1'] >= zfp_thresh).astype(int)

    all_gt_list = []
    all_pred_opt_list = []
    all_pred_zero_fp_list = []

    df_report_sorted = df_report.sort_values(by=['image_name', 'path_rank'])

    for _, row in df_report_sorted.iterrows():
        obj_id = row['object_id']
        img_name = row['image_name']
        path_rank = row['path_rank']
        suffix_text = row['suffix_text']
        ground_truth = row['is_correct']
        
        prob_c1 = row['prob_class1']
        pred_opt = row['pred_opt']
        pred_zero_fp = row['pred_zero_fp']

        all_gt_list.append(ground_truth)
        all_pred_opt_list.append(pred_opt)
        all_pred_zero_fp_list.append(pred_zero_fp)

        gt_str = "Correct" if ground_truth == 1 else "Incorrect"
        pred_opt_str = "Correct" if pred_opt == 1 else "Incorrect"
        pred_zero_fp_str = "Correct" if pred_zero_fp == 1 else "Incorrect"
        prob_str = f"{prob_c1:.4f}"

        print(f"{obj_id:<12} | {img_name:<7} | {path_rank:<4} | {gt_str:<13} | {pred_opt_str:<12} | {pred_zero_fp_str:<15} | {prob_str:<11} | {suffix_text[:50]}") # Truncate suffix
    
    print("-" * 110)
    return all_gt_list, all_pred_opt_list, all_pred_zero_fp_list

gt_train, pred_opt_train, pred_zfp_train = evaluate_report_and_collect_preds(df_train, X_train_scaled, y_train, "Training Set", optimal_threshold, threshold_zero_fp)
gt_test, pred_opt_test, pred_zfp_test = evaluate_report_and_collect_preds(df_test, X_test_scaled, y_test, "Test Set", optimal_threshold, threshold_zero_fp)


# 11. Calculate Overall Accuracies
print("\n--- Overall Performance Metrics ---")

print("\n--- Training Data ---")
acc_opt_train = accuracy_score(gt_train, pred_opt_train)
print(f"Optimal Accuracy  (Threshold = {optimal_threshold:.4f}): {acc_opt_train:.4f}")
cm_opt_train = confusion_matrix(gt_train, pred_opt_train, labels=[0,1]).ravel()
print(f"  TN={cm_opt_train[0]}, FP={cm_opt_train[1]}, FN={cm_opt_train[2]}, TP={cm_opt_train[3]}")


acc_zfp_train = accuracy_score(gt_train, pred_zfp_train)
print(f"Zero FP Accuracy  (Threshold = {threshold_zero_fp:.4f}): {acc_zfp_train:.4f}")
cm_zfp_train = confusion_matrix(gt_train, pred_zfp_train, labels=[0,1]).ravel()
print(f"  TN={cm_zfp_train[0]}, FP={cm_zfp_train[1]}, FN={cm_zfp_train[2]}, TP={cm_zfp_train[3]}")


print("\n--- Test Data ---")
acc_opt_test = accuracy_score(gt_test, pred_opt_test)
print(f"Optimal Accuracy  (Threshold = {optimal_threshold:.4f}): {acc_opt_test:.4f}")
cm_opt_test = confusion_matrix(gt_test, pred_opt_test, labels=[0,1]).ravel()
print(f"  TN={cm_opt_test[0]}, FP={cm_opt_test[1]}, FN={cm_opt_test[2]}, TP={cm_opt_test[3]}")

acc_zfp_test = accuracy_score(gt_test, pred_zfp_test)
print(f"Zero FP Accuracy  (Threshold = {threshold_zero_fp:.4f}): {acc_zfp_test:.4f}")
cm_zfp_test = confusion_matrix(gt_test, pred_zfp_test, labels=[0,1]).ravel()
print(f"  TN={cm_zfp_test[0]}, FP={cm_zfp_test[1]}, FN={cm_zfp_test[2]}, TP={cm_zfp_test[3]}")
if cm_zfp_test[1] != 0:
    print(f"  (Warning: False Positives = {cm_zfp_test[1]} on test data with Zero FP threshold. Threshold was optimized for train data.)")