import os
import base64
import numpy as np
import pandas as pd
from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList
from llama_cpp.llama_chat_format import Llava16ChatHandler
from load_ollama_models import get_ollama_blob_path
import joblib

# Configuration
PROMPT_TEXT = "You are a robot. Based on what you see, you must identify the object gripped in your robotic arm. Disregard the arm and background clutter, focus on the object. Be concise, identify the object."
PREFIX_STRING = "The object is a"

IMAGE_DIR = "dataset_vlm_0328"
# Images of the same object, 4 on robot art, 1 on table.
# This is a simulation. In reality, when connected to the robot, we should keep taking new images until we are sure enough.
IMAGE_NAMES = ["01.jpg", "02.jpg", "03.jpg", "04.jpg", "49.jpg"] # example with coffee cup images
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES if os.path.exists(os.path.join(IMAGE_DIR, name))]
if not IMAGE_PATHS:
    print(f"Error: No images found in {IMAGE_DIR}.")
    exit(1)

# VLM Model
MODEL_NAME = "LlaVA 7B"
MODEL_PATH = get_ollama_blob_path("llava", "model")
MMPROJ_PATH = get_ollama_blob_path("llava", "projector")

# Which threshold to use for being "sure" enough to stop
# Options: 'optimal' or 'zero_fp'. Or a custom float value (ex: 0.85)
CONFIDENCE_TARGET_TYPE = 'zero_fp'

# Classifier
CLASSIFIER_MODEL_PATH = "3-classifier_llava7b.joblib"
BASE_FEATURE_COLUMNS = [
    'mean_prob', 'min_prob', 'max_prob', 'std_dev_prob',
    'prob_first_token', 'prob_last_token', 'suffix_len',
]

# VLM Generation Params
N_GPU_LAYERS = -1
N_CTX = 2048
TOP_K_FOR_PREFIX_SEARCH = 32000
MAX_SUFFIX_TOKENS_TO_GENERATE = 10
VLM_TEMPERATURE = 0.0


# Helper Functions
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_prob_pct(prob, numbers=3):
    return f"{prob * 100:.{numbers}f}%"

# VLM Logits Processor
class ForcePrefixAndRecordGreedySuffixProcessor(LogitsProcessor):
    def __init__(self, llm_instance, fixed_prefix_token_ids, top_k_check=500, verbose=False):
        self.llm = llm_instance
        self.fixed_prefix_token_ids = fixed_prefix_token_ids
        self.num_prefix_steps = len(fixed_prefix_token_ids)
        self.top_k_check = top_k_check
        self.verbose = verbose

        self.current_step_idx = 0
        self.generated_suffix_details = []

    def __call__(self, input_ids, scores):
        scores_np = np.array(scores, dtype=np.float32)
        scores_exp = np.exp(scores_np - np.max(scores_np))
        probs = scores_exp / np.sum(scores_exp)

        is_prefix_step = self.current_step_idx < self.num_prefix_steps
        target_token_id = -1

        if is_prefix_step:
            target_token_id = self.fixed_prefix_token_ids[self.current_step_idx]
        else: # suffix step
            chosen_token_id = np.argmax(probs)
            chosen_token_prob = probs[chosen_token_id]
            token_str = self.llm.tokenizer().decode([int(chosen_token_id)])

            self.generated_suffix_details.append({
                'token_str': token_str,
                'prob': float(chosen_token_prob)
            })
            target_token_id = chosen_token_id

        # Force target_token_id
        modified_scores = np.full_like(scores_np, -float('inf'))
        modified_scores[target_token_id] = 0.0

        self.current_step_idx += 1
        return modified_scores

    def get_results(self):
        return {
            'suffix_text': "".join([d['token_str'] for d in self.generated_suffix_details]).strip(),
            'suffix_token_details': self.generated_suffix_details,
        }


# Classifier functions
def extract_features_for_classifier(prob_list: list[float]):
    P = np.array(prob_list)
    features = {
        'mean_prob': np.mean(P),
        'min_prob': np.min(P),
        'max_prob': np.max(P),
        'std_dev_prob': np.std(P) if len(P) > 1 else 0.0,
        'prob_first_token': P[0],
        'prob_last_token': P[-1],
        'suffix_len': float(len(P)),
    }
    return pd.Series(features, index=BASE_FEATURE_COLUMNS)

def get_classifier_prob_class1(classifier_model, scaled_features_array):
    pred_prob_all = classifier_model.predict_proba(scaled_features_array)
    if pred_prob_all.shape[1] == 2:
        if classifier_model.classes_[1] == 1:
            return pred_prob_all[:, 1]
        else:
            return pred_prob_all[:, 0]
    elif pred_prob_all.shape[1] == 1:
        if classifier_model.classes_[0] == 1:
            return pred_prob_all[:, 0]
    return np.zeros(scaled_features_array.shape[0])



# Main Execution
if __name__ == "__main__":
    print(f"===== Loading VLM Model: {MODEL_NAME} =====")
    if not (MODEL_PATH and MMPROJ_PATH and os.path.exists(MODEL_PATH) and os.path.exists(MMPROJ_PATH)):
        print(f"Error: Model or projector path not found or not configured for {MODEL_NAME}.")
        print(f"Model: {MODEL_PATH}, MMProj: {MMPROJ_PATH}")
        exit(1)
    try:
        chat_handler = Llava16ChatHandler(clip_model_path=MMPROJ_PATH, verbose=False)
        llm = Llama(
            model_path=MODEL_PATH,
            chat_handler=chat_handler,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            logits_all=True,
            verbose=False
        )
        prefix_tokens = llm.tokenizer().encode(PREFIX_STRING, add_bos=False)
    except Exception as e:
        print(f"Error loading VLM model {MODEL_NAME}: {e}")
        exit(1)

    print(f"\n===== Loading Classifier from: {CLASSIFIER_MODEL_PATH} =====")
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        print(f"Error: Classifier model file not found at {CLASSIFIER_MODEL_PATH}")
        exit(1)
    try:
        classifier_data = joblib.load(CLASSIFIER_MODEL_PATH)
        classifier_model = classifier_data['model']
        scaler = classifier_data['scaler']
        optimal_threshold = classifier_data.get('optimal_threshold', 0.5)
        zero_fp_threshold = classifier_data.get('zero_fp_threshold', 0.5)
        loaded_feature_names = classifier_data.get('feature_names', [])

        if loaded_feature_names != BASE_FEATURE_COLUMNS:
            print(f"Error: Feature names mismatch! Loaded: {loaded_feature_names}, Current: {BASE_FEATURE_COLUMNS}")
            exit(1)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        exit(1)

    # Determine the stopping threshold
    stop_threshold_value = 0.0
    if isinstance(CONFIDENCE_TARGET_TYPE, str):
        if CONFIDENCE_TARGET_TYPE == 'optimal':
            stop_threshold_value = optimal_threshold
            print(f"Will stop if confidence >= optimal_threshold ({format_prob_pct(stop_threshold_value,2)})")
        elif CONFIDENCE_TARGET_TYPE == 'zero_fp':
            stop_threshold_value = zero_fp_threshold
            print(f"Will stop if confidence >= zero_fp_threshold ({format_prob_pct(stop_threshold_value,2)})")
        else:
            print(f"Warning: Unknown CONFIDENCE_TARGET_TYPE string: '{CONFIDENCE_TARGET_TYPE}'. Using optimal_threshold.")
            stop_threshold_value = optimal_threshold
    elif isinstance(CONFIDENCE_TARGET_TYPE, float):
        stop_threshold_value = CONFIDENCE_TARGET_TYPE
        print(f"Will stop if confidence >= custom threshold ({format_prob_pct(stop_threshold_value,2)})")
    else:
        print(f"Warning: Invalid CONFIDENCE_TARGET_TYPE. Using optimal_threshold.")
        stop_threshold_value = optimal_threshold


    best_overall_confidence = -1.0
    best_overall_vlm_suffix = "N/A"
    best_overall_suffix_details = []
    identified_confidently = False

    final_identification_text = "N/A"
    final_identification_confidence = 0.0
    final_identification_details = []


    for i, current_image_path in enumerate(IMAGE_PATHS):
        print(f"\n===== Processing Image {i+1}/{len(IMAGE_PATHS)}: {current_image_path} =====")
        if not os.path.exists(current_image_path):
            print(f"Error: Image file not found at {current_image_path}, skipping.")
            continue

        base64_image = image_to_base64(current_image_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": PROMPT_TEXT}
            ]}
        ]

        vlm_processor = ForcePrefixAndRecordGreedySuffixProcessor(
            llm_instance=llm,
            fixed_prefix_token_ids=prefix_tokens,
            top_k_check=TOP_K_FOR_PREFIX_SEARCH,
            verbose=False
        )
        logits_processors = LogitsProcessorList([vlm_processor])

        try:
            completion = llm.create_chat_completion(
                messages=messages,
                max_tokens=len(prefix_tokens) + MAX_SUFFIX_TOKENS_TO_GENERATE,
                temperature=VLM_TEMPERATURE,
                logits_processor=logits_processors,
                stop=["<0x0A>", "\n", "</s>", ".", ",", ";"] # stop tokens
            )
        except Exception as e:
            print(f"Error during VLM generation for {current_image_path}: {e}")
            continue

        vlm_results = vlm_processor.get_results()
        generated_suffix_text = vlm_results['suffix_text']
        suffix_token_details = vlm_results['suffix_token_details']

        print(f"        \033[1m{generated_suffix_text}\033[0m")

        suffix_probabilities = [detail['prob'] for detail in suffix_token_details]
        features_series = extract_features_for_classifier(suffix_probabilities)
        features_for_scaling = pd.DataFrame([features_series.values], columns=BASE_FEATURE_COLUMNS)

        scaled_features = scaler.transform(features_for_scaling)
        prob_class1 = get_classifier_prob_class1(classifier_model, scaled_features)[0]

        # Update best result if current is better
        if prob_class1 > best_overall_confidence:
            best_overall_confidence = prob_class1
            best_overall_vlm_suffix = generated_suffix_text
            best_overall_suffix_details = suffix_token_details

        # Check if confidence target is met
        if prob_class1 >= stop_threshold_value:
            print(f"Classifier Confidence (in 'Correctness'): \033[92m{format_prob_pct(prob_class1,2)}\033[0m (>= {format_prob_pct(stop_threshold_value,2)}). Stopping here.")
            identified_confidently = True
            final_identification_text = PREFIX_STRING + " \033[1m" + generated_suffix_text + "\033[0m"
            final_identification_confidence = prob_class1
            final_identification_details = suffix_token_details
            break
        else:
            if i < len(IMAGE_PATHS) - 1:
                print(f"Classifier Confidence (in 'Correctness'): \033[91m{format_prob_pct(prob_class1,2)}\033[0m (< {format_prob_pct(stop_threshold_value,2)}). Requesting next image...")
            else:
                print(f"Processed all {len(IMAGE_PATHS)} images. Highest confidence \033[91m{format_prob_pct(best_overall_confidence,2)}\033[0m did not reach target {format_prob_pct(stop_threshold_value,2)}.\n(This is where the robot would keep taking images, or stop after a certain number of attempts.)")


    print("\n\n==================== FINAL ROBOT CONCLUSION ====================")
    if identified_confidently:
        print(f"Robot is \033[92mSURE\033[0m. Identified Object:\n\033[1m{best_overall_vlm_suffix}\033[0m")
        print(f"Confidence: {format_prob_pct(final_identification_confidence, 2)} (Target: {format_prob_pct(stop_threshold_value, 2)})")
        print("\nSuffix Token Probabilities for the confident identification:")
        for detail in final_identification_details:
            print(f"  Token: {repr(detail['token_str'])}, P: {format_prob_pct(detail['prob'])}")
    else:
        print(f"Robot is \033[91mUNSURE\033[0m after checking {len(IMAGE_PATHS)} images. Best guess:\n\033[1m{best_overall_vlm_suffix}\033[0m")
        print(f"Confidence: {format_prob_pct(best_overall_confidence, 2)} (Target: {format_prob_pct(stop_threshold_value,2)})")
        print("\nSuffix Token Probabilities for the best guess:")
        if best_overall_suffix_details:
            for detail in best_overall_suffix_details:
                print(f"  Token: {repr(detail['token_str'])}, P: {format_prob_pct(detail['prob'])}")
        else:
            print("  No suffix tokens were generated for the best guess attempt.")

    print("================================================================")