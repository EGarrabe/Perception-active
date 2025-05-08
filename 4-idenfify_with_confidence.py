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
SINGLE_IMAGE_PATH = "dataset_vlm_0328/09.jpg"

# VLM Model Config
MODEL_NAME = "LlaVA 7B"
MODEL_PATH = get_ollama_blob_path("llava", "model")
MMPROJ_PATH = get_ollama_blob_path("llava", "projector")

# Classifier Config
CLASSIFIER_MODEL_PATH = "3-classifier_llava7b.joblib"
BASE_FEATURE_COLUMNS = [ # must match the classifier training features
    'mean_prob', 'min_prob', 'max_prob', 'std_dev_prob',
    'prob_first_token', 'prob_last_token', 'suffix_len',
]

# VLM Generation Params
N_GPU_LAYERS = -1
N_CTX = 2048
TOP_K_FOR_PREFIX_SEARCH = 32000 # How deep to look for prefix tokens (here vocab size)
MAX_SUFFIX_TOKENS_TO_GENERATE = 10 # Max tokens for the object name
VLM_TEMPERATURE = 0.0 # For deterministic output

# --- Helper Functions ---
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_prob_pct(prob, numbers=3):
    return f"{prob * 100:.{numbers}f}%"

# --- VLM Logits Processor ---
class ForcePrefixAndRecordGreedySuffixProcessor(LogitsProcessor):
    def __init__(self, llm_instance, fixed_prefix_token_ids, top_k_check=500, verbose=False):
        self.llm = llm_instance
        self.fixed_prefix_token_ids = fixed_prefix_token_ids
        self.num_prefix_steps = len(fixed_prefix_token_ids)
        self.top_k_check = top_k_check
        self.verbose = verbose

        self.current_step_idx = 0
        self.generated_suffix_details = []
        self.prefix_forced_token_ids = []
        self.suffix_forced_token_ids = []


    def __call__(self, input_ids, scores):
        scores_np = np.array(scores, dtype=np.float32)
        scores_exp = np.exp(scores_np - np.max(scores_np)) # Softmax
        probs = scores_exp / np.sum(scores_exp)

        is_prefix_step = self.current_step_idx < self.num_prefix_steps
        target_token_id = -1

        if is_prefix_step:
            target_token_id = self.fixed_prefix_token_ids[self.current_step_idx]
            self.prefix_forced_token_ids.append(target_token_id)

        else: # suffix step
            chosen_token_id = np.argmax(probs)
            chosen_token_prob = probs[chosen_token_id]
            token_str = self.llm.tokenizer().decode([int(chosen_token_id)])
            
            self.generated_suffix_details.append({
                'token_str': token_str,
                'prob': float(chosen_token_prob)
            })
            target_token_id = chosen_token_id
            self.suffix_forced_token_ids.append(target_token_id)

        # force target_token_id
        modified_scores = np.full_like(scores_np, -float('inf'))
        modified_scores[target_token_id] = 0.0

        self.current_step_idx += 1
        return modified_scores

    def get_results(self):
        return {
            'suffix_text': "".join([d['token_str'] for d in self.generated_suffix_details]).strip(),
            'suffix_token_details': self.generated_suffix_details,
        }


# --- Classifier functions ---
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


# --- Main Execution ---
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
        print(f"VLM loaded. Prefix string '{PREFIX_STRING}' tokenized to: {prefix_tokens}")
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

    print(f"\n===== Processing Image: {SINGLE_IMAGE_PATH} =====")
    base64_image = image_to_base64(SINGLE_IMAGE_PATH)
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
        verbose=True
    )
    logits_processors = LogitsProcessorList([vlm_processor])
    
    print("\n--- VLM Generation Started ---")
    completion = llm.create_chat_completion(
        messages=messages,
        max_tokens=len(prefix_tokens)+MAX_SUFFIX_TOKENS_TO_GENERATE,
        temperature=VLM_TEMPERATURE,
        logits_processor=logits_processors,
        stop=["<0x0A>", "\n", "</s>", ".", ",", ";"]
    )
    vlm_results = vlm_processor.get_results()

    generated_suffix_text = vlm_results['suffix_text']
    suffix_token_details = vlm_results['suffix_token_details']
    
    print("\n--- VLM Output ---")
    full_vlm_response = PREFIX_STRING + " \033[1m" + generated_suffix_text + "\033[0m"
    print(f"Full Response: {full_vlm_response}")
    
    print("\nSuffix Token Probabilities:")
    for detail in suffix_token_details:
        print(f"  Token: {repr(detail['token_str'])}, Probability: {format_prob_pct(detail['prob'])}")

    print("\n--- Classifier Prediction ---")
    suffix_probabilities = [detail['prob'] for detail in suffix_token_details]
    features_series = extract_features_for_classifier(suffix_probabilities)
    features_for_scaling = pd.DataFrame([features_series.values], columns=BASE_FEATURE_COLUMNS)
    # features_for_scaling = features_series.to_numpy().reshape(1, -1)
    
    scaled_features = scaler.transform(features_for_scaling)
    
    prob_class1 = get_classifier_prob_class1(classifier_model, scaled_features)[0]
    
    pred_opt = "\033[32mLikely Correct\033[0m" if prob_class1 >= optimal_threshold else "\033[31mLikely Incorrect\033[0m"
    pred_0fp = "\033[32mLikely Correct\033[0m" if prob_class1 >= zero_fp_threshold else "\033[31mLikely Incorrect\033[0m"

    print(f"Classifier Confidence (in 'Correctness'): {format_prob_pct(prob_class1, 2)}")
    print(f"Predicted Status (Optimal accuracy training)  : {pred_opt} (Threshold: {optimal_threshold:.4f})")
    print(f"Predicted Status (Zero FalsePositive training): {pred_0fp} (Threshold: {zero_fp_threshold:.4f})")