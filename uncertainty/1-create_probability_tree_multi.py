from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList
from llama_cpp.llama_chat_format import Llava16ChatHandler
import base64
import os
import json
from PIL import Image
import math
import numpy as np
import heapq
from typing import Dict, List
import itertools
import graphviz
import colorsys
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
import csv
from PIL import Image, ImageDraw, ImageFont
import glob
import collections
from load_ollama_models import get_ollama_blob_path

# Configuration
PROMPT_TEXT = "You are a robot. Based on what you see, you must identify the object gripped in your robotic arm. Disregard the arm and background clutter, focus on the object. Be concise, identify the object."
NUM_PATHS_TO_FIND = 200
RANKS_K = 200
VERBOSE_STEP_BY_STEP = False

MAX_TOKENS_TO_GENERATE = 20
N_GPU_LAYERS = -1
N_CTX = 2048

MODEL_CONFIGS = [
    {
        "name": f"LlaVA 7B",
        "model_path": get_ollama_blob_path("llava", "model"),
        "mmproj_path": get_ollama_blob_path("llava", "projector"),
    },
    {
        "name": f"LlaVA 13B",
        "model_path": get_ollama_blob_path("llava:13b", "model"),
        "mmproj_path": get_ollama_blob_path("llava:13b", "projector"),
    },
    # {
    #     "name": f"LlaVA 34B",
    #     "model_path": get_ollama_blob_path("llava:34b", "model"),
    #     "mmproj_path": get_ollama_blob_path("llava:34b", "projector"),
    # },
]

IMAGE_FOLDER = "./dataset_vlm_0328_chosen"
IMAGE_PATHS = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")))

TARGET_WORDS_LIST = (
    # [["mug", "cup"]] * 4 +
    # [["glove", "gloves"]] * 4 +
    # [["sponge"]] * 4 +
    # [["plate", "dish", "tray"]] * 4 +
    # [["fan"]] * 4 +
    # [["soap", "hand sanitizer", "cleaner", "cleaning solution"]] * 4 +
    # [["cloth", "rag", "napkin"]] * 4 +
    # [["controller"]] * 4 +
    # [["screwdriver", "screw driver"]] * 4 +
    # [["circuit board", "computer board", "electronic board", "electronics board", "microcontroller", "pcb"]] * 4 +
    # [["mouse"]] * 4 +
    [["cleaning foam", "cleaning solution", "cleaning product", "whiteboard cleaner", "whiteboard foam"]] * 4
)
if len(IMAGE_PATHS) != len(TARGET_WORDS_LIST):
    print(f"FATAL ERROR: Number of images ({len(IMAGE_PATHS)}) does not match number of target words ({len(TARGET_WORDS_LIST)}).")
    exit(1)

if not IMAGE_PATHS:
    print(f"FATAL ERROR: No images found in folder '{IMAGE_FOLDER}'")
    exit(1)



OUTPUT_BASE_DIR = f"multi_run_n{NUM_PATHS_TO_FIND}_k{RANKS_K}_{len(TARGET_WORDS_LIST)}imgs"
SUMMARY_CSV_FILENAME = os.path.join(OUTPUT_BASE_DIR, "summary_results.csv")
SUMMARY_TABLE_FILENAME = os.path.join(OUTPUT_BASE_DIR, "summary_table.png")

EOS_TOKENS = [2]


candidate_counter = itertools.count() # Tie-breaker for Heap

def format_prob_pct(prob, numbers=3):
    if prob > 0: return f"{prob * 100:.{numbers}f}%"
    else: return f"{0:.{numbers}f}%"

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    with Image.open(image_path) as img:
        if img.mode != 'RGB': img = img.convert('RGB')
    return encoded_string


# Graphviz Functions
def seq_to_node_id(seq):
    if not seq: return "ROOT"
    return "S_" + "_".join(map(str, seq))

def calculate_penwidth(probability, min_width=0.5, max_width=5.0):
    width = min_width + (max_width - min_width) * probability
    return f"{width:.2f}"

def get_root_color(target_prob_estimate):
    prob = max(0.0, min(1.0, target_prob_estimate)); hue = prob * 120.0 / 360.0
    saturation = 0.9; lightness = 0.6; rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    return hex_color

def add_graph_step(graph, parent_sequence, current_sequence, step_data: Dict, graph_elements_added):
    parent_id = seq_to_node_id(parent_sequence); current_id = seq_to_node_id(current_sequence); edge_key = f"edge_{parent_id}_{current_id}"
    token = step_data.get('token', '?'); rank = step_data.get('rank_chosen', '?'); logprob = step_data.get('logprob', -math.inf) # Use the *actual* logprob passed
    prob = math.exp(logprob) if logprob > -math.inf else 0.0; penwidth = calculate_penwidth(prob)
    node_label = f"R{rank}\n{repr(token)}"
    if current_id not in graph_elements_added: graph.node(current_id, label=node_label, shape='box'); graph_elements_added.add(current_id)
    if edge_key not in graph_elements_added: graph.edge(parent_id, current_id, label=f"{format_prob_pct(prob)}", penwidth=penwidth); graph_elements_added.add(edge_key)

def add_pruned_node(graph, parent_sequence, pruned_sequence, pruned_token, probability, termination_type, graph_elements_added):
    parent_id = seq_to_node_id(parent_sequence); pruned_id = seq_to_node_id(pruned_sequence); edge_key = f"edge_{parent_id}_{pruned_id}"
    rank = pruned_sequence[-1]; penwidth = calculate_penwidth(probability); node_shape = 'ellipse'; node_style = 'filled'
    node_label = f"R{rank}\n{repr(pruned_token)}"
    if termination_type == 'EOS': node_color = 'lightcoral'
    elif termination_type == 'TARGET': node_color = 'lightgreen'
    else: node_color = 'grey'
    if pruned_id not in graph_elements_added: graph.node(pruned_id, label=node_label, style=node_style, fillcolor=node_color, shape=node_shape); graph_elements_added.add(pruned_id)
    if edge_key not in graph_elements_added: graph.edge(parent_id, pruned_id, label=f"{format_prob_pct(probability)}", penwidth=penwidth); graph_elements_added.add(edge_key)


# Combined Logits Processor
class AnalyzeAndForceProcessor(LogitsProcessor):
    """
    Stores top K candidates, applies deviations, and calculates
    the cumulative log probability of the path taken.
    """
    def __init__(self, llm_instance, deviations, k_store, verbose_step=False):
        self.llm = llm_instance # We use its tokenizer
        self.deviations = deviations
        self.k = k_store
        self.verbose_step = verbose_step
        self.current_step = 0
        self.cumulative_logprob = 0.0
        self.step_data = {} # {step_index: [{'id':int, 'token':str, 'logprob':float}, ...]}
        self.path_details = {} # minimal info for path reconstruction {step: {'token':str, 'logprob':float, 'rank':int, 'id':int}}

    def __call__(self, input_ids, scores):
        step_index = self.current_step

        # We get the original probability distribution of the next possible tokens according to the LLM
        # Then record the top k most probable original candidate tokens & their probabilities (later used to find alternative paths)
        k = self.k
        scores_exp = np.exp(scores - np.max(scores))
        probs = scores_exp / np.sum(scores_exp) # softmax
        top_k_indices = np.argsort(probs)[::-1][:k]
        top_k_info_for_step = []
        logprob_lookup = {}
        for i in range(k):
            token_id = top_k_indices[i]
            probability = probs[token_id]
            logprob = math.log(probability)
            top_k_info_for_step.append({'id': int(token_id), 'token': self.llm.tokenizer().decode([int(token_id)]), 'logprob': logprob})
            logprob_lookup[int(token_id)] = logprob # original logprob

        self.step_data[step_index] = top_k_info_for_step # based on unmodified scores, for analysis later

        # We modify the token probabilties (logits) to force the LLM to choose a specific path
        modified_scores = np.copy(scores)
        forced_rank_this_step = self.deviations.get(step_index, 1) # get forced rank if there is one, else default rank (1)
        if forced_rank_this_step > 1:
            indices_to_suppress = np.argsort(scores)[::-1][:forced_rank_this_step - 1]
            if indices_to_suppress.size > 0:
                modified_scores[indices_to_suppress] = -float('inf') # paths already visited are removed from consideration

        # Track which token was chosen at each step and its original probability/rank
        # Then calculate cumulative probability using original probabilities
        chosen_token_id = int(np.argmax(modified_scores)) # token the LLM will actually generate for this step
        # Use the ORIGINAL log probability of the chosen token to get the correct path probability
        chosen_token_logprob = logprob_lookup.get(chosen_token_id, -math.inf)
        self.cumulative_logprob += chosen_token_logprob

        # Determine actual rank based on original probabilities
        actual_rank_taken = -1
        for r, cand_data in enumerate(top_k_info_for_step):
            if cand_data['id'] == chosen_token_id:
                actual_rank_taken = r + 1
                break
        if actual_rank_taken == -1 and chosen_token_id >= 0:
            actual_rank_taken = k + 1 # Indicate it was outside stored K

        chosen_token_str = self.llm.tokenizer().decode([chosen_token_id])

        self.path_details[step_index] = {
            'token': chosen_token_str,
            'logprob': chosen_token_logprob,
            'rank_chosen': actual_rank_taken if actual_rank_taken != -1 else '?',
            'token_id': chosen_token_id
        }

        if self.verbose_step:
            print(f"    Step {step_index:<2}: Rank={actual_rank_taken if actual_rank_taken!=-1 else '?'}{' (Forced ' + str(forced_rank_this_step) + ')' if forced_rank_this_step>1 else ''} "
                f"CHOSEN -> ID={chosen_token_id:<5} Tok={repr(chosen_token_str):<15} P={format_prob_pct(math.exp(chosen_token_logprob)):<9} CumP={format_prob_pct(math.exp(self.cumulative_logprob))} (CumLogP={self.cumulative_logprob:.4f})")
            print(f"           Top {len(top_k_info_for_step)} raw candidates considered (Before Force Processor):")
            for rank, candidate_data in enumerate(top_k_info_for_step):
                cand_id = candidate_data['id']; cand_token = candidate_data['token']; cand_logprob = candidate_data['logprob']
                cand_prob = math.exp(cand_logprob) if cand_logprob > -math.inf else 0.0; highlight = ""
                if cand_id == chosen_token_id: highlight = " ***** CHOSEN *****"
                elif rank + 1 < forced_rank_this_step and step_index in self.deviations: highlight = f" (Suppressed Rank {rank+1})"
                print(f"             {rank+1:<2}. ID={cand_id:<5} Tok={repr(cand_token):<15} P={format_prob_pct(cand_prob):<9}{highlight}")
            print("-" * 20)

        self.current_step += 1
        return modified_scores # modified scores for llama.cpp to select from


    def get_final_logprob(self) -> float:
        return self.cumulative_logprob

    def get_step_data(self) -> Dict[int, List[Dict]]:
        """Returns the stored top K data for all steps."""
        return self.step_data

    def get_path_details(self) -> Dict[int, Dict]:
        """Returns the details of the path actually taken."""
        return self.path_details


# --- Run a Single Generation Path (Using AnalyzeAndForce Processor) ---
def run_generation_path(llm, messages, max_tokens, path_name, force_rank_sequence=None, target_words=[], verbose_step_by_step=True) -> Dict:
    if force_rank_sequence is None: force_rank_sequence = tuple()
    deviations_dict = {i: rank for i, rank in enumerate(force_rank_sequence) if rank > 1}

    print(f"\n--- Starting Generation Path: {path_name} ", end=" ")
    print(f"(Forcing Ranks: {str(force_rank_sequence) if force_rank_sequence else 'Baseline'}, then greedy) ---")

    # Processor
    processor = AnalyzeAndForceProcessor(llm_instance=llm, deviations=deviations_dict, k_store=RANKS_K, verbose_step=verbose_step_by_step)
    logits_processors = LogitsProcessorList([processor])

    # Initialization
    generated_text = ""
    token_count = 0
    found_target = False
    targets_lower = [w.lower() for w in target_words] if target_words else []
    found_word = ""
    target_found_index = -1
    logprob_at_target_found = -math.inf

    # Generation (non streaming)
    _ = llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.0,
        logprobs=False,
        top_logprobs=0,
        stream=False,
        logits_processor=logits_processors
    )

    # Processing output
    # (reconstruct text, token count, check for target, find logprob_at_target from the processor's detailed step-by-step record)
    path_details_from_processor = processor.get_path_details()
    temp_text_list = []
    cumulative_logprob_check = 0.0

    for i in sorted(path_details_from_processor.keys()): # for each step
        step_info = path_details_from_processor[i]
        temp_text_list.append(step_info['token'])
        token_count = i + 1

        step_logprob = step_info.get('logprob', -math.inf)
        cumulative_logprob_check += step_logprob

        # Check if target found after this step
        generated_text = "".join(temp_text_list)
        if targets_lower and not found_target and any(target in generated_text.lower() for target in targets_lower):
            if any(generated_text.lower().strip().endswith(target) for target in targets_lower):
                found_target = True
                found_word = next(target for target in targets_lower if generated_text.lower().strip().endswith(target))
                target_found_index = i
                logprob_at_target_found = cumulative_logprob_check
                break

    if found_target:
        token_count = target_found_index + 1

    total_logprob = processor.get_final_logprob()
    if found_target:
        total_logprob = logprob_at_target_found


    # Reconstruct the generated rank sequence from path details
    generated_rank_sequence_list = [path_details_from_processor[i]['rank_chosen'] for i in sorted(path_details_from_processor.keys())[:token_count]] # Use token_count limit
    final_generated_rank_sequence = tuple(generated_rank_sequence_list)

    # Get the stored raw top-K data
    stored_raw_data = processor.get_step_data()

    # --- Calculate final metrics ---
    avg_logprob = total_logprob / token_count if token_count > 0 else -math.inf
    perplexity = math.exp(-avg_logprob) if avg_logprob > -math.inf else math.inf

    return {
        'text': generated_text, # (can be shortened)
        'token_count': token_count,
        'details': path_details_from_processor,
        'raw_top_k_data': stored_raw_data,
        'total_logprob': total_logprob,
        'found_target': found_target,
        'target_found_index': target_found_index,
        'logprob_at_target_found': logprob_at_target_found,
        'forced_rank_sequence': force_rank_sequence,
        'generated_rank_sequence': final_generated_rank_sequence,
        'name': path_name,
        'target_word_used': found_word
    }


# Print path results summary
def print_path_results_summary(path_name, results, target_words, target_word_used):
    print(f"\n--- {path_name} ---")
    total_logprob = results['total_logprob']
    total_prob = math.exp(total_logprob) if total_logprob > -math.inf else 0.0
    logprob_at_target = results.get('logprob_at_target_found', -math.inf)
    prob_at_target = math.exp(logprob_at_target) if logprob_at_target > -math.inf else 0.0
    found_target = results.get('found_target', False); target_index = results.get('target_found_index', -1)
    print(f"Generated Text ({results['token_count']} tokens):\n\033[1m{repr(results['text'])}\033[0m")
    if found_target:
        print(f"->\033[32m Target '{target_word_used}' FOUND ending at step {target_index}.\033[0m")
        print(f"   Full Path Probability: {format_prob_pct(prob_at_target)}")
    else:
        print(f"->\033[31m Targets '{target_words}' NOT FOUND in this path.\033[0m")
        print(f"   Full Path Probability: {format_prob_pct(total_prob)}")


# Analyze path (Using rank Sequence & Raw Data)
def analyze_path_and_update_candidates(path_results, candidate_heap, visited_or_pruned_prefixes, graph, graph_elements_added, global_cumulative_explored, global_target_mass, target_words, verbose, eos_tokens):
    path_name = path_results.get('name', 'Unknown')
    if verbose: print(f"\n--- Analyzing Path '{path_name}' ---")
    raw_top_k_data = path_results.get('raw_top_k_data', {}) # data stored by processor
    generated_rank_sequence = path_results.get('generated_rank_sequence', tuple())
    target_found_index = path_results.get('target_found_index', -1)
    deviation_limit_index = target_found_index + 1 if target_found_index != -1 else len(generated_rank_sequence) # loop doesnt include limit so +1

    cumulative_logprob_before_step = 0.0
    text_prefix = ""
    new_candidates_added = 0
    pruned_target_prob_local = 0.0
    pruned_eos_prob_local = 0.0
    skipped_visited_count = 0

    for i in range(deviation_limit_index): # for each step
        top_k_raw = raw_top_k_data.get(i, [])

        step_details = path_results['details'].get(i)
        step_logprob = step_details.get('logprob', -math.inf)
        step_token = step_details.get('token', '')

        current_path_prefix = generated_rank_sequence[:i]

        for rank_idx in range(1, min(RANKS_K, len(top_k_raw))): # for each rank
            rank_to_force = rank_idx + 1
            if rank_to_force == generated_rank_sequence[i]: continue

            deviation_data = top_k_raw[rank_idx]
            dev_token_id = deviation_data.get('id', -1)
            dev_token = deviation_data.get('token', '')
            dev_logprob = deviation_data.get('logprob', -math.inf)

            logprob_dev_branch_prefix = cumulative_logprob_before_step + dev_logprob
            prob_dev_branch = math.exp(logprob_dev_branch_prefix)
            branch_rank_prefix = current_path_prefix + (rank_to_force,)

            if branch_rank_prefix in visited_or_pruned_prefixes:
                skipped_visited_count += 1
                continue
            visited_or_pruned_prefixes.add(branch_rank_prefix)

            is_eos = (dev_token_id in eos_tokens)
            current_text_if_deviated = text_prefix + dev_token
            is_target = any(current_text_if_deviated.lower().strip().endswith(target) for target in target_words)

            if is_eos or is_target: # check EOS or target
                termination_type = 'EOS' if is_eos else 'TARGET'
                if verbose:
                    token_print = f"\033[31m{repr(dev_token)}\033[0m" if is_eos else f"\033[32m{repr(dev_token)}\033[0m"
                    print(f"    -> Pruning Prefix: Ranks={branch_rank_prefix}, ID={dev_token_id}, Token={token_print}, P={format_prob_pct(prob_dev_branch)} ({termination_type})")
                global_cumulative_explored += prob_dev_branch
                if is_target: global_target_mass += prob_dev_branch; pruned_target_prob_local += prob_dev_branch
                else: pruned_eos_prob_local += prob_dev_branch
                add_pruned_node(graph, current_path_prefix, branch_rank_prefix, dev_token, prob_dev_branch, termination_type, graph_elements_added)
            else:
                count = next(candidate_counter)
                heapq.heappush(candidate_heap, (-logprob_dev_branch_prefix, count, branch_rank_prefix))
                new_candidates_added += 1

        # Update cumulative logprob based on the token actually chosen
        cumulative_logprob_before_step += step_logprob
        text_prefix += step_token

    if verbose:
        print(f"--- Analysis Complete for Path '{path_name}' ---")
        if skipped_visited_count > 0: print(f"    Skipped {skipped_visited_count} already visited branch prefixes.")
        if pruned_target_prob_local > 0 or pruned_eos_prob_local > 0: print(f"    Pruned NEW Branches Total: Target={format_prob_pct(pruned_target_prob_local)}, EOS={format_prob_pct(pruned_eos_prob_local)}")
        print(f"    Added {new_candidates_added} new candidate prefixes to heap (Currently {len(candidate_heap)} candidates).")
    print(f"Target Found / Total Explored:  \033[1m{format_prob_pct(global_target_mass)} / {format_prob_pct(global_cumulative_explored)}\033[0m")
    return global_cumulative_explored, global_target_mass


def render_current_graph(graph, path_count, target_mass, explored_mass, target_words, run_output_dir):
    """Updates graph label/root color and renders the graph (first/last only)."""

    should_save = (path_count in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200, 300, 500])

    target_prob_estimate = target_mass / explored_mass if explored_mass > 0 else 0.0
    root_color = get_root_color(target_prob_estimate)
    prob_mouse_str = format_prob_pct(target_mass)
    prob_explored_str = format_prob_pct(explored_mass)
    estimate_str = format_prob_pct(target_prob_estimate)
    graph_label = ( f"Exploration State after Path {path_count}\n" f"P({target_words}) Mass: {prob_mouse_str} | " f"Explored Mass: {prob_explored_str}" )
    root_label_node = f"P({target_words})\n{estimate_str}"
    graph.node('ROOT', label=root_label_node, style='filled', fillcolor=root_color, shape='doubleoctagon', width='1.2', height='0.8', fixedsize='true')
    graph.attr('graph', label=graph_label, labelloc='t', fontsize='12')

    if should_save:
        print(f"\n--- Rendering graph ---")
        output_filename = os.path.join(run_output_dir, f'exploration_tree_path_{path_count}')
        try:
            graph.render(output_filename, format='png', cleanup=True, view=False)
        except Exception as gv_e:
            print(f"\n!!! Graphviz rendering failed (is it installed and in PATH? https://graphviz.org/download/ ): {gv_e} !!!")


def plot_probabilities(history, filename, target_words):

    # Extract data
    path_counts = [entry['path_count'] for entry in history]
    explored_probs = [entry['explored_prob'] for entry in history]
    target_probs = [entry['target_prob'] for entry in history]
    estimate_probs = [entry['estimate_prob'] for entry in history]

    # setup
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_explored = 'tab:blue'
    ax1.set_xlabel('Paths Generated/Analyzed')
    ax1.set_ylabel('Cumulative Probability Mass (%)', color=color_explored)

    # cumulative probabilities
    line1, = ax1.plot(path_counts, np.array(explored_probs) * 100, color=color_explored, marker='.', linestyle='-', label='Total Explored Prob.')
    line2, = ax1.plot(path_counts, np.array(target_probs) * 100, color='tab:green', marker='.', linestyle='-', label=f'Prob. Mass leading to "{target_words}"')
    ax1.tick_params(axis='y', labelcolor=color_explored)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.grid(True, axis='y', linestyle=':')
    ax1.set_ylim(0, 105)

    # estimated probability
    ax2 = ax1.twinx()
    color_estimate = 'tab:red'
    ax2.set_ylabel(f'Estimated P("{target_words}") (%)', color=color_estimate)
    line3, = ax2.plot(path_counts, np.array(estimate_probs) * 100, color=color_estimate, marker='x', linestyle='--', label=f'Est. P("{target_words}")')
    ax2.tick_params(axis='y', labelcolor=color_estimate)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # legend/title
    lines = [line1, line2, line3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
    plt.title('Evolution of Explored Probability and Target Estimate')

    # save
    fig.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close(fig)


def run_single_exploration(llm_instance, model_config, image_path, prompt_text, target_words, config, run_output_dir):
    """
    Performs the full probability exploration for one model/image combination.
    Saves graphviz, plot, and returns summary statistics.
    """
    run_start_time = time.time()
    model_name = model_config['name']
    image_name = os.path.basename(image_path)
    print("\n" + "="*80); print(f"   Running Exploration for: {model_name} : {image_name}"); print("="*80)

    llm = llm_instance

    os.makedirs(run_output_dir, exist_ok=True)
    
    # Input preparation
    base64_image = image_to_base64(image_path)
    messages = [ { "role": "user", "content": [ {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}, {"type": "text", "text": prompt_text} ] } ]

    # Initialization
    graph = graphviz.Digraph(f'ExplorationTree_{model_name}_{image_name}', graph_attr={'rankdir': 'TB'})
    graph_elements_added = set()
    root_id = seq_to_node_id(tuple())
    graph.node(root_id, label='Calculating...', shape='doubleoctagon', style='filled', fillcolor='grey')
    graph_elements_added.add(root_id)
    all_paths_info, candidate_prefixes_heap = [], []
    visited_or_pruned_prefixes = set()
    visited_or_pruned_prefixes.add(tuple())
    cumulative_probability_explored = 0.0
    probability_mass_leading_to_target = 0.0
    history = []

    llm.reset() # reset each run

    heapq.heappush(candidate_prefixes_heap, (0.0, next(candidate_counter), tuple())) # seed heap (root node)

    # Main loop
    paths_generated_count = 0
    while paths_generated_count < config['NUM_PATHS_TO_FIND'] and candidate_prefixes_heap:
        neg_logprob_prefix, count, rank_sequence_prefix = heapq.heappop(candidate_prefixes_heap)
        paths_generated_count += 1

        if not rank_sequence_prefix:
            path_name = f"Path {paths_generated_count}: Baseline"
        else:
            path_name = f"Path {paths_generated_count}: Seq{'_'.join([f'S{s}R{r}' for s, r in enumerate(rank_sequence_prefix)])}"

        # Generate path
        if config['VERBOSE_STEP_BY_STEP']: print("\n" + "-"*60); print(f" " * 10 + f"Starting Generation for {path_name}"); print("-"*60)
        generated_path_results = run_generation_path( llm, messages, config['MAX_TOKENS_TO_GENERATE'], path_name=path_name, force_rank_sequence=rank_sequence_prefix, target_words=target_words, verbose_step_by_step=config['VERBOSE_STEP_BY_STEP'] )
        current_path_info = { "name": path_name, "results": generated_path_results, "rank_sequence": generated_path_results['generated_rank_sequence'] }

        # Add to visited/graph
        full_gen_seq = generated_path_results['generated_rank_sequence']
        visited_or_pruned_prefixes.add(full_gen_seq)
        all_paths_info.append(current_path_info)
        parent_seq = tuple()
        for i, rank in enumerate(full_gen_seq):
            current_seq = full_gen_seq[:i+1]; parent_id = seq_to_node_id(parent_seq)
            if parent_id not in graph_elements_added and parent_id != "ROOT": print(f"!!! Warning: Parent node {parent_id} not found for path {path_name}, step {i}. Graph might be broken. !!!")
            step_data = generated_path_results['details'].get(i)
            if step_data: add_graph_step(graph, parent_seq, current_seq, step_data, graph_elements_added)
            parent_seq = current_seq
        if full_gen_seq: final_node_id = seq_to_node_id(full_gen_seq)
        if final_node_id in graph_elements_added: gen_path_color = 'lightgreen' if generated_path_results['found_target'] else 'lightcoral'; graph.node(final_node_id, style='filled', fillcolor=gen_path_color)

        # Accumulate probabilities
        current_found_target = generated_path_results['found_target']
        current_final_prob = math.exp(generated_path_results['logprob_at_target_found']) if current_found_target else math.exp(generated_path_results['total_logprob'])
        current_target_mass = math.exp(generated_path_results['logprob_at_target_found']) if current_found_target else 0.0
        cumulative_probability_explored += current_final_prob
        probability_mass_leading_to_target += current_target_mass

        # Reporting
        print_path_results_summary(path_name, generated_path_results, target_words, generated_path_results['target_word_used'])
        # print(f"Target Found / Total Explored:  {format_prob_pct(probability_mass_leading_to_target)} / {format_prob_pct(cumulative_probability_explored)}")

        # Analyze
        cumulative_probability_explored, probability_mass_leading_to_target = analyze_path_and_update_candidates(generated_path_results, candidate_prefixes_heap, visited_or_pruned_prefixes, graph, graph_elements_added, cumulative_probability_explored, probability_mass_leading_to_target, target_words, verbose=config['VERBOSE_STEP_BY_STEP'], eos_tokens=config['EOS_TOKENS'])

        # Record history
        current_estimate = probability_mass_leading_to_target / cumulative_probability_explored if cumulative_probability_explored > 0 else 0.0
        history.append({
            'path_count': paths_generated_count,
            'explored_prob': cumulative_probability_explored,
            'target_prob': probability_mass_leading_to_target,
            'estimate_prob': current_estimate,
        })

        # Graph
        render_current_graph(graph, paths_generated_count, probability_mass_leading_to_target, cumulative_probability_explored, target_words, run_output_dir)

    # Final summary for this run
    print("\n" + "="*70); print(f" " * 15 + f"FINAL SUMMARY for {model_name} / {image_name}"); print("="*70)
    print(f"Generated {len(all_paths_info)} paths")
    print(f"Probability mass {target_words} / Total Explored: {format_prob_pct(probability_mass_leading_to_target)} / {format_prob_pct(cumulative_probability_explored)}")
    final_estimate = probability_mass_leading_to_target / cumulative_probability_explored if cumulative_probability_explored > 0 else 0.0
    print(f"--> Estimated P({target_words}): \033[1m{final_estimate * 100:.3f}%\033[0m")

    # Plot
    run_plot_filename = f"prob_evo_{model_name}_{image_name}.png"
    plot_probabilities(history, os.path.join(run_output_dir, run_plot_filename), target_words)

    # saving texts
    generated_texts_data = []
    for path_info in all_paths_info:
        results = path_info.get('results', {})
        text = results.get('text', '<Error: No text generated>')
        success = results.get('found_target', False)
        logprob = results.get('logprob_at_target_found', -math.inf) if success else results.get('total_logprob', -math.inf)
        prob = math.exp(logprob) if logprob > -math.inf else 0.0
        generated_texts_data.append({'text': text, 'success': success, 'prob': prob})

    # Sort: success (True) first, then by probability descending
    generated_texts_data.sort(key=lambda x: (x['success'], x['prob']), reverse=True)

    output_txt_filename = os.path.join(run_output_dir, f"generated_texts_{model_name}_{image_name}.txt")
    try:
        with open(output_txt_filename, 'w', encoding='utf-8') as f:
            for item in generated_texts_data:
                status_marker = "[SUCCESS]" if item['success'] else "[FAIL]"
                prob_marker = f"P={item['prob']:.6f}"
                f.write(f"{status_marker} {prob_marker} | {item['text']}\n")
        print(f"Saved {len(generated_texts_data)} generated texts to {output_txt_filename}")
    except Exception as txt_e:
        print(f"!!! Error saving generated texts to {output_txt_filename}: {txt_e} !!!")

    run_end_time = time.time()
    print(f"\nRun for {model_name}/{image_name} finished in {run_end_time - run_start_time:.2f} seconds.")

    return {
        "model_name": model_name,
        "image_path": image_path,
        "image_name": image_name,
        "target_words": target_words,
        "final_explored_prob": cumulative_probability_explored,
        "final_target_prob": probability_mass_leading_to_target,
        "final_estimate_prob": final_estimate,
        "paths_generated": len(all_paths_info),
        "runtime": run_end_time - run_start_time
    }


def create_summary_table_from_csv(csv_filepath, prompt_text, output_filename):
    """Creates the summary table image by reading data from the results CSV."""
    if not os.path.exists(csv_filepath):
        print(f"!!! ERROR: CSV file not found at {csv_filepath}. Cannot generate table. !!!")
        return

    # Read data from CSV
    results_by_image = collections.defaultdict(dict) # {image_path: {model_name: estimate_prob}}
    target_words_map = {} # {image_path: target_word}
    unique_images = set()
    max_target_word_len = 0
    unique_model_names = []

    with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            print(f"!!! ERROR: CSV file {csv_filepath} is empty or has no header. !!!")
            return

        for row in reader:
            # Extract data from the current row dictionary.
            image_path = row['image_path']
            model_name = row['model_name']
            target_words = row['target_words']

            max_target_word_len = max(max_target_word_len, len(target_words))

            estimate = float(row['final_estimate_prob'])
            explored = float(row['final_explored_prob'])
            runtime = float(row['runtime'])

            results_by_image[image_path][model_name] = {
                'estimate': estimate,
                'explored': explored,
                'runtime': runtime
            }

            if image_path not in target_words_map:
                target_words_map[image_path] = target_words
            unique_images.add(image_path)
    
            if model_name not in unique_model_names:
                unique_model_names.append(model_name)

    found_models_list = unique_model_names

    # Drawing setup
    multiplier = 2 # This will affect the image resolution
    target_word_col_width = max(200, 2+(max_target_word_len * 15)) * multiplier
    unique_images_list = sorted(list(unique_images))
    padding = 0
    prompt_area_height = 140 * multiplier
    row_height = 200 * multiplier
    header_height = int(row_height // 1.5)
    thumb_size = (int(row_height * 1.333), row_height)
    col_widths = [thumb_size[0], 200 * multiplier, target_word_col_width] # Thumb, Name, Target Word
    model_col_width = 300 * multiplier
    col_widths.extend([model_col_width] * len(found_models_list) * 3)
    table_width = sum(col_widths) + padding
    table_height = prompt_area_height + header_height + (len(unique_images_list) * row_height) + padding + 1

    try:
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if not os.path.exists(font_path): font_path="arial.ttf"
            font_prompt = ImageFont.truetype(font_path.replace("Sans","Sans-Bold"), 26 * multiplier)
            font_header = ImageFont.truetype(font_path.replace("Sans","Sans-Bold"), 30 * multiplier)
            font = ImageFont.truetype(font_path, 35 * multiplier)
        except IOError:
            print("Warning: Default font not found. Using PIL default.")
            font = ImageFont.load_default(); font_header = font; font_prompt = font
    except Exception as font_e:
        print(f"Error loading font: {font_e}. Using default.")
        font = ImageFont.load_default(); font_header = font; font_prompt = font

    img = Image.new('RGB', (table_width, table_height), color = 'white')
    draw = ImageDraw.Draw(img)
    prompt_wrapped = "\n".join(prompt_text[i:i+130] for i in range(0, len(prompt_text), 130))
    draw.text((padding, padding), f"Prompt:\n{prompt_wrapped}", fill='black', font=font_prompt)

    # Table headers
    current_x = padding; current_y = prompt_area_height
    headers = ["Image", "Image Name", "Target Word"]
    for model_name_short in found_models_list:
        headers.append(f"\n\nSuccess")
        headers.append(f"---     {model_name_short}     ---\n\nExplored")
        headers.append(f"\n\nTime")
    for i, header in enumerate(headers):
        draw.rectangle([current_x, current_y, current_x + col_widths[i], current_y + header_height], fill='#E0E0E0')
        text_bbox = draw.textbbox((0,0), header, font=font_header)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = current_x + (col_widths[i] - text_width) // 2
        text_y = current_y + (header_height - text_height) // 2
        draw.text((text_x, text_y), header, fill='black', font=font_header, align='center')
        current_x += col_widths[i]

    # Table rows
    current_y += header_height
    for image_path in unique_images_list:
        current_x = padding
        image_name = os.path.basename(image_path)
        image_target = target_words_map.get(image_path, "N/A")[1:-1]

        # Thumbail
        try:
            thumb = Image.open(image_path)
            thumb.thumbnail(thumb_size)
            paste_x = current_x + (col_widths[0] - thumb.width) // 2
            paste_y = current_y + (row_height - thumb.height) // 2 + 1
            img.paste(thumb, (paste_x, paste_y))
        except Exception as thumb_e:
            draw.text((current_x + 5, current_y + 5), "No Img", fill='grey', font=font)
        current_x += col_widths[0]

        # Image Name
        draw.rectangle([current_x, current_y, current_x + col_widths[1], current_y + row_height], outline='grey'); draw.text((current_x + 5, current_y + (row_height - 16)//2 ), image_name, fill='black', font=font)
        current_x += col_widths[1]

        # Target Word
        draw.rectangle([current_x, current_y, current_x + col_widths[2], current_y + row_height], outline='grey'); draw.text((current_x + 5, current_y + (row_height - 16)//2 ), image_target, fill='black', font=font)
        current_x += col_widths[2]

        # Model Results
        for model_idx, model_name in enumerate(found_models_list):
            model_results_dict  = results_by_image[image_path].get(model_name)
            base_model_col_idx = 3 + model_idx * 3

            if model_results_dict:
                # Cell 1: Success Rate
                estimate = model_results_dict['estimate']
                cell_text_1 = format_prob_pct(estimate, 3)
                cell_color_1 = get_root_color(estimate)
                col_idx_1 = base_model_col_idx
                draw.rectangle([current_x, current_y, current_x + col_widths[col_idx_1], current_y + row_height], fill=cell_color_1, outline='grey')
                text_bbox = draw.textbbox((0,0), cell_text_1, font=font); tw=text_bbox[2]-text_bbox[0]; th=text_bbox[3]-text_bbox[1]
                tx = current_x + (col_widths[col_idx_1] - tw) // 2; ty = current_y + (row_height - th) // 2
                draw.text((tx, ty), cell_text_1, fill='black', font=font)
                current_x += col_widths[col_idx_1]

                # Cell 2: Explored %
                explored = model_results_dict['explored']
                cell_text_2 = format_prob_pct(explored, 3)
                cell_color_2 = 'white'
                col_idx_2 = base_model_col_idx + 1
                draw.rectangle([current_x, current_y, current_x + col_widths[col_idx_2], current_y + row_height], fill=cell_color_2, outline=get_root_color(explored), width=10*multiplier)
                text_bbox = draw.textbbox((0,0), cell_text_2, font=font); tw=text_bbox[2]-text_bbox[0]; th=text_bbox[3]-text_bbox[1]
                tx = current_x + (col_widths[col_idx_2] - tw) // 2; ty = current_y + (row_height - th) // 2
                draw.text((tx, ty), cell_text_2, fill='black', font=font)
                current_x += col_widths[col_idx_2]

                # Cell 3: Runtime
                runtime = model_results_dict['runtime']
                cell_text_3 = f"{runtime:.1f}s"
                cell_color_3 = 'white'
                col_idx_3 = base_model_col_idx + 2
                draw.rectangle([current_x, current_y, current_x + col_widths[col_idx_3], current_y + row_height], fill=cell_color_3, outline='grey')
                text_bbox = draw.textbbox((0,0), cell_text_3, font=font); tw=text_bbox[2]-text_bbox[0]; th=text_bbox[3]-text_bbox[1]
                tx = current_x + (col_widths[col_idx_3] - tw) // 2; ty = current_y + (row_height - th) // 2
                draw.text((tx, ty), cell_text_3, fill='black', font=font)
                current_x += col_widths[col_idx_3]

            else: # no data -> N/A in all 3 cells
                for i in range(3):
                    col_idx = base_model_col_idx + i
                    draw.rectangle([current_x, current_y, current_x + col_widths[col_idx], current_y + row_height], fill='#E0E0E0', outline='grey')
                    text_bbox = draw.textbbox((0,0), "N/A", font=font); tw=text_bbox[2]-text_bbox[0]; th=text_bbox[3]-text_bbox[1]
                    tx = current_x + (col_widths[col_idx] - tw) // 2; ty = current_y + (row_height - th) // 2
                    draw.text((tx, ty), "N/A", fill='black', font=font)
                    current_x += col_widths[col_idx]

        current_y += row_height

    # Save image
    img.save(output_filename)
    print(f"Saved summary table to {output_filename}")



def run_main():
    all_run_results = []

    # Exploration for each model and image
    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg['name']
        chat_handler = None
        llm = None
        chat_handler = Llava16ChatHandler(clip_model_path=model_cfg['mmproj_path'], verbose=False)
        llm = Llama(model_path=model_cfg['model_path'], chat_handler=chat_handler, n_ctx=N_CTX, n_gpu_layers=N_GPU_LAYERS, logits_all=True, verbose=False)
        print(f"Model {model_name} loaded successfully.")

        for img_path, current_target_words in zip(IMAGE_PATHS, TARGET_WORDS_LIST):
            image_name = os.path.basename(img_path)
            run_dir = os.path.join(OUTPUT_BASE_DIR, model_name, image_name.replace('.jpg',''))

            run_config = {
                'NUM_PATHS_TO_FIND': NUM_PATHS_TO_FIND,
                'RANKS_K': RANKS_K,
                'MAX_TOKENS_TO_GENERATE': MAX_TOKENS_TO_GENERATE,
                'N_CTX': N_CTX,
                'N_GPU_LAYERS': N_GPU_LAYERS,
                'VERBOSE_STEP_BY_STEP': VERBOSE_STEP_BY_STEP,
                'EOS_TOKENS': EOS_TOKENS,
            }

            summary = run_single_exploration(llm_instance=llm, model_config=model_cfg, image_path=img_path, prompt_text=PROMPT_TEXT, target_words=current_target_words, config=run_config, run_output_dir=run_dir)
            all_run_results.append(summary)
        
        del llm
        del chat_handler
        chat_handler = None
        llm = None

    # Save results to CSV
    try:
        fieldnames = list(all_run_results[0].keys())
        with open(SUMMARY_CSV_FILENAME, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_run_results) # assume all dicts have same keys
        print(f"Successfully saved {len(all_run_results)} results to CSV {SUMMARY_CSV_FILENAME}.")
    except Exception as csv_e:
        print(f"!!! Error writing CSV file ({SUMMARY_CSV_FILENAME}): {csv_e} !!!")




if __name__ == "__main__":
    # Create main output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Run everything (comment to generate just the summary table if csv available)
    run_main()

    # Generate final summary table image
    if os.path.exists(SUMMARY_CSV_FILENAME):
        create_summary_table_from_csv(SUMMARY_CSV_FILENAME, PROMPT_TEXT, SUMMARY_TABLE_FILENAME)
    else:
        print(f"Skipping summary table image generation: CSV file {SUMMARY_CSV_FILENAME} not found.")
    
    print("\nDone.\n")