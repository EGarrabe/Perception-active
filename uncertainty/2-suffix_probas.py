import os
import csv
import time
import base64
import math
import numpy as np
import heapq
import itertools
from llama_cpp import Llama, LogitsProcessor, LogitsProcessorList
from llama_cpp.llama_chat_format import Llava16ChatHandler
from load_ollama_models import get_ollama_blob_path
import collections
from PIL import Image, ImageDraw, ImageFont
import colorsys

# Configuration
PROMPT_TEXT = "You are a robot. Based on what you see, you must identify the object gripped in your robotic arm. Disregard the arm and background clutter, focus on the object. Be concise, identify the object."
PREFIX_STRING = "The object is a"

IMAGE_FOLDER_PATH = "dataset_vlm_0328_chosen"
OUTPUT_CSV_PATH = "prefix_probas.csv"

NUM_PATHS_TO_FIND = 3
RANKS_K = 3
MAX_SUFFIX_TOKENS = 10

N_GPU_LAYERS = -1
N_CTX = 2048
TOP_K_FOR_PREFIX_SEARCH = 500

MODEL_CONFIGS = [
    {
        "name": f"LlaVA 7B",
        "model_path": get_ollama_blob_path("llava", "model"),
        "mmproj_path": get_ollama_blob_path("llava", "projector"),
    },
    # {
    #     "name": f"LlaVA 13B",
    #     "model_path": get_ollama_blob_path("llava:13b", "model"),
    #     "mmproj_path": get_ollama_blob_path("llava:13b", "projector"),
    # },
]

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') # Decode to string

def format_prob_pct(prob, numbers=3):
    return f"{prob * 100:.{numbers}f}%"


class ForcePrefixAndExploreSuffixProcessor(LogitsProcessor):
    """
    Forces a fixed prefix and suffix rank sequence.
    """
    def __init__(self, llm_instance, fixed_prefix_token_ids, suffix_rank_sequence_to_force=(), k_store=RANKS_K, verbose=False):
        self.llm = llm_instance
        self.fixed_prefix_token_ids = fixed_prefix_token_ids
        self.num_prefix_steps = len(fixed_prefix_token_ids)
        self.suffix_rank_sequence_to_force = suffix_rank_sequence_to_force
        self.k = k_store
        self.verbose = verbose

        self.current_step = 0
        self.cumulative_logprob = 0.0
        self.logprob_at_prefix_end = -math.inf
        self.step_details = []
        self.suffix_step_analysis_data = {} # {suffix_step_index: [{'id', 'token', 'logprob'}, ...]}
        self.generated_suffix_ranks_actually_taken = []
        self.generation_successful = True

    def __call__(self, input_ids, scores):
        step_index = self.current_step

        is_prefix_step = step_index < self.num_prefix_steps
        suffix_step_index = step_index - self.num_prefix_steps if not is_prefix_step else -1

        # Calculate original probabilities and get top K
        scores_np = np.array(scores, dtype=np.float32)
        scores_exp = np.exp(scores_np - np.max(scores_np))
        probs = scores_exp / np.sum(scores_exp)
        k_to_consider = TOP_K_FOR_PREFIX_SEARCH if is_prefix_step else self.k
        top_k_indices = np.argsort(probs)[::-1][:k_to_consider]

        original_logprobs = {}
        top_k_candidates_for_analysis = []
        for i, token_id in enumerate(top_k_indices):
            probability = probs[token_id]
            logprob = math.log(probability) if probability > 0 else -math.inf
            original_logprobs[int(token_id)] = math.log(probability)
            if not is_prefix_step and i < self.k:
                top_k_candidates_for_analysis.append({
                    'id': int(token_id),
                    'token': self.llm.tokenizer().decode([int(token_id)]),
                    'logprob': logprob
                })
        if not is_prefix_step:
            self.suffix_step_analysis_data[suffix_step_index] = top_k_candidates_for_analysis


        # Determine the token to force and modify scores
        modified_scores = np.copy(scores_np)
        rank_to_force = 1
        target_token_id = -1

        if is_prefix_step:
            target_token_id = self.fixed_prefix_token_ids[step_index]
            rank_indices = np.where(top_k_indices == target_token_id)[0]
            if len(rank_indices) == 0:
                if self.verbose: print(f"Warning: Prefix token {target_token_id} not found in top {k_to_consider}")
                self.generation_successful = False
                modified_scores[:] = -float('inf')
                modified_scores[0] = 0 # token 0
            else:
                rank_index = rank_indices[0]
                rank_to_force = rank_index + 1
        else: # suffix step
            if suffix_step_index < len(self.suffix_rank_sequence_to_force):
                rank_to_force = self.suffix_rank_sequence_to_force[suffix_step_index]
                if rank_to_force > len(top_k_indices):
                    if self.verbose: print(f"Warning: Requested rank {rank_to_force} > available top_k {len(top_k_indices)}")
                    self.generation_successful = False
                    modified_scores[:] = -float('inf')
                    modified_scores[0] = 0
                else:
                    target_token_id = top_k_indices[rank_to_force - 1]
            else: # greedy
                rank_to_force = 1
                target_token_id = top_k_indices[0]

        # forcing
        if self.generation_successful and rank_to_force > 1:
            indices_to_suppress = top_k_indices[:rank_to_force - 1]
            if indices_to_suppress.size > 0:
                modified_scores[indices_to_suppress] = -float('inf')

        # chosen token and its original probability
        chosen_token_id = int(np.argmax(modified_scores)) if self.generation_successful else 0
        chosen_token_logprob = original_logprobs.get(chosen_token_id, -math.inf) if self.generation_successful else -math.inf
        chosen_token_str = self.llm.tokenizer().decode([chosen_token_id]) if self.generation_successful else "[FAIL]"

        # actual original rank taken
        actual_rank_taken = -1
        if self.generation_successful:
            try:
                rank_index = np.where(top_k_indices == chosen_token_id)[0][0]
                actual_rank_taken = rank_index + 1
            except IndexError:
                if self.verbose: print(f"Warning: Chosen token {chosen_token_id} not found in full original sorted list?")
                actual_rank_taken = -2
        else:
            actual_rank_taken = -99


        self.cumulative_logprob += chosen_token_logprob
        step_info = {
            'token_id': chosen_token_id,
            'token_str': chosen_token_str,
            'original_rank': actual_rank_taken,
            'logprob': chosen_token_logprob,
            'cumulative_logprob': self.cumulative_logprob
        }
        self.step_details.append(step_info)

        if not is_prefix_step:
            self.generated_suffix_ranks_actually_taken.append(actual_rank_taken)

        if step_index == self.num_prefix_steps - 1:
            self.logprob_at_prefix_end = self.cumulative_logprob

        self.current_step += 1
        return modified_scores

    def get_results(self):
        return {
            'details': self.step_details,
            'suffix_step_analysis': self.suffix_step_analysis_data,
            'generated_suffix_ranks': tuple(self.generated_suffix_ranks_actually_taken),
            'total_logprob': self.cumulative_logprob,
            'logprob_at_prefix_end': self.logprob_at_prefix_end,
            'num_prefix': self.num_prefix_steps,
            'num_suffix': len(self.step_details) - self.num_prefix_steps,
            'generation_successful': self.generation_successful
        }



def run_suffix_search(llm, image_path, prefix_tokens, verbose_processor=False):
    """Runs the best-first suffix search for a single image and loaded model"""
    num_prefix_steps = len(prefix_tokens)
    base64_image = image_to_base64(image_path)
    messages = [{"role": "user","content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}, {"type": "text", "text": PROMPT_TEXT}]}]

    completed_paths = []
    candidate_heap = []
    visited_suffix_rank_sequences = set()
    initial_suffix_logprob = 0.0
    tie_breaker = itertools.count()
    heapq.heappush(candidate_heap, (-initial_suffix_logprob, next(tie_breaker), tuple()))
    visited_suffix_rank_sequences.add(tuple())

    # print(f"Starting Suffix Exploration for {os.path.basename(image_path)}")

    while candidate_heap and len(completed_paths) < NUM_PATHS_TO_FIND:

        neg_logprob_suffix_so_far, _, current_suffix_ranks_to_force = heapq.heappop(candidate_heap)

        explore_processor = ForcePrefixAndExploreSuffixProcessor(
            llm_instance=llm,
            fixed_prefix_token_ids=prefix_tokens,
            suffix_rank_sequence_to_force=current_suffix_ranks_to_force,
            k_store=RANKS_K,
            verbose=verbose_processor
        )
        logits_processors = LogitsProcessorList([explore_processor])
        max_tokens_for_run = num_prefix_steps + MAX_SUFFIX_TOKENS

        try:
            _ = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens_for_run,
                temperature=0.0,
                logits_processor=logits_processors,
                stop=["<0x0A>", "\n", "</s>", "."] # stop tokens
            )
        except Exception as e:
            print(f"Error during generation for {os.path.basename(image_path)} with ranks {current_suffix_ranks_to_force}: {e}")
            continue

        results = explore_processor.get_results()

        if not results['generation_successful']:
            continue

        all_steps_details = results['details']
        actual_generated_suffix_ranks = results['generated_suffix_ranks']
        num_generated_suffix_tokens = len(actual_generated_suffix_ranks)

        if num_generated_suffix_tokens > 0:
            suffix_steps = all_steps_details[num_prefix_steps:]
            if not suffix_steps:
                continue
            
            suffix_token_id_prob_pairs = []
            for step in suffix_steps:
                token_id = step['token_id']
                logprob = step['logprob']
                prob = math.exp(logprob) if logprob > -math.inf else 0.0
                suffix_token_id_prob_pairs.append((token_id, prob))
            first_token_logprob = suffix_steps[0]['logprob']
            first_token_prob = math.exp(first_token_logprob)

            suffix_text = "".join([step['token_str'] for step in suffix_steps])
            final_suffix_logprob = sum(step['logprob'] for step in suffix_steps)
            avg_suffix_logprob = final_suffix_logprob / num_generated_suffix_tokens

            completed_info = {
                "suffix_text": suffix_text.strip(),
                "first_token_prob": first_token_prob,
                "suffix_logprob": final_suffix_logprob,
                "suffix_prob": math.exp(final_suffix_logprob),
                "avg_suffix_logprob": avg_suffix_logprob,
                "avg_suffix_prob": math.exp(avg_suffix_logprob),
                "full_rank_sequence": actual_generated_suffix_ranks,
                "num_suffix_tokens": num_generated_suffix_tokens,
                "suffix_perplexity": math.exp(-avg_suffix_logprob),
                "suffix_token_probs_list": suffix_token_id_prob_pairs
            }
            completed_paths.append(completed_info)
            print(f"Suffix: {repr(completed_info['suffix_text'])}  -  [P]{format_prob_pct(completed_info['suffix_prob'])} [AVG]{format_prob_pct(completed_info['avg_suffix_prob'])}")
            # print(f"Suffix Prob: {format_prob_pct(completed_info['suffix_prob'])} (Avg: {format_prob_pct(completed_info['avg_suffix_prob'])})")
            # print(f"Suffix Perplexity: {completed_info['suffix_perplexity']:.4f}")

            visited_suffix_rank_sequences.add(actual_generated_suffix_ranks)

        # analyze for new candidates
        suffix_analysis_data = results['suffix_step_analysis']
        logprob_accum_at_suffix_step = {}
        current_suffix_cum_logprob = 0.0
        for i in range(num_generated_suffix_tokens):
            step_detail = all_steps_details[num_prefix_steps + i]
            current_suffix_cum_logprob += step_detail['logprob']
            logprob_accum_at_suffix_step[i] = current_suffix_cum_logprob

        new_candidates_added = 0
        for i in range(num_generated_suffix_tokens):
            suffix_step_idx = i
            alternatives_at_step = suffix_analysis_data.get(suffix_step_idx, [])
            logprob_before_this_suffix_step = logprob_accum_at_suffix_step.get(i-1, 0.0)

            for rank_idx in range(1, min(RANKS_K, len(alternatives_at_step))):
                rank_to_force_next = rank_idx + 1
                chosen_rank_at_this_step = actual_generated_suffix_ranks[i]

                if rank_to_force_next == chosen_rank_at_this_step:
                    continue

                deviation_data = alternatives_at_step[rank_idx]
                dev_token_id = deviation_data['id']
                dev_logprob = deviation_data['logprob']

                new_suffix_rank_prefix = actual_generated_suffix_ranks[:i] + (rank_to_force_next,)

                if new_suffix_rank_prefix in visited_suffix_rank_sequences:
                    continue

                logprob_new_suffix_prefix = logprob_before_this_suffix_step + dev_logprob

                visited_suffix_rank_sequences.add(new_suffix_rank_prefix)
                heapq.heappush(candidate_heap, (-logprob_new_suffix_prefix, next(tie_breaker), new_suffix_rank_prefix))
                new_candidates_added += 1

    completed_paths.sort(key=lambda x: x['suffix_perplexity'], reverse=False)
    return completed_paths

def get_dynamic_prob_color(value, min_val, max_val):
    """Generates color based on value relative to dynamic min/max range (Red to Green)"""
    default_color = '#FFFFFF'
    if not isinstance(value, (int, float)) or math.isnan(value):
        return default_color
    if min_val >= max_val:
        return '#00FF00' if value == min_val else default_color
    clamped_val = max(min_val, min(max_val, value))
    normalized_score = (clamped_val - min_val) / (max_val - min_val)
    hue = normalized_score * 120.0 / 360.0
    saturation = 0.9
    lightness = 0.7
    try:
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    except Exception:
        hex_color = default_color
    return hex_color

def get_dynamic_ppl_color(value, min_val, max_val):
    """Generates color based on PPL relative to dynamic min/max range (Green to Red)"""
    default_color = '#FFFFFF'
    if not isinstance(value, (int, float)) or math.isnan(value):
        return default_color
    if min_val >= max_val:
        return '#00FF00' if value == min_val else default_color
    clamped_val = max(min_val, min(max_val, value))
    normalized_score = (max_val - clamped_val) / (max_val - min_val)
    hue = normalized_score * 120.0 / 360.0
    saturation = 0.9
    lightness = 0.7

    try:
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    except Exception:
        hex_color = default_color
    return hex_color


def create_summary_table_from_csv(csv_filepath, prompt_text, prefix_string, output_filename):
    """Creates the summary table image by reading data from the results CSV"""
    if not os.path.exists(csv_filepath):
        print(f"!!! ERROR: CSV file not found at {csv_filepath}. Cannot generate table. !!!")
        return

    # Read data from CSV and collect values for range calculation
    results_by_image = collections.defaultdict(lambda: collections.defaultdict(list))
    runtimes = {}
    unique_images = set()
    unique_model_names = []
    max_suffix_len = 0
    max_rank_seq_len = 0
    num_paths_per_entry = 0

    # Lists to store values for dynamic range calculation
    all_first_token_probs = []
    all_suffix_probs = []
    all_avg_suffix_probs = []
    all_perplexities = []

    with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames:
            print(f"!!! ERROR: CSV file {csv_filepath} is empty or has no header. !!!")
            return
        expected_headers = ["model_name", "image_path", "image_name", "path_rank", "suffix_text", "suffix_prob", "avg_suffix_prob", "suffix_perplexity", "full_rank_sequence", "runtime"]
        if not all(h in reader.fieldnames for h in expected_headers):
             print(f"!!! ERROR: CSV file {csv_filepath} is missing required headers. Found: {reader.fieldnames}. Expected: {expected_headers} !!!")
             return

        for row in reader:
            try:
                image_path = row['image_path']
                model_name = row['model_name']
                path_rank = int(row['path_rank'])
                runtime = float(row['runtime'])
                first_token_prob = float(row['first_token_prob']) if row['first_token_prob'] else None
                suffix_prob = float(row['suffix_prob'])
                avg_suffix_prob = float(row['avg_suffix_prob'])
                ppl_str = row['suffix_perplexity']
                suffix_perplexity = float(ppl_str) if ppl_str and ppl_str.lower() != 'none' else None

                if (image_path, model_name) not in runtimes:
                    runtimes[(image_path, model_name)] = runtime

                max_suffix_len = max(max_suffix_len, len(row['suffix_text']))
                rank_seq_str = row['full_rank_sequence']
                max_rank_seq_len = max(max_rank_seq_len, len(rank_seq_str))
                num_paths_per_entry = max(num_paths_per_entry, path_rank)

                all_first_token_probs.append(first_token_prob)
                all_suffix_probs.append(suffix_prob)
                all_avg_suffix_probs.append(avg_suffix_prob)
                if suffix_perplexity is not None and not math.isnan(suffix_perplexity):
                    all_perplexities.append(suffix_perplexity)

                path_data = {
                    'path_rank': path_rank,
                    'suffix_text': row['suffix_text'],
                    'first_token_prob': first_token_prob,
                    'suffix_prob': suffix_prob,
                    'avg_suffix_prob': avg_suffix_prob,
                    'suffix_perplexity': suffix_perplexity,
                    'full_rank_sequence': rank_seq_str
                }
                results_by_image[image_path][model_name].append(path_data)

                unique_images.add(image_path)
                if model_name not in unique_model_names:
                    unique_model_names.append(model_name)

            except (ValueError, KeyError, TypeError) as e:
                print(f"!!! WARNING: Skipping row due to error: {e}. Row: {row} !!!")
                continue

    min_first_tok_prob = np.nanmin(all_first_token_probs) if all_first_token_probs else 0.0
    max_first_tok_prob = np.nanmax(all_first_token_probs) if all_first_token_probs else 1.0
    min_prob = np.nanmin(all_suffix_probs) if all_suffix_probs else 0.0
    max_prob = np.nanmax(all_suffix_probs) if all_suffix_probs else 1.0
    min_avg_prob = np.nanmin(all_avg_suffix_probs) if all_avg_suffix_probs else 0.0
    max_avg_prob = np.nanmax(all_avg_suffix_probs) if all_avg_suffix_probs else 1.0
    min_ppl = np.nanmin(all_perplexities) if all_perplexities else 1.0
    max_ppl = np.nanmax(all_perplexities) if all_perplexities else 5.0

    print(f"Dynamic Ranges Calculated:")
    print(f"  First Token Prob: [{min_first_tok_prob:.4f}, {max_first_tok_prob:.4f}]")
    print(f"  Suffix Prob: [{min_prob:.4f}, {max_prob:.4f}]")
    print(f"  Avg Suffix Prob: [{min_avg_prob:.4f}, {max_avg_prob:.4f}]")
    print(f"  Perplexity: [{min_ppl:.4f}, {max_ppl:.4f}]")


    # sort paths within each model/image entry by rank
    for img_path in results_by_image:
        for model_n in results_by_image[img_path]:
            results_by_image[img_path][model_n].sort(key=lambda x: x['path_rank'])

    found_models_list = unique_model_names
    unique_images_list = sorted(list(unique_images))

    if not unique_images_list or not found_models_list:
        print("!!! ERROR: No valid data found in CSV to generate table. !!!")
        return
    if num_paths_per_entry == 0:
        if results_by_image:
            first_image = next(iter(results_by_image))
            if results_by_image[first_image]:
                first_model = next(iter(results_by_image[first_image]))
                num_paths_per_entry = len(results_by_image[first_image][first_model])
        if num_paths_per_entry == 0:
             print("!!! ERROR: Could not determine number of paths per entry from CSV data. !!!")
             return


    multiplier = 2
    padding = 0 * multiplier
    prompt_area_height = 60 * multiplier
    model_header_height = 40 * multiplier
    row_header_height = 20 * multiplier
    image_content_height = row_header_height + (num_paths_per_entry * 30 * multiplier)
    sub_row_height = (image_content_height - row_header_height) / num_paths_per_entry
    main_row_height = max(image_content_height, 20 * multiplier)
    thumb_width = int(main_row_height * (4/3))
    print(main_row_height, thumb_width)
    thumb_size = (thumb_width, main_row_height)

    # Column Widths
    img_thumb_col_width = thumb_size[0]
    img_name_col_width = 120 * multiplier
    suffix_text_col_width = int(max(150, max_suffix_len * 6.5) * multiplier)
    first_tok_prob_col_width = 100 * multiplier
    prob_col_width = 100 * multiplier
    avg_prob_col_width = 100 * multiplier
    ppl_col_width = 100 * multiplier
    rank_seq_col_width = max(120, max_rank_seq_len * 6) * multiplier
    model_block_width = suffix_text_col_width + first_tok_prob_col_width + prob_col_width + avg_prob_col_width + ppl_col_width + rank_seq_col_width

    col_widths = [img_thumb_col_width, img_name_col_width]
    col_widths.extend([model_block_width] * len(found_models_list))

    table_width = sum(col_widths) + padding * 2
    # (total height includes prompt, model headers, and space for each image row)
    table_height = prompt_area_height + model_header_height + (len(unique_images_list) * main_row_height) + padding * 2

    try:
        try:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            if not os.path.exists(font_path): font_path="arial.ttf"
            font_prompt = ImageFont.truetype(font_path.replace("Sans","Sans-Bold"), 16 * multiplier)
            font_model_header = ImageFont.truetype(font_path.replace("Sans","Sans-Bold"), 18 * multiplier)
            font_sub_header = ImageFont.truetype(font_path.replace("Sans","Sans-Bold"), 12 * multiplier)
            font_sub_row = ImageFont.truetype(font_path, 15 * multiplier)
            font_img_name = ImageFont.truetype(font_path, 18 * multiplier)
        except IOError:
            print("Warning: Specific/Arial font not found. Using PIL default.")
            default_font_size = 10 * multiplier
            font = ImageFont.load_default(size=default_font_size)
            font_prompt = font_model_header = font_sub_header = font_sub_row = font_img_name = font
    except Exception as font_e:
        print(f"Error loading font: {font_e}. Using default.")
        default_font_size = 10 * multiplier
        font = ImageFont.load_default(size=default_font_size)
        font_prompt = font_model_header = font_sub_header = font_sub_row = font_img_name = font


    img = Image.new('RGB', (table_width, table_height), color = 'white')
    draw = ImageDraw.Draw(img)

    max_prompt_width = table_width - 2 * padding
    prompt_lines = []
    current_line = "Prompt: "
    for word in prompt_text.split():
        test_line = current_line + word + " "
        bbox = draw.textbbox((0,0), test_line, font=font_prompt)
        if bbox[2] < max_prompt_width:
            current_line = test_line
        else:
            prompt_lines.append(current_line.strip())
            current_line = word + " "
    prompt_lines.append(current_line.strip())
    prompt_display_text = "\n".join(prompt_lines)
    prompt_display_text += f"\n\nPrefix: {prefix_string}"

    draw.text((padding, padding), prompt_display_text, fill='black', font=font_prompt)

    # Top Headers (Image, Name, Model Names + Runtimes)
    current_y = prompt_area_height
    current_x = padding

    header_texts = ["Image", "Image Name"] + found_models_list
    header_widths = [img_thumb_col_width, img_name_col_width] + [model_block_width] * len(found_models_list)

    for i, header in enumerate(header_texts):
        header_end_x = current_x + header_widths[i]
        header_end_y = current_y + model_header_height
        draw.rectangle([current_x, current_y, header_end_x, header_end_y], fill='#D0D0D0', outline='black')

        model_runtime_text = ""
        if i >= 2:
            model_name = header_texts[i]
            example_runtime = "N/A"
            model_runtimes = [rt for (img_p, mod_n), rt in runtimes.items() if mod_n == model_name]
            if model_runtimes:
                example_runtime = f"{np.mean(model_runtimes):.1f}s avg"

            model_runtime_text = f"\n({example_runtime})"

        text_content = f"{header}{model_runtime_text}"
        text_bbox = draw.textbbox((0,0), text_content, font=font_model_header, align='center')
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = current_x + (header_widths[i] - text_width) // 2
        text_y = current_y + (model_header_height - text_height) // 2
        draw.text((text_x, text_y), text_content, fill='black', font=font_model_header, align='center')

        current_x = header_end_x

    # Table rows
    current_y += model_header_height

    for image_path in unique_images_list:
        row_start_y = current_y
        row_end_y = row_start_y + main_row_height
        current_x = padding
        image_name = os.path.basename(image_path)

        # Image Thumbnail Column
        thumb_col_x_end = current_x + img_thumb_col_width
        draw.rectangle([current_x, row_start_y, thumb_col_x_end, row_end_y], outline='grey')
        try:
            thumb = Image.open(image_path)
            thumb.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            paste_x = current_x + (img_thumb_col_width - thumb.width) // 2
            paste_y = row_start_y + (main_row_height - thumb.height) // 2
            img.paste(thumb, (paste_x, paste_y))
        except Exception as thumb_e:
            print(f"Warning: Could not load thumbnail for {image_path}: {thumb_e}")
            err_text = "No Img"
            err_bbox = draw.textbbox((0,0), err_text, font=font_img_name)
            err_w = err_bbox[2]-err_bbox[0]; err_h = err_bbox[3]-err_bbox[1]
            err_x = current_x + (img_thumb_col_width - err_w)//2
            err_y = row_start_y + (main_row_height - err_h)//2
            draw.text((err_x, err_y), err_text, fill='grey', font=font_img_name)
        current_x = thumb_col_x_end

        # Image Name Column
        name_col_x_end = current_x + img_name_col_width
        draw.rectangle([current_x, row_start_y, name_col_x_end, row_end_y], outline='grey')
        name_bbox = draw.textbbox((0,0), image_name, font=font_img_name)
        name_h = name_bbox[3] - name_bbox[1]
        name_y = row_start_y + (main_row_height - name_h) // 2
        draw.text((current_x + 5 * multiplier, name_y), image_name, fill='black', font=font_img_name)
        current_x = name_col_x_end

        # Model Result Blocks
        for model_idx, model_name in enumerate(found_models_list):
            model_block_start_x = current_x
            model_block_end_x = model_block_start_x + model_block_width

            sub_header_y = row_start_y
            sub_header_end_y = sub_header_y + row_header_height
            sub_header_x = model_block_start_x
            sub_headers = ["Suffix", "1TkProb", "Prob", "AvgProb", "PPL", "Ranks"]
            sub_header_widths = [suffix_text_col_width, first_tok_prob_col_width, prob_col_width, avg_prob_col_width, ppl_col_width, rank_seq_col_width]

            for j, sub_h in enumerate(sub_headers):
                sub_header_cell_end_x = sub_header_x + sub_header_widths[j]
                draw.rectangle([sub_header_x, sub_header_y, sub_header_cell_end_x, sub_header_end_y], fill='#E0E0E0', outline='grey')
                sh_bbox = draw.textbbox((0,0), sub_h, font=font_sub_header)
                sh_width = sh_bbox[2] - sh_bbox[0]; sh_height = sh_bbox[3] - sh_bbox[1]
                sh_x = sub_header_x + (sub_header_widths[j] - sh_width) // 2
                sh_y = sub_header_y + (row_header_height - sh_height) // 2
                draw.text((sh_x, sh_y), sub_h, fill='black', font=font_sub_header)
                sub_header_x = sub_header_cell_end_x

            # data sub rows
            path_list = results_by_image[image_path].get(model_name, [])
            sub_row_y = sub_header_end_y

            temp_x = model_block_start_x
            for w in sub_header_widths:
                draw.line([(temp_x, sub_header_end_y), (temp_x, row_end_y)], fill='grey')
                temp_x += w
            draw.line([(temp_x, sub_header_end_y), (temp_x, row_end_y)], fill='grey')

            for i in range(num_paths_per_entry):
                sub_row_start_x = model_block_start_x
                sub_row_end_y = sub_row_y + sub_row_height
                draw.line([(model_block_start_x, sub_row_y), (model_block_end_x, sub_row_y)], fill='lightgrey')

                if i < len(path_list):
                    path_data = path_list[i]
                    first_tok_prob_val = path_data['first_token_prob']
                    suffix_prob_val = path_data['suffix_prob']
                    avg_prob_val = path_data['avg_suffix_prob']
                    perplexity_val = path_data['suffix_perplexity']

                    data_points_text = [
                        path_data['suffix_text'],
                        format_prob_pct(first_tok_prob_val, 2) if first_tok_prob_val is not None else "N/A",
                        format_prob_pct(suffix_prob_val, 2),
                        format_prob_pct(avg_prob_val, 2),
                        f"{perplexity_val:.2f}" if perplexity_val is not None else "N/A",
                        path_data['full_rank_sequence']
                    ]

                    for j, point_text in enumerate(data_points_text):
                        cell_width = sub_header_widths[j]
                        cell_start_x = sub_row_start_x
                        cell_end_x = cell_start_x + cell_width
                        cell_start_y = sub_row_y
                        cell_end_y = sub_row_end_y

                        cell_color = 'white'
                        if j == 1: # 'FirstTok%' col
                            cell_color = get_dynamic_prob_color(first_tok_prob_val, min_first_tok_prob, max_first_tok_prob)
                        elif j == 2: # 'Prob' col
                            cell_color = get_dynamic_prob_color(suffix_prob_val, min_prob, max_prob)
                        elif j == 3: # 'AvgProb' col
                            cell_color = get_dynamic_prob_color(avg_prob_val, min_avg_prob, max_avg_prob)
                        elif j == 4: # 'PPL' col
                            cell_color = get_dynamic_ppl_color(perplexity_val, min_ppl, max_ppl)

                        draw.rectangle([cell_start_x, cell_start_y, cell_end_x, cell_end_y], fill=cell_color)

                        point_bbox = draw.textbbox((0,0), str(point_text), font=font_sub_row)
                        point_h = point_bbox[3] - point_bbox[1]
                        point_y = cell_start_y + (sub_row_height - point_h) // 2
                        draw.text((cell_start_x + 5 * multiplier, point_y), str(point_text), fill='black', font=font_sub_row)

                        sub_row_start_x = cell_end_x
                else:
                    for j in range(len(sub_header_widths)):
                        cell_width = sub_header_widths[j]
                        cell_start_x = sub_row_start_x
                        cell_end_x = cell_start_x + cell_width
                        cell_start_y = sub_row_y
                        cell_end_y = sub_row_end_y

                        draw.rectangle([cell_start_x, cell_start_y, cell_end_x, cell_end_y], fill='white')

                        na_bbox = draw.textbbox((0,0), "N/A", font=font_sub_row)
                        na_h = na_bbox[3] - na_bbox[1]
                        na_y = cell_start_y + (sub_row_height - na_h) // 2
                        draw.text((cell_start_x + 5 * multiplier, na_y), "N/A", fill='grey', font=font_sub_row)
                        sub_row_start_x = cell_end_x

                sub_row_y = sub_row_end_y

            current_x = model_block_end_x

        current_y = row_end_y

    try:
        img.save(output_filename)
        print(f"Saved summary table to {output_filename}")
    except Exception as save_e:
        print(f"!!! ERROR: Failed to save image {output_filename}: {save_e} !!!")



if __name__ == "__main__":
    all_results_for_csv = []
    csv_header = [
        "model_name",
        "image_path",
        "image_name",
        "path_rank",
        "suffix_text",
        "first_token_prob",
        "suffix_prob",
        "avg_suffix_prob",
        "suffix_perplexity",
        "full_rank_sequence",
        "num_suffix_tokens",
        "suffix_token_id_probs",
        "runtime"
    ]

    for model_config in MODEL_CONFIGS:
        model_name = model_config["name"]
        model_path = model_config["model_path"]
        mmproj_path = model_config["mmproj_path"]

        print(f"\n===== Loading Model: {model_name} =====")
        print(f"Model Path: {model_path}")
        print(f"Proj Path: {mmproj_path}")

        try:
            chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                logits_all=True,
                verbose=False
            )
            prefix_tokens = llm.tokenizer().encode(PREFIX_STRING, add_bos=False)
            print(f"Prefix Token IDs: {prefix_tokens}")

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Skipping this model.")
            continue

        print(f"\n--- Processing Images in: {IMAGE_FOLDER_PATH} ---")
        image_files = [f for f in os.listdir(IMAGE_FOLDER_PATH)
                    if os.path.isfile(os.path.join(IMAGE_FOLDER_PATH, f)) and
                    os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}]

        if not image_files:
            print(f"Warning: No images found in {IMAGE_FOLDER_PATH}")
            continue

        for image_name in image_files:
            image_path = os.path.join(IMAGE_FOLDER_PATH, image_name)
            print(f"Processing: {model_name} - {image_name}")

            start_time = time.time()
            try:
                found_paths = run_suffix_search(llm, image_path, prefix_tokens, verbose_processor=False)
                end_time = time.time()
                runtime = end_time - start_time

                print(f"  Found {len(found_paths)} paths in {runtime:.2f} seconds.")
                for i, path_result in enumerate(found_paths):
                    print(path_result.get("suffix_token_probs_list", []))
                    token_id_prob_list = path_result.get("suffix_token_probs_list", [])
                    # format as string "id1:prob1;id2:prob2;..."
                    token_id_probs_str = ";".join([f"{tid}:{p:.4f}" for tid, p in token_id_prob_list])
                    result_row = {
                        "model_name": model_name,
                        "image_path": image_path,
                        "image_name": image_name,
                        "path_rank": i + 1, # 1-based rank
                        "suffix_text": path_result["suffix_text"],
                        "suffix_prob": path_result["suffix_prob"],
                        "first_token_prob": path_result["first_token_prob"],
                        "avg_suffix_prob": path_result["avg_suffix_prob"],
                        "suffix_perplexity": path_result["suffix_perplexity"],
                        "full_rank_sequence": str(path_result["full_rank_sequence"]),
                        "num_suffix_tokens": path_result["num_suffix_tokens"],
                        "suffix_token_id_probs": token_id_probs_str,
                        "runtime": runtime
                    }
                    all_results_for_csv.append(result_row)

            except Exception as e:
                end_time = time.time()
                runtime = end_time - start_time
                print(f"!! Error processing {image_name} with {model_name}: {e}")
                result_row = {
                    "model_name": model_name, "image_path": image_path, "image_name": image_name,
                    "path_rank": -1, "suffix_text": f"[ERROR: {e}]", "first_token_prob": 0, "suffix_prob": 0,
                    "avg_suffix_prob": 0, "suffix_perplexity": None, "full_rank_sequence": None,
                    "num_suffix_tokens": 0, "suffix_token_id_probs": "[ERROR]", "runtime": runtime
                }
                all_results_for_csv.append(result_row)


        del llm
        del chat_handler


    # Write Results to CSV
    print(f"\nWriting {len(all_results_for_csv)} results to {OUTPUT_CSV_PATH}")
    if all_results_for_csv:
        try:
            with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
                writer.writerows(all_results_for_csv)
            print("CSV writing complete.")
        except IOError as e:
            print(f"Error writing CSV file: {e}")
    else:
        print("No results generated to write to CSV.")

    
    # Create Summary Table from CSV
    OUTPUT_FOLDER = os.path.dirname(OUTPUT_CSV_PATH)
    SUMMARY_TABLE_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "summary_table.png")
    NUM_PATHS_TO_DISPLAY = 3
    create_summary_table_from_csv(OUTPUT_CSV_PATH, PROMPT_TEXT, PREFIX_STRING, SUMMARY_TABLE_OUTPUT_PATH)