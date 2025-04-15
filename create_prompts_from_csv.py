
import csv
import itertools
import os

"""

This script takes a CSV file containing a list of options for a prompt and generates all possible combinations of those options.

example of the csv file format:

	part1name		part2name		part3name		...
	option1			option1			option1			...
	option2			option2			(none)			...
					option3							...
					...


The script will create a text file containing all possible combinations of the options of the CSV file.

"""

def clean_prompt(prompt):
    """Clean spacing, punctuation, and capitalization in a prompt."""
    prompt = prompt.strip()
    if not prompt:
        return ""

    # Capitalize first letter
    prompt = prompt[0].upper() + prompt[1:]

    # Ensure proper punctuation at end
    if prompt[-1] not in ".!?":
        prompt += "."
    return prompt

def generate_prompts_from_csv(csv_path, output_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')  # Use tab as delimiter
        headers = next(reader)  # skip header row
        rows = list(reader)

    # Transpose rows into columns, skipping empty cells
    num_columns = len(headers)
    columns = [[] for _ in range(num_columns)]

    for row in rows:
        for i in range(num_columns):
            if i < len(row):
                value = row[i]
                if value == "":
                    continue  # skip truly empty cells
                elif value.lower() == "(none)":
                    columns[i].append("")  # treat "(none)" as an empty string
                else:
                    columns[i].append(value)

    # Generate all combinations
    all_combinations = itertools.product(*columns)

    # Write cleaned prompt combinations to output
    with open(output_path, "w", encoding='utf-8') as out_file:
        for combo in all_combinations:
            prompt = "".join(part for part in combo)
            prompt = clean_prompt(prompt)
            if prompt:
                out_file.write(prompt + "\n")

# Example usage
generate_prompts_from_csv("ressources/prompt_exploration.csv", "output/all_prompts.txt")
