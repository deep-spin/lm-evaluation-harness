import argparse
import json
import random
import uuid
import os

from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm

# --- Helper functions ---

def build_context(pairs, fmt):
    """
    Build a context string from a list of (key, value) pairs
    according to the specified format.
    """
    if fmt == "json":
        context_obj = {k: v for k, v in pairs}
        return json.dumps(context_obj)
    elif fmt == "csv":
        return "\n".join(f"{k},{v}" for k, v in pairs)
    elif fmt == "tsv":
        return "\n".join(f"{k}\t{v}" for k, v in pairs)
    elif fmt == "text":
        return "\n".join(f"{k} => {v}" for k, v in pairs)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def generate_pairs_until(tokenizer, target_token_count, fmt, min_pairs_required):
    """
    Generate key–value pairs until the formatted context reaches at least
    `target_token_count` tokens and at least `min_pairs_required` pairs have been generated.
    """
    pairs = []
    all_uuids = set()
    context_str = ""
    while True:
        new_key = str(uuid.uuid4())
        new_value = str(uuid.uuid4())
        if new_key in all_uuids or new_value in all_uuids:
            continue
        all_uuids.add(new_key)
        all_uuids.add(new_value)
        pairs.append((new_key, new_value))
        context_str = build_context(pairs, fmt)
        token_count = len(tokenizer.encode(context_str))
        if token_count >= target_token_count and len(pairs) >= min_pairs_required:
            break
    return pairs, context_str

def select_demo_and_query_groups(total_pairs, num_extracted_keys, num_demos):
    """
    Select demonstration groups and query indices with no overlap.
    Each demonstration group will contain `num_extracted_keys` indices.
    The query group is also a set of `num_extracted_keys` indices.
    This function guarantees that there is no overlap between the demo groups and the query.
    """
    all_indices = list(range(total_pairs))
    required_demo = num_demos * num_extracted_keys
    required_total = required_demo + num_extracted_keys
    if total_pairs < required_total:
        raise ValueError("Not enough pairs to ensure non-overlap between demonstrations and query.")
    
    # Select demonstration indices first.
    demo_chosen = random.sample(all_indices, required_demo)
    remaining = list(set(all_indices) - set(demo_chosen))
    query_indices = random.sample(remaining, num_extracted_keys)
    
    # Split demonstration indices into groups.
    demo_groups = [demo_chosen[i * num_extracted_keys:(i+1) * num_extracted_keys] for i in range(num_demos)]
    return demo_groups, query_indices


def generate_example(context_size, tokenizer, fmt, num_extracted_keys, num_demos):
    """
    Generate one dataset record (an example) and return a tuple (record, depth).
    """
    # We need at least (1 + num_demos) groups of pairs.
    min_pairs_required = num_extracted_keys * (1 + num_demos)
    pairs, context_str = generate_pairs_until(tokenizer, context_size, fmt, min_pairs_required)

    # Select demonstration groups and query indices (ensuring no overlap).
    demo_groups, query_indices = select_demo_and_query_groups(len(pairs), num_extracted_keys, num_demos)

    # Build demonstration examples.
    demonstrations = []
    for group in demo_groups:
        demo_keys = [pairs[idx][0] for idx in group]
        demo_values = [pairs[idx][1] for idx in group]
        # Instead of using a template, simply concatenate the keys.
        demo_question = ", ".join(demo_keys)
        demo_answer = ", ".join(demo_values)
        demonstrations.append({"question": demo_question, "answer": demo_answer})

    # Retrieve query keys and corresponding answer values.
    query_keys = [pairs[idx][0] for idx in query_indices]
    query_values = [pairs[idx][1] for idx in query_indices]

    # Compute the average depth (position percentage) of the query keys.
    depths = [((idx + 1) / len(pairs)) * 100 for idx in query_indices]
    avg_depth = sum(depths) / len(depths)

    question = ", ".join(query_keys)
    answer = ", ".join(query_values)

    record = {
        "context": context_str,
        "demonstrations": demonstrations,
        "question": question,
        "answer": answer,
        "first_depth": depths[0],
        "depth": avg_depth,
        "num_pairs": len(pairs)
    }
    return record, depths[0]

# --- Main dataset generation function ---

def generate_dataset(output_path, context_size, model_name_or_path, fmt, num_extracted_keys,
                     num_examples_per_bucket, num_buckets, num_demos):
    if output_path and os.path.exists(output_path):
        raise ValueError(f"Output path {output_path} already exists")
    
    # Load tokenizer for token counting.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    records = []
    bucket_size = 100 / num_buckets

    # Total examples will be number_of_examples_at_depth * num_parts
    with tqdm(total=num_examples_per_bucket * num_buckets) as pbar:
        for bucket in range(num_buckets):
            bucket_lower = bucket * bucket_size
            bucket_upper = (bucket + 1) * bucket_size
            count_in_bucket = 0
            attempts = 0
            max_attempts = 100 * num_examples_per_bucket  # safety cap
            while count_in_bucket < num_examples_per_bucket and attempts < max_attempts:
                attempts += 1
                record, first_depth = generate_example(context_size, tokenizer, fmt, num_extracted_keys, num_demos)
                # We ensure that the first depth is within the current bucket.
                # This is similar to the original Needle in a Haystack benchmakr.
                if bucket_lower <= first_depth < bucket_upper:
                    records.append(record)
                    count_in_bucket += 1
                    pbar.update(1)
            if count_in_bucket < num_examples_per_bucket:
                print(f"Warning: Only generated {count_in_bucket} examples for bucket {bucket} "
                    f"({bucket_lower:.2f}% to {bucket_upper:.2f}%) after {attempts} attempts.")

    dataset = Dataset.from_list(records)
    print(f"Generated {len(records)} examples across {num_buckets} buckets.")

    if output_path:
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    else:
        print("No output path provided; dataset generated but not saved to disk.")

    return dataset

# --- Main entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate a dataset for testing long context extraction tasks "
                    "with a specified number of examples per depth bucket."
    )
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional output path to store the dataset in .parquet format using the datasets library.")
    parser.add_argument("--context_size", type=int, required=True,
                        help="Target context size in number of tokens for each generated record.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path to load the tokenizer (e.g., 'gpt2').")
    parser.add_argument("--format", type=str, choices=["json", "csv", "tsv", "text"], required=True,
                        help="Format of the generated model input: json, csv, tsv, or text (with 'key => value' per line).")
    parser.add_argument("--num_extracted_keys", type=int, required=True,
                        help="Number of keys the model should extract (i.e. query keys).")
    parser.add_argument("--num_examples_per_bucket", type=int, required=True,
                        help="Number of examples to generate in each depth bucket.")
    parser.add_argument("--num_buckets", type=int, required=True,
                        help="Number of equally sized depth buckets (over 0% to 100%).")
    parser.add_argument("--num_demos", type=int, default=2,
                        help="Optional: Number of demonstration examples per record (default is 2).")

    args = parser.parse_args()

    generate_dataset(
        output_path=args.output_path,
        context_size=args.context_size,
        model_name_or_path=args.model_name_or_path,
        fmt=args.format,
        num_extracted_keys=args.num_extracted_keys,
        num_examples_per_bucket=args.num_examples_per_bucket,
        num_buckets=args.num_buckets,
        num_demos=args.num_demos
    )

if __name__ == "__main__":
    main()