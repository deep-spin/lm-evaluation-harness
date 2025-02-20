import argparse
import json
import random
import uuid
import os

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm

# --- Deterministic UUID generator ---
def deterministic_uuid():
    # Generate a UUID using random.getrandbits(128); this will be deterministic given a fixed seed.
    return str(uuid.UUID(int=random.getrandbits(128)))

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

# Global variable for storing context hints.
CONTEXT_HINTS = {}

def generate_pairs_until(tokenizer, target_token_count, fmt, min_pairs_required):
    """
    Generate key–value pairs using a binary search strategy and a context size hint.
    
    A global dictionary, CONTEXT_HINTS, maps target token counts to the minimal number of pairs
    found previously. If a hint for a smaller target is available, the hint is scaled by the ratio
    of target_token_count to the known target.
    
    The function returns the minimal list of pairs and the corresponding formatted context
    that reaches at least target_token_count tokens and at least min_pairs_required pairs.
    """
    global CONTEXT_HINTS

    pairs = []
    seen = set()

    def get_more(n):
        while len(pairs) < n:
            new_key = deterministic_uuid()
            new_value = deterministic_uuid()
            # Ensure uniqueness.
            if new_key in seen or new_value in seen:
                continue
            seen.add(new_key)
            seen.add(new_value)
            pairs.append((new_key, new_value))
    
    # Set the initial lower bound.
    low = min_pairs_required
    # Determine a starting guess for the upper bound using hints if available.
    candidate_keys = [k for k in CONTEXT_HINTS if k <= target_token_count]
    if candidate_keys:
        best_candidate = max(candidate_keys)
        start_guess = int(CONTEXT_HINTS[best_candidate] * (target_token_count / best_candidate))
    else:
        start_guess = 500
    high = max(start_guess, min_pairs_required)

    get_more(high)
    context_str = build_context(pairs[:high], fmt)
    token_count = len(tokenizer.encode(context_str))
    
    # Expand high until the token count requirement is met.
    while token_count < target_token_count:
        low = high
        high *= 2
        get_more(high)
        context_str = build_context(pairs[:high], fmt)
        token_count = len(tokenizer.encode(context_str))
    
    # Binary search between low and high.
    while low < high:
        mid = (low + high) // 2
        context_str = build_context(pairs[:mid], fmt)
        token_count = len(tokenizer.encode(context_str))
        if token_count >= target_token_count:
            high = mid
        else:
            low = mid + 1
    
    n = low
    if n < min_pairs_required:
        raise ValueError(f"Could not generate enough pairs to reach {min_pairs_required} pairs.")
    context_str = build_context(pairs[:n], fmt)
    
    # Save the hint for the current target_token_count.
    CONTEXT_HINTS[target_token_count] = n
    
    return pairs[:n], context_str

def select_demo_and_query_groups(total_pairs, num_queries, num_demos):
    """
    Select demonstration groups and query indices with no overlap.
    Each demonstration group will contain `num_queries` indices.
    The query group is also a set of `num_queries` indices.
    This function guarantees that there is no overlap between the demo groups and the query.
    """
    all_indices = list(range(total_pairs))
    required_demo = num_demos * num_queries
    required_total = required_demo + num_queries
    if total_pairs < required_total:
        raise ValueError("Not enough pairs to ensure non-overlap between demonstrations and query.")
    
    # Select demonstration indices first.
    demo_chosen = random.sample(all_indices, required_demo)
    remaining = list(set(all_indices) - set(demo_chosen))
    query_indices = random.sample(remaining, num_queries)
    
    # Split demonstration indices into groups.
    demo_groups = [demo_chosen[i * num_queries:(i+1) * num_queries] for i in range(num_demos)]
    return demo_groups, query_indices

def generate_example(context_size, tokenizer, fmt, num_q, num_demos):
    """
    Generate one dataset record (an example) and return a tuple (record, first_depth).
    """
    # We need at least (1 + num_demos) groups of pairs.
    min_pairs_required = num_q * (1 + num_demos)
    pairs, context_str = generate_pairs_until(tokenizer, context_size, fmt, min_pairs_required)

    # Select demonstration groups and query indices (ensuring no overlap).
    demo_groups, query_indices = select_demo_and_query_groups(len(pairs), num_q, num_demos)

    # Build demonstration examples.
    demonstrations = []
    for group in demo_groups:
        demo_keys = [pairs[idx][0] for idx in group]
        demo_values = [pairs[idx][1] for idx in group]
        # Simply concatenate the keys.
        demo_question = ", ".join(demo_keys)
        demo_answer = ", ".join(demo_values)
        demonstrations.append({"question": demo_question, "answer": demo_answer})

    # Retrieve query keys and corresponding answer values.
    query_keys = [pairs[idx][0] for idx in query_indices]
    query_values = [pairs[idx][1] for idx in query_indices]

    # Compute the depth (as a percentage) of each query key and take the first key's depth.
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

def generate_dataset(context_size, model_name_or_path, fmt, num_q,
                     num_examples_per_bucket, num_buckets, num_demos):
    """
    Generate a dataset (as a Hugging Face Dataset) for a single configuration.
    """    
    # Load tokenizer for token counting.
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    records = []
    bucket_size = 100 / num_buckets

    # Use tqdm to track progress across all buckets.
    with tqdm(total=num_examples_per_bucket * num_buckets,
              desc=f"Generating for ctx={context_size}, num_q={num_q}") as pbar:
        for bucket in range(num_buckets):
            bucket_lower = bucket * bucket_size
            bucket_upper = (bucket + 1) * bucket_size
            count_in_bucket = 0
            attempts = 0
            max_attempts = 100 * num_examples_per_bucket  # safety cap
            while count_in_bucket < num_examples_per_bucket and attempts < max_attempts:
                attempts += 1
                record, first_depth = generate_example(context_size, tokenizer, fmt, num_q, num_demos)
                # Ensure the first query key's depth falls within the current bucket.
                if bucket_lower <= first_depth < bucket_upper:
                    records.append(record)
                    count_in_bucket += 1
                    pbar.update(1)
            if count_in_bucket < num_examples_per_bucket:
                print(f"Warning: Only generated {count_in_bucket} examples for bucket {bucket} "
                      f"({bucket_lower:.2f}% to {bucket_upper:.2f}%) after {attempts} attempts.")

    ds = Dataset.from_list(records)
    print(f"Generated {len(records)} examples across {num_buckets} buckets for ctx={context_size}, num_q={num_q}.")
    return ds

# --- Main entry point ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate a DatasetDict for testing long context extraction tasks "
                    "with multiple configurations (each configuration is a split)."
    )
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path to store the DatasetDict.")
    parser.add_argument("--push_to_hub_name", type=str, default=None,
                        help="Optional: Push the generated DatasetDict to the Hub.")
    parser.add_argument("--push_to_hub_private", action="store_true",
                        help="Optional: Push the generated DatasetDict to the Hub as private.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--context_sizes", type=int, nargs="+", required=True,
                        help="List of target context sizes (in tokens) for each generated record.")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Model name or path to load the tokenizer (e.g., 'gpt2').")
    parser.add_argument("--format", type=str, choices=["json", "csv", "tsv", "text"], required=True,
                        help="Format of the generated model input: json, csv, tsv, or text (with 'key => value' per line).")
    parser.add_argument("--num_queries", type=int, nargs="+", required=True,
                        help="List of numbers of query keys the model should extract.")
    parser.add_argument("--num_examples_per_bucket", type=int, required=True,
                        help="Number of examples to generate in each depth bucket.")
    parser.add_argument("--num_buckets", type=int, required=True,
                        help="Number of equally sized depth buckets (over 0% to 100%).")
    parser.add_argument("--num_demos", type=int, default=2,
                        help="Optional: Number of demonstration examples per record (default is 2).")

    args = parser.parse_args()

    random.seed(args.seed)

    dataset_splits = {}
    # For each combination (Cartesian product) of context size and number of query keys,
    # generate a split.
    for ctx in args.context_sizes:
        for num_q in args.num_queries:
            split_name = f"ctx_{ctx}_num_q_{num_q}"
            print(f"Generating split: {split_name}")
            ds = generate_dataset(
                context_size=ctx,
                model_name_or_path=args.model_name_or_path,
                fmt=args.format,
                num_q=num_q,
                num_examples_per_bucket=args.num_examples_per_bucket,
                num_buckets=args.num_buckets,
                num_demos=args.num_demos
            )
            dataset_splits[split_name] = ds

    dataset_dict = DatasetDict(dataset_splits)

    if args.output_path:
        if os.path.exists(args.output_path):
            raise ValueError(f"Output path {args.output_path} already exists")
        dataset_dict.save_to_disk(args.output_path)
        print(f"DatasetDict saved to {args.output_path}")
    else:
        print("No output path provided; data generated but not saved to disk.")
    
    if args.push_to_hub_name:
        dataset_dict.push_to_hub(
            args.push_to_hub_name,
            private=args.push_to_hub_private,
        )
        print(f"DatasetDict pushed to Hub with name '{args.push_to_hub_name}': private={args.push_to_hub_private}.")
    else:
        print("No hub name provided; data generated but not pushed to hub.")

if __name__ == "__main__":
    main()