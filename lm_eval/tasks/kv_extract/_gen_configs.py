#!/usr/bin/env python3
import os

# Global variables: adjust these lists as needed.
CONTEXT_SIZES = [4096, 8192]
NUM_QUERIES = [1, 2]

# Template for individual task YAML files.
# These files include dataset_name and include.
INDIVIDUAL_TEMPLATE = """dataset_name: ctx_{ctx}_num_q_{num_q}
include: _default_template_yaml
task: kv_extract_ctx_{ctx}_num_q_{num_q}
"""

# Template for the aggregated group YAML file.
# This one removes dataset_name and include, and uses the keys 'group', 'task', etc.
GROUP_TEMPLATE = """group: kv_extract
task:
{tasks_list}
aggregate_metric_list:
  - metric: acc
    aggregation: mean
metadata:
  version: 1.0
"""

def main():
    # Determine the directory where the script is located.
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # List to collect the names of all individual tasks.
    task_names = []

    # Generate individual YAML files for every combination of context size and number of queries.
    for ctx in CONTEXT_SIZES:
        for num_q in NUM_QUERIES:
            task_name = f"kv_extract_ctx_{ctx}_num_q_{num_q}"
            task_names.append(task_name)
            filename = f"{task_name}.yaml"
            filepath = os.path.join(script_dir, filename)
            content = INDIVIDUAL_TEMPLATE.format(ctx=ctx, num_q=num_q)
            with open(filepath, "w") as f:
                f.write(content)
            print(f"Generated individual task file: {filepath}")

    # Create the aggregated group YAML file.
    # Build the list of tasks with proper indentation (2 spaces for each entry).
    tasks_list = "".join(f"  - {name}\n" for name in task_names)
    group_filename = "_kv_extract.yaml"
    group_filepath = os.path.join(script_dir, group_filename)
    group_content = GROUP_TEMPLATE.format(tasks_list=tasks_list.rstrip())
    with open(group_filepath, "w") as f:
        f.write(group_content)
    print(f"Generated aggregated group file: {group_filepath}")

if __name__ == "__main__":
    main()
