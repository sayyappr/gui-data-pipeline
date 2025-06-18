import os
import json
import yaml
import warnings
from PIL import Image
from pathlib import Path
from pydantic import ValidationError
from schema import BoundingBox, UIElement, InteractionStep, Trajectory
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import json


# failure_reasons_counter[dataset_name][(exception_type, message)] = count
failure_reasons_counter = defaultdict(lambda: defaultdict(int))

CONFIG_PATH = "config/osatlas_config.yaml"
OUTPUT_PATH = "outputs/osatlas"
LOG_DIR = Path("logs")
WARNING_LOG_DIR = LOG_DIR / "warnings"
SKIPPED_LOG_DIR = LOG_DIR / "skipped"
SUMMARY_PATH = LOG_DIR / "summary.json"

LOG_DIR.mkdir(exist_ok=True)
WARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED_LOG_DIR.mkdir(parents=True, exist_ok=True)

summary = {
    "total_samples": 0,
    "parsed_successfully": 0,
    "warnings_count": 0,
    "skipped_count": 0
    # "skipped_files": [],
    # "warning_files": []
}

def log_to(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# def parse_element(raw, image_width, image_height):
#     bbox_norm = raw["bbox"]

#     if any(coord < 0 or coord > 1 for coord in bbox_norm):
#         raise ValueError(f"[Critical] Normalized bbox out of range: {bbox_norm}")

#     bbox_abs = BoundingBox(
#         left=int(round(bbox_norm[0] * image_width)),
#         top=int(round(bbox_norm[1] * image_height)),
#         right=int(round(bbox_norm[2] * image_width)),
#         bottom=int(round(bbox_norm[3] * image_height)),
#     )
#     el = UIElement(
#         bbox=bbox_abs,
#         element_type=raw.get("data_type") or None
#     )
#     el._instruction = raw.get("instruction", "").strip()
#     return el

def parse_element(raw, image_width, image_height):
    bbox_norm = raw["bbox"]

    if any(coord < 0 or coord > 1 for coord in bbox_norm):
        raise ValueError(f"[Critical] Normalized bbox out of range: {bbox_norm}")

    left = int(round(bbox_norm[0] * image_width))
    top = int(round(bbox_norm[1] * image_height))
    right = int(round(bbox_norm[2] * image_width))
    bottom = int(round(bbox_norm[3] * image_height))

    if right <= left or bottom <= top:
        raise ValueError(f"[BBox Parse Error] Invalid dimensions: left={left}, top={top}, right={right}, bottom={bottom}")

    bbox_abs = BoundingBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom
    )

    el = UIElement(
        bbox=bbox_abs,
        element_type=raw.get("data_type") or None
    )
    el._instruction = raw.get("instruction", "").strip()
    
    return el


def parse_interaction_step(item, image_path_root, dataset_name):
    image_id = Path(item["img_filename"]).stem
    image_path = os.path.join(image_path_root, item["img_filename"])

    # Build hierarchical log directory (e.g., desktop_domain/linux_splited)
    dataset_dir = Path(image_path_root).parent.name  # e.g., 'desktop_domain'
    json_file_stem = Path(image_path_root).name      # e.g., 'linux_splited'
    log_subdir = Path(dataset_dir) / json_file_stem  # e.g., 'desktop_domain/linux_splited'

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        failure_reasons_counter[dataset_name][(err_type, err_msg)] += 1

        summary["skipped_count"] += 1
        summary["skipped_files"].append(image_id)
        per_dataset_summary[dataset_name]["skipped"] += 1
        log_to(SKIPPED_LOG_DIR / log_subdir / f"{image_id}.txt", f"[Image Error] {e}")
        return None
    
    input_elements = item.get("elements", [])
    elements = []
    failed_elements_info = []

    for idx, e in enumerate(input_elements):
        try:
            el = parse_element(e, width, height)
            elements.append(el)
        except Exception as err:
            err_type = type(err).__name__
            err_msg = str(err)
            failure_reasons_counter[dataset_name][(err_type, err_msg)] += 1
            instruction = e.get("instruction", "[No instruction]")
            bbox = e.get("bbox", [])
            reason = f"{type(err).__name__}: {err}"

            failed_elements_info.append({
                "index": idx,
                "instruction": instruction,
                "bbox": bbox,
                "error": reason
            })

            log_message = (
                f"[BBox Parse Error] {reason}\n"
                f"[**Image**]: {image_id}\n"
                f"[**Instruction**]: {instruction}\n"
                f"[**Normalized BBox**]: {bbox}\n"
            )
            log_to(WARNING_LOG_DIR / log_subdir / f"{image_id}_element{idx}.txt", log_message)

            summary["warnings_count"] += 1
            summary["warning_files"].append(f"{image_id}#element{idx}")
            per_dataset_summary[dataset_name]["warnings_count"] += 1
            continue

    if not elements:
        summary["skipped_count"] += 1
        summary["skipped_files"].append(image_id)
        per_dataset_summary[dataset_name]["skipped"] += 1

        reason = (
            "No UI elements were present.\n"
            if len(input_elements) == 0 else
            "All UI elements failed validation.\n"
        )

        log_msg = (
            f"{reason}\n"
            f"Image file: {item['img_filename']}\n\n"
            f"Failed elements:\n" + json.dumps(failed_elements_info, indent=2)
        )
        log_to(SKIPPED_LOG_DIR / log_subdir / f"{image_id}.txt", log_msg)
        return None


    summary["parsed_successfully"] += 1
    per_dataset_summary[dataset_name]["parsed"] += 1

    return InteractionStep(
        image_path=f"{dataset_name}/{item["img_filename"]}",
        image_width=width,
        image_height=height,
        all_ui_elements=elements,
        conversation_list=[],
    )


def parse_all_datasets(config_yaml):
    with open(config_yaml, "r") as f:
        config = yaml.safe_load(f)

    all_steps = []
    summary["datasets"] = {}  # Add per-dataset stats

    for dataset in config["datasets"]:
        json_path = dataset["json_path"]
        image_dir = dataset["images_folder"]

        with open(json_path, "r") as jf:
            data = json.load(jf)

        parsed = 0
        dataset_name = os.path.basename(json_path)
        
        # Auto-detect flat format and group by image
        if isinstance(data, list) and "elements" not in data[0]:
            print(f"🛠 Detected flat element list in {dataset_name} — regrouping by image")
            grouped = defaultdict(list)
            for entry in data:
                grouped[entry["img_filename"]].append({
                    "instruction": entry.get("instruction"),
                    "bbox": entry.get("bbox"),
                    "data_type": entry.get("data_type")
                })
            data = [{"img_filename": img, "elements": elements} for img, elements in grouped.items()]

        for item in tqdm(data, desc=f"Parsing {json_path}"):
            summary["total_samples"] += 1
            step = parse_interaction_step(item, image_dir, dataset["name"])
            if step:
                all_steps.append(step)
                parsed += 1

        skipped = len(data) - parsed
        warnings = len([
            w for w in summary["warning_files"]
            if dataset_name in w or os.path.splitext(json_path)[0] in w
        ])

        summary["datasets"][dataset_name] = {
            "total": len(data),
            "parsed": parsed,
            "skipped": skipped,
            "warnings": warnings,
            "parsed_percent": round(100 * parsed / len(data), 2),
            "skipped_percent": round(100 * skipped / len(data), 2),
            "warning_percent": round(100 * warnings / len(data), 2)
        }

    # Optional: Save to disk
    with open("logs/final_summary.json", "w") as f:
        json.dump(summary["datasets"], f, indent=2)

    return all_steps, config["datasets"]


def generate_grouped_output(domain_to_steps, output_path):
    from schema import Trajectory

    trajectory_list = []
    for domain, steps in domain_to_steps.items():
        # Group by image_path to merge annotations
        grouped_by_image = defaultdict(list)
        for step in steps:
            grouped_by_image[step.image_path].append(step)

        merged_steps = []
        for image_path, step_group in grouped_by_image.items():
            all_elements = []
            for step in step_group:
                all_elements.extend(step.all_ui_elements)

            seen = set()
            unique_elements = []
            for el in all_elements:
                key = (
                    el.bbox.left, el.bbox.top,
                    el.bbox.right, el.bbox.bottom,
                    el.element_type
                )
                if key not in seen:
                    seen.add(key)
                    unique_elements.append(el)

            merged_steps.append(
                InteractionStep(
                    image_path=image_path,
                    image_width=step_group[0].image_width,
                    image_height=step_group[0].image_height,
                    all_ui_elements=unique_elements,
                    conversation_list=[],
                )
            )

        traj = Trajectory(
            data_source="os-atlas",
            is_navigation=False,
            domain=domain,
            steps=merged_steps
        )
        trajectory_list.append(traj.model_dump())

    # Save the unified grouped trajectory list
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectory_list, f, indent=2)

    print(f"Grouped Trajectory list saved to {output_path}")



# Global summary
summary = {
    "parsed_successfully": 0,
    "skipped_count": 0,
    "warnings_count": 0,
    "skipped_files": [],
    "warning_files": [],
    "total_samples": 0,
}

# Per-dataset summary
per_dataset_summary = defaultdict(lambda: {
    "parsed": 0,
    "skipped": 0,
    "warnings_count": 0
})

# def main():
#     steps, dataset_configs = parse_all_datasets(CONFIG_PATH)

#     # Determine the majority domain (based on config)
#     domain_counter = defaultdict(int)
#     domain_map = {}  # dataset_name → domain

#     for ds in dataset_configs:
#         domain_map[ds["name"]] = ds["domain"]
        

#     # Group steps by domain
#     domain_to_steps = defaultdict(list)
#     for step in steps:
#         dataset_name = step.image_path.split("/")[0]  # e.g., 'linux'
#         domain = domain_map.get(dataset_name, "web")
#         domain_to_steps[domain].append(step)

#     trajectories = []
#     for domain, steps in domain_to_steps.items():
#         traj = Trajectory(
#             data_source="os-atlas",
#             is_navigation=False,
#             domain=domain,
#             steps=steps
#         )
#         trajectories.append(traj.model_dump())                            
#         output_path = f"outputs/osatlas_parsed_{domain}.json"
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, "w") as f:
#             f.write(traj.model_dump_json(indent=2))
#         print(f"✅ Saved trajectory for domain '{domain}' to {output_path}")

#     generate_grouped_output(steps, "parsed_output_grouped.json")

#     with open(SUMMARY_PATH, "w") as f:
#         json.dump(summary, f, indent=2)

#     print("\n📊 Per-Dataset Summary:")
#     for dataset, stats in per_dataset_summary.items():
#         total = stats["parsed"] + stats["skipped"]
#         parsed_pct = (stats["parsed"] / total) * 100 if total > 0 else 0
#         skipped_pct = (stats["skipped"] / total) * 100 if total > 0 else 0
#         warnings = stats["warnings_count"]
#         print(f"  {dataset}: Parsed = {stats['parsed']} / {total} ({parsed_pct:.2f}%) | Skipped = {stats['skipped']} ({skipped_pct:.2f}%) | Warnings = {warnings}")

#     formatted_failure_reasons = {
#         dataset: {
#             f"{err_type}: {err_msg}": count
#             for (err_type, err_msg), count in reasons.items()
#         }
#         for dataset, reasons in failure_reasons_counter.items()
#     }

#     with open("logs/failure_summary_log.json", "w") as f:
#         json.dump(formatted_failure_reasons, f, indent=2)

#     print("✅ OS Atlas parsing complete.")
#     print(f"Parsed steps: {summary['parsed_successfully']} / {summary['total_samples']}")
#     print(f"Warnings: {summary['warnings_count']} | Skipped: {summary['skipped_count']}")


def main():
    steps, dataset_configs = parse_all_datasets(CONFIG_PATH)

    # Map dataset name to its domain
    domain_map = {ds["name"]: ds["domain"] for ds in dataset_configs}

    # Group steps by domain
    domain_to_steps = defaultdict(list)
    for step in steps:
        dataset_name = step.image_path.split("/")[0]  # e.g., 'linux'
        domain = domain_map.get(dataset_name, "web")
        domain_to_steps[domain].append(step)

    all_trajectories = []

    # Save per-domain trajectories
    for domain, steps in domain_to_steps.items():
        traj = Trajectory(
            data_source="os-atlas",
            is_navigation=False,
            domain=domain,
            steps=steps
        )
        all_trajectories.append(traj.model_dump())

        output_path = f"{OUTPUT_PATH}/osatlas_parsed_{domain}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(traj.model_dump_json(indent=2))
        print(f"Saved trajectory for domain '{domain}' to {output_path}")

    # Save unified trajectory file (list of trajectories)
    unified_path = f"{OUTPUT_PATH}/osatlas_parsed_unified.json"
    with open(unified_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)
    print(f"📦 Saved unified trajectory list to {unified_path}")

    # Save grouped output (raw merged steps)
    generate_grouped_output(domain_to_steps, f"{OUTPUT_PATH}/parsed_output_grouped.json")

    # Save summary
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print per-dataset stats
    print("\n📊 Per-Dataset Summary:")
    for dataset, stats in per_dataset_summary.items():
        total = stats["parsed"] + stats["skipped"]
        parsed_pct = (stats["parsed"] / total) * 100 if total > 0 else 0
        skipped_pct = (stats["skipped"] / total) * 100 if total > 0 else 0
        warnings = stats["warnings_count"]
        print(f"  {dataset}: Parsed = {stats['parsed']} / {total} ({parsed_pct:.2f}%) | Skipped = {stats['skipped']} ({skipped_pct:.2f}%) | Warnings = {warnings}")

    # Format normalized failure reasons
    def normalize_reason(err_type, err_msg):
        if err_type == "ValueError" and "Invalid dimensions" in err_msg:
            return "ValueError: Invalid bounding box dimensions"
        elif err_type == "ValueError" and "Normalized bbox out of range" in err_msg:
            return "ValueError: Normalized bbox out of range"
        elif err_type == "IsADirectoryError":
            return "IsADirectoryError: Path is a directory, not an image"
        else:
            return f"{err_type}: {err_msg}"

    normalized_failure_reasons = defaultdict(lambda: defaultdict(int))
    for dataset, reason_counts in failure_reasons_counter.items():
        for (err_type, err_msg), count in reason_counts.items():
            key = normalize_reason(err_type, err_msg)
            normalized_failure_reasons[dataset][key] += count

    # Save failure summary
    with open("logs/failure_summary_log.json", "w") as f:
        json.dump(normalized_failure_reasons, f, indent=2)

    print("✅ OS Atlas parsing complete.")
    print(f"Parsed steps: {summary['parsed_successfully']} / {summary['total_samples']}")
    print(f"Warnings: {summary['warnings_count']} | Skipped: {summary['skipped_count']}")


if __name__ == "__main__":
    main()


# Step 1: Normalize error types/messages
def normalize_reason(err_type, err_msg):
    if err_type == "ValueError" and "Invalid dimensions" in err_msg:
        return "ValueError: Invalid bounding box dimensions"
    elif err_type == "IsADirectoryError":
        return "IsADirectoryError: Path is a directory, not an image"
    else:
        return f"{err_type}: {err_msg}"

# Step 2: Aggregate cleaned failure reasons
normalized_failure_reasons = defaultdict(lambda: defaultdict(int))

for dataset, reason_counts in failure_reasons_counter.items():
    for (err_type, err_msg), count in reason_counts.items():
        key = normalize_reason(err_type, err_msg)
        normalized_failure_reasons[dataset][key] += count

# Step 3: Save the cleaned summary
os.makedirs("logs", exist_ok=True)
with open("logs/failure_summary_log.json", "w") as f:
    json.dump(normalized_failure_reasons, f, indent=2)
