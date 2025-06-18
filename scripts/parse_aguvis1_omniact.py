import os
import re
import json
from PIL import Image
from tqdm import tqdm
from schema import *
import pyautogui

# === CONFIG ===
input_json = "data/aguvis_stage1/omniact/omniact_fix.json"
image_folder = "data/aguvis_stage1/omniact/images"
output_json = "outputs/aguvis_omniact_all.json"
log_dir = "logs/failed"

# === SETUP ===
os.makedirs("outputs", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === HELPER: Normalize relative click to pixel click ===
def normalize_click_command(cmd: str, width: int, height: int) -> str:
    match = re.match(r"pyautogui\.click\(x=([0-9.]+), y=([0-9.]+)\)", cmd)
    if match:
        x_rel = float(match.group(1))
        y_rel = float(match.group(2))
        x_abs = int(round(x_rel * width))
        y_abs = int(round(y_rel * height))
        return f"pyautogui.click(x={x_abs}, y={y_abs})"
    return cmd

# === HELPER: Write individual failure logs ===
def write_individual_log(index: int, image_name: str, reason: str, details: str = ""):
    safe_name = image_name.replace("/", "_")
    log_path = os.path.join(log_dir, f"{str(index).zfill(3)}_{safe_name}.log")
    with open(log_path, "w") as f:
        f.write(f"Image: {image_name}\n")
        f.write(f"Reason: {reason}\n")
        if details:
            f.write(f"Details: {details}\n")

# === LOAD DATA ===
with open(input_json, "r") as f:
    data = json.load(f)

output = []

# === MAIN PARSE LOOP ===
for i, entry in enumerate(tqdm(data)):
    image_name = entry.get("image")
    image_path_full = os.path.join(image_folder, image_name)

    # --- Skip if image missing ---
    if not os.path.exists(image_path_full):
        write_individual_log(i, image_name, "MISSING IMAGE")
        continue

    # --- Load image size ---
    try:
        with Image.open(image_path_full) as img:
            width, height = img.size
    except Exception as e:
        write_individual_log(i, image_name, "IMAGE OPEN ERROR", str(e))
        continue

    # --- Parse conversation ---
    conv = entry.get("conversations", [])
    if len(conv) != 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
        write_individual_log(i, image_name, "INVALID CONVERSATION FORMAT")
        continue

    try:
        # Human input
        #human_msg = UIConversationInput(from_="human", value=conv[0]["value"])
        human_msg = UIConversationInput.model_validate(conv[0])
        
        # GPT actions (may have multiple pyautogui lines)
        raw_actions = conv[1]["value"].split("\n")
        actions = []
        for cmd in raw_actions:
            cmd = cmd.strip()
            if not cmd:
                continue
            norm_cmd = normalize_click_command(cmd, width, height)
            actions.append(UIAction(pyautogui=norm_cmd))

        #gpt_msg = UIConversationOutput(from_="gpt", actions=actions)
        gpt_msg = UIConversationOutput.model_validate({
            "from": "gpt",
            "actions": actions
            })

        # Build step
        step = InteractionStep(
            image_path=f"aguvis/train/images/{image_name}",
            image_width=width,
            image_height=height,
            all_ui_elements=None,
            conversation_list=[[human_msg, gpt_msg]]
        )

        # Build trajectory
        traj = Trajectory(
            data_source="aguvis-stage1",
            is_navigation=False,
            domain="web",
            steps=[step]
        )

        output.append(traj.model_dump(mode="python", exclude_none=True))

    except Exception as e:
        write_individual_log(i, image_name, "PARSING ERROR", str(e))
        continue

# === WRITE OUTPUT ===
with open(output_json, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ DONE:")
print(f" - Parsed: {len(output)}")
print(f" - Skipped: {len(os.listdir(log_dir))} → see {log_dir}/")
