import json
from collections import OrderedDict

def add_mode_to_json(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        data = json.loads(line)
        first_key = list(data.keys())[0]
        mode = "train" if first_key == "lr" else "val" if first_key == "acc/top1" else None
        if mode:
            new_data = OrderedDict([("mode", mode)])
            new_data.update(data)
            if mode == "val":
                step = data.get("step")
                new_data["iter"] = step
                new_data["epoch"] = step
            updated_lines.append(json.dumps(new_data))

    with open(output_file_path, 'w') as file:
        file.write("\n".join(updated_lines))

# Replace 'input_file_path.json' and 'output_file_path.json' with the actual paths to your JSON files
add_mode_to_json(
    'work_dirs/ciis_21-2/20250310_130346/vis_data/20250310_130346.json',
    'work_dirs/ciis_21-2/20250310_130346/vis_data/20250310_130346_modified.json'
)