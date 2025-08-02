import yaml
import sys
import os

def flatten_dict(d, parent_key='', sep='__'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

CONFIG_FILE = "/opt/project/config.yaml"

if not os.path.exists(CONFIG_FILE):
    sys.exit(0)

cfg = {}
try:
    with open(CONFIG_FILE, 'r') as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"# Error reading config: {e}", file=sys.stderr)
    sys.exit(1)

airflow_config = cfg.get('airflow', {})
flat_config = flatten_dict(airflow_config)

for key, value in flat_config.items():
    var_name = f'AIRFLOW__{key.upper()}'
    value_str = str(value).replace("'", "'\\''")
    print(f"export {var_name}='{value_str}'")
