import os, json

MLE_DIR = "/mnt/sfs_turbo/exp_data/demo1bench/{exp_id}/prepared/baseline.json"
BASEDIR = "./data/demos_with_trajectory"

with open("/mnt/sfs_turbo/yujiezheng/mle-bench-log/aide-4o/statistic_models-gpt-4o-aide.json") as f:
    gt = json.load(f)

for filename in os.listdir(BASEDIR):
    exp_id, ext = os.path.splitext(filename)
    if ext != ".json":
        continue
    if os.path.exists(MLE_DIR.format(exp_id=exp_id)) and exp_id in gt:
        with open(MLE_DIR.format(exp_id=exp_id)) as f:
            bl = json.load(f)
        bl["baseline"]["gold"] = gt[exp_id]["gold_threshold"]
        bl["baseline"]["silver"] = gt[exp_id]["silver_threshold"]
        bl["baseline"]["bronze"] = gt[exp_id]["bronze_threshold"]
        bl["baseline"]["median"] = gt[exp_id]["median_threshold"]
        del bl["baseline"]["random"]
        del bl["name"]
        del bl["metric"]
        del bl["total_teams"]
        with open(os.path.join(BASEDIR, filename)) as f:
            old = json.load(f)
        old["comp_info"] = bl
        with open(os.path.join(BASEDIR, filename), 'w') as f:
            json.dump(old, f, ensure_ascii=False, indent=4)
