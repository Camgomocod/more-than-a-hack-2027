"""Master Runner for Phase 1 EDA scripts."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "eda_01_ground_truth.py",
    "eda_02_ball_occlusions.py",
    "eda_03_smart_tagging.py"
]

def main():
    print("======================================================================")
    print("Executing All Phase 1 EDA Integrables")
    print("======================================================================")
    python_bin = sys.executable

    for script in SCRIPTS:
        script_path = os.path.join(HERE, script)
        print(f"\n--> Running {script} ...")
        cmd = [python_bin, script_path]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] {script} failed with returncode {res.returncode}")
            sys.exit(res.returncode)

    print("\n======================================================================")
    print("All Phase 1 EDA scripts completed successfully!")
    print("Generated Reports:")
    reports_dir = os.path.join(HERE, "reports")
    for r in sorted(os.listdir(reports_dir)):
        if r.endswith(".md"):
            print(f"  - eda/reports/{r}")
    print("======================================================================")

if __name__ == "__main__":
    main()
