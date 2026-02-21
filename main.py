import subprocess
import sys
import os
import time
from pathlib import Path

def run_script(script_path: str):
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ Error executing {script_path}. Exiting.")
        sys.exit(result.returncode)
        
    print(f"\n✅ Finished {script_path} in {elapsed:.1f}s")
    print("-" * 60)

def main():
    # Ensure we run from the project root
    project_root = Path(__file__).parent.absolute()
    os.chdir(project_root)
    
    if not (project_root / "images" / "raw_").exists():
        print("⚠️ Warning: 'images/raw_' directory not found.")
        print("Please ensure your raw images are placed in 'images/raw_'.")
    
    scripts_to_run = [
        "scripts/phase1_illumination_flattening.py",
        "scripts/phase2_weld_roi_boundaries.py",
        "scripts/phase3_keyhole_module.py",
        "scripts/phase3_1_traverse_edges.py",
        "scripts/phase3_2_defect_masks.py",
        "scripts/phase4_classify_defects.py",
        "scripts/phase5_metrics_and_plots.py",
    ]
    
    total_start = time.time()
    for script in scripts_to_run:
        script_full_path = project_root / script
        if not script_full_path.exists():
            print(f"❌ Script not found: {script_full_path}")
            sys.exit(1)
            
        run_script(script)
        
    total_elapsed = time.time() - total_start
    print(f"\n🎉 Entire pipeline completed successfully in {total_elapsed:.1f}s.")

if __name__ == "__main__":
    main()
