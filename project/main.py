import os
import subprocess

# Ensure output folders exist
os.makedirs("outputs/descriptive_charts", exist_ok=True)
os.makedirs("outputs/predictive_results", exist_ok=True)
os.makedirs("outputs/prescriptive_results", exist_ok=True)

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        print(f"\n=== Running {script_name} ===")
        subprocess.run(["python", script_name], check=True)
        print(f"=== {script_name} finished successfully ===")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        exit(1)

def main():
    print("Starting End-to-End Data Analytics Pipeline...")

    # 1️⃣ Descriptive & Diagnostic Analytics
    run_script("notebooks/descriptive_diagnostic.py")

    # 2️⃣ Predictive Analytics
    run_script("notebooks/predictive.py")

    # 3️⃣ Prescriptive Analytics
    run_script("notebooks/prescriptive.py")

    # 4️⃣ Generate Summary
    run_script("notebooks/summary.py")

    print("\nAll analytics phases completed. Check 'outputs/' folder for results.")

if __name__ == "__main__":
    main()
