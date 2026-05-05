import subprocess
import sys
import os

def launch_app():
    """
    Launches the StressIntel PRO Flask application.
    """
    print("🚀 Initializing StressIntel PRO Institutional System...")
    print("📂 Project: Minor Project Team 32")
    print("🔍 Model: Research-Grade XGBoost Ensemble")
    print("--------------------------------------------------")
    
    # Path to the Flask app
    app_path = os.path.join(os.getcwd(), "app.py")
    
    if not os.path.exists(app_path):
        print(f"❌ Error: app.py not found at {app_path}")
        return

    # Use the current Python interpreter
    python_exe = sys.executable
    
    try:
        # Run the Flask app
        subprocess.run([python_exe, app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 System shutdown by researcher.")
    except Exception as e:
        print(f"❌ Critical System Error: {e}")

if __name__ == "__main__":
    launch_app()
