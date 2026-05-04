import os
import subprocess
import sys

def run_app():
    """
    Helper script to launch the StressIntel PRO Streamlit application.
    """
    print("\n" + "="*50)
    print("🚀 LAUNCHING STRESSINTEL PRO (RESEARCH EDITION)")
    print("="*50)
    print("\nApplying Research-Grade Theme...")
    print("Loading AI Models and XAI Explainers...")
    
    app_path = os.path.join("src", "ui", "app_streamlit.py")
    
    try:
        # Launching the Flask application
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 System shutdown successfully.")
    except Exception as e:
        print(f"\n❌ Error launching app: {e}")
        print("\nPlease ensure dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    run_app()
