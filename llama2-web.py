import subprocess
import multiprocessing
import time
import subprocess
import sys

# --- Veryfing Installation ---
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import zmq
except ImportError:
    print("zmq not found. Installing...")
    install("zmq")

try:
    import streamlit
except ImportError:
    print("streamlit not found. Installing...")
    install("streamlit")

def run_back():
    subprocess.call(["python", "model-back.py"])

def run_front():
    time.sleep(10)  # Wait for 10 seconds
    subprocess.call(["streamlit", "run", "model-front.py"])

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_back)
    p1.start()

    p2 = multiprocessing.Process(target=run_front)
    p2.start()

    p1.join()
    p2.join()
