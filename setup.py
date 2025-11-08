import subprocess
import os

# Create virtual environment
subprocess.run(["python", "-m", "venv", "trelisEnv"])

# Activate virtual environment (This step is intended for use outside of Jupyter)
# For Jupyter, we skip activation and directly install packages in the next steps

# Install ipykernel (This may not work as intended because the environment is not activated)
subprocess.run(["pip", "install", "ipykernel"])

# Install Jupyter kernel (This installs the kernel but doesn't affect the current Jupyter session)
subprocess.run(["python", "-m", "ipykernel", "install", "--user", "--name=trelisEnv"])
