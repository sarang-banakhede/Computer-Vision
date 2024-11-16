import os
import subprocess

def launch_tensorboard(base_dir="runs"):
    os.chdir(base_dir)
    model_dirs = [d for d in os.listdir() if os.path.isdir(d)]
    
    if not model_dirs:
        print("No model directories found.")
        return
    
    print("Available model directories:")
    for i, dir_name in enumerate(model_dirs):
        print(f"{i + 1}: {dir_name}")
    
    try:
        selection = int(input("Enter the number of the model directory to use: "))
        if 1 <= selection <= len(model_dirs):
            selected_dir = model_dirs[selection - 1]
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    print(f"Launching TensorBoard for model directory: {selected_dir}")
    subprocess.run(["tensorboard", "--logdir=" + selected_dir])

if __name__ == "__main__":
    launch_tensorboard()
