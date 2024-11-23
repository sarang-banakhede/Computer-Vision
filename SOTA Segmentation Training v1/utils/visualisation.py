import os
import json
import matplotlib.pyplot as plt

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def plot_metric(train_metrics, test_metrics, metric_name, model_name, save_dir):
    train_epochs = [int(epoch.split('_')[1]) for epoch in train_metrics.keys()]
    
    train_fg = [train_metrics[f'epoch_{epoch}']['foreground'][metric_name] for epoch in train_epochs]
    train_bg = [train_metrics[f'epoch_{epoch}']['background'][metric_name] for epoch in train_epochs]
    test_fg = [test_metrics[f'epoch_{epoch}']['foreground'][metric_name] for epoch in train_epochs]
    test_bg = [test_metrics[f'epoch_{epoch}']['background'][metric_name] for epoch in train_epochs]
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_epochs, train_fg, label=f'Train FG {metric_name}', marker='o')
    plt.plot(train_epochs, train_bg, label=f'Train BG {metric_name}', marker='o')
    plt.plot(train_epochs, test_fg, label=f'Test FG {metric_name}', marker='x')
    plt.plot(train_epochs, test_bg, label=f'Test BG {metric_name}', marker='x')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} for {model_name}')
    plt.legend()
    plt.grid(True)
    
    if metric_name in ['dice_scores', 'recall', 'precision', 'f1', 'pixel_accuracies', 'iou_scores']:
        plt.ylim(0, 1)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{metric_name}.png'))
    plt.close()

def plot_all_metrics(train_metrics, test_metrics, model_name, save_dir):
    first_epoch = list(train_metrics.keys())[0]
    available_metrics = train_metrics[first_epoch]['foreground'].keys()
    
    for metric in available_metrics:
        plot_metric(train_metrics, test_metrics, metric, model_name, save_dir)

    epochs = [int(epoch.split('_')[1]) for epoch in train_metrics.keys()]
    train_loss = [train_metrics[f'epoch_{epoch}']['loss'] for epoch in epochs]
    test_loss = [test_metrics[f'epoch_{epoch}']['loss'] for epoch in epochs]
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, test_loss, label='Test Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss for {model_name}')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

def plot(result_folder, plots_folder):
    model_names = [name for name in os.listdir(result_folder) 
                  if os.path.isdir(os.path.join(result_folder, name))]

    for model_name in model_names:
        model_result_path = os.path.join(result_folder, model_name)
        train_json_path = os.path.join(model_result_path, 'train_results.json')
        test_json_path = os.path.join(model_result_path, 'test_results.json')
        
        try:
            train_metrics = load_json(train_json_path)
            test_metrics = load_json(test_json_path)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file: {e}")
            continue
        
        save_dir = os.path.join(plots_folder, model_name)
        plot_all_metrics(train_metrics, test_metrics, model_name, save_dir)
        print(f"Plots saved for {model_name}")

if __name__ == "__main__":
    pass