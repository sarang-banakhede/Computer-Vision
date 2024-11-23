import os
import yaml
from ultralytics import YOLO

def train_model(model_name, data_yaml, epochs, image_size, batch):

    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        name=f'{model_name}_custom'
    )

def main():

    data_yaml_file = 'config/Dataset.yaml'
    train_yaml_file = 'config/train.yaml'

    with open(train_yaml_file, 'r') as file:
        train_config = yaml.safe_load(file)
        
        for model in train_config['models']:
            for version in train_config['versions']:
                print(f"Starting training for {model}{version}")
                train_model(f'{model}{version}.pt',
                            data_yaml_file,
                            train_config['epochs'],
                            train_config['imgsz'],
                            train_config['batch'])
                print(f"Training for {model}{version} completed")
                print("-" * 50)

if __name__ == "__main__":
    main()
