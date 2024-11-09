from utility.utils import read_yaml

tomato_config_file = read_yaml('config/config_file.yaml')

class TomatoConfig:
    train_path= tomato_config_file['train_path']
    test_path= tomato_config_file['test_path']
    image_size= tomato_config_file['image_size']
    channels= tomato_config_file['channels']
    epochs= tomato_config_file['epochs']
    batch_size= tomato_config_file['batch_size']
    learning_rate= tomato_config_file['learning_rate']
    accuracy_task= tomato_config_file['accuracy_task']
    accuracy_avg= tomato_config_file['accuracy_avg']
    optim_step_size = tomato_config_file['optim_step_size']
    output_dir= tomato_config_file['output_dir']
    output_save_model= tomato_config_file['output_save_model']