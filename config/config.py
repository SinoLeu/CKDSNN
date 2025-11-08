import yaml
from argparse import Namespace

def load_config_from_yaml(file_path):
    """
    从 YAML 文件加载配置并返回一个 Namespace 对象。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    
    # 将字典转换为 Namespace 对象
    return Namespace(**config_dict)

# 替换原来的 parse_args 函数
def parse_args_yml(config_file):
    """
    从 YAML 文件加载配置。
    """
    # config_file = "config.yml"  # 指定 YAML 配置文件路径
    return load_config_from_yaml(config_file)

# 测试代码
# if __name__ == "__main__":
#     args = parse_args("config/train_resnet.yml")
#     print(args.loss_param)
#     print("Batch Size:", args.batch_size)
#     print("Learning Rate:", args.learning_rate)
#     print("ResNet Scale:", args.resnet_scale)
#     print("Is Distributed:", args.is_distributed)
#     print("Is Transfer:", args.is_transfer)