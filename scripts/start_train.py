import logging
from train.trainer import dpok, d3po, ddpo
from train.train import CurriculumTrainerArguments, DiffusionCurriculumTrainer
from transformers.hf_argparser import HfArgumentParser
from utils import setup_logger
import sys


def main():
    setup_logger(logging.INFO)
    parser = HfArgumentParser(CurriculumTrainerArguments)
    if sys.argv[-1].endswith(".yml") or sys.argv[-1].endswith(".yaml"):
        curriculum_args, *_ = parser.parse_yaml_file(sys.argv[-1], allow_extra_keys=True)
    else:
        curriculum_args, *_ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # 根据选择的RL算法选择相应的Config类
    if curriculum_args.rl_algorithm == "dpok":
        ConfigClass = dpok.Config
    elif curriculum_args.rl_algorithm == "d3po":
        ConfigClass = d3po.Config
    elif curriculum_args.rl_algorithm == "ddpo":
        ConfigClass = ddpo.Config
    else:
        raise ValueError(f"不支持的RL算法: {curriculum_args.rl_algorithm}，支持的算法有: ddpo, d3po, dpok")

    # 解析RL特定参数
    parser = HfArgumentParser([CurriculumTrainerArguments, ConfigClass])
    if sys.argv[-1].endswith(".yml") or sys.argv[-1].endswith(".yaml"):
        curriculum_args, rl_args = parser.parse_yaml_file(sys.argv[-1], True)
    else:
        curriculum_args, rl_args = parser.parse_args_into_dataclasses()

    trainer = DiffusionCurriculumTrainer(curriculum_args, rl_args)
    trainer.train()


if __name__ == "__main__":
    main()
