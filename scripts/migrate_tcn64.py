"""离线迁移完整 IQL32 权重，不覆盖原模型；不继承旧优化器与评估排名。"""
import argparse
import copy
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soku_iql import checkpoint
from soku_iql.config import load
from soku_iql.learner import Learner
from soku_iql.models import expand_tcn32_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/iql_suika.yaml')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('迁移目标已存在，禁止覆盖')
    package, digest = checkpoint.load_source(args.source)
    if package['config']['model']['temporal_mode'] != 'tcn':
        raise ValueError('源模型必须是当前 Joint144 IQL TCN32')
    config = load(args.config, package['bc_config'])
    if config['training']['burn_in'] != 63:
        raise ValueError('目标配置需要 burn_in=63')
    original_device = config['training']['device']
    cpu_config = copy.deepcopy(config)
    cpu_config['training']['device'] = 'cpu'
    torch.manual_seed(config['seed'])
    learner = Learner(cpu_config)
    for name in ('actor', 'q1', 'q2', 'value', 'target_q1', 'target_q2'):
        prefix = name + '.'
        state = {k[len(prefix):]: v for k, v in package['networks'].items() if k.startswith(prefix)}
        expand_tcn32_weights(getattr(learner.networks, name), state)
    # 旧目标参数保留，新目标块与其 online 对应块一致，避免随机目标偏差。
    for target, online in ((learner.networks.target_q1, learner.networks.q1), (learner.networks.target_q2, learner.networks.q2)):
        target.tcn.blocks[4].load_state_dict(online.tcn.blocks[4].state_dict())
    learner.config['training']['device'] = original_device
    provenance = {**package['provenance'], 'migration': 'tcn32_to_tcn64', 'migration_source': str(Path(args.source).resolve()),
                  'migration_sha256': digest, 'migration_source_step': package['step']}
    checkpoint.save(output, learner, package['bc_config'], package['normalization'], package['split_hash'],
                    0, 0, 0, float('inf'), provenance)
    print(f'已迁移至 {output}；新增块随机初始化，优化器/计数/排名重置；需重新训练评估。原文件未修改。')


if __name__ == '__main__':
    main()
