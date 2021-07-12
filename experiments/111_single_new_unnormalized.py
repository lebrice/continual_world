from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps': int(4e6),
  'replay_size': int(1e6),
  'batch_size': 256,
  'hidden_sizes': [256, 256, 256, 256],
  'activation': 'lrelu',
  'use_layer_norm': True,
  'lr': 3e-4,
  'alpha': 0.4,
}

params_grid = {
  'seed': list(range(5)),
  'task': list(range(50)),
}
name = globals()['script'][:-3]

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script='python3 run_single.py',
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
