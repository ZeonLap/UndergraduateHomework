# 代码说明

- `layers.py`：修改了`Layer`的`_save_for_backward`方法，以便储存不止一个中间结果，加速反向传播的计算。
- `run_mlp.py`与`solve_net.py`：使用了`wandb`库进行数据可视化，并增加了训练时的可选择参数设置。
- `test.sh`与`test_hp.sh`：用于批量训练的脚本。