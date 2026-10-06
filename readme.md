# VF-CL

## 环境

```bash
pip install torch torchvision numpy scikit-learn
```

## V-LETO CL验证 (论文复现)

```bash
python -u main.py \
  --cl_method proto_evolve --ul_method retrain \
  --replay_mode prototype \
  --model_type small_cnn --num_parties 4 --aggregation sum \
  --custom_tasks "0,1,2|3,4,5|6,7|8,9" \
  --data cifar10 --num_classes 10 \
  --epochs_per_task 30 \
  --unlearn_after_tasks 99,99 --unlearn_classes "0;4" \
  --batch_size 64 --device cuda:0 --seed 42
```

结果: AVG=55.7% (论文报告52.26%)

## 全量Benchmark

```bash
python -u main.py --run_all \
  --replay_mode prototype \
  --model_type small_cnn --num_parties 4 --aggregation sum \
  --data cifar10 --num_tasks 5 --classes_per_task 2 --num_classes 10 \
  --epochs_per_task 30 --ul_epochs 5 \
  --unlearn_after_tasks 2,3 --unlearn_classes "0;4" \
  --batch_size 64 --device cuda:0 --seed 42
```

## Baseline列表

| CL方法 | UL方法 | 说明 |
|--------|--------|------|
| proto_evolve (V-LETO) | luv / ga / retrain | VFL持续学习SOTA |
| proto_aug (PASS+VFL) | luv | Prototype增强 |
| proto_fedspace (FedSpace+VFL) | luv | Prototype + 表征约束 |
| finetune | luv / ga | 无CL保护 (下界) |
| Oracle | - | 每步联合重训 (上界) |

## 输出

结果保存在 `./results/` 下，`benchmark_summary.json` 包含所有baseline对比。
