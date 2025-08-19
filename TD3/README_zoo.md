
### 训练模型

```bash
cd /Users/GuanJ1u/Code/GitHub_repo/DRL-robot-navigation2/TD3
./train_with_zoo.py --mode train --config ./configs/td3_velodyne.yaml
```

### 评估模型

```bash
./train_with_zoo.py --mode evaluate --model ./pytorch_models/zoo_models/td3/VelodyneEnv-v0_1.zip
```

### 超参数优化

```bash
./train_with_zoo.py --mode optimize --config ./configs/td3_velodyne.yaml
```

# 加载训练好的模型
model = TD3.load("pytorch_models/zoo_models/td3/VelodyneEnv-v0_1.zip")

# 进一步训练
model.learn(total_timesteps=100000)

# 保存模型
model.save("pytorch_models/td3_finetuned_model")
```