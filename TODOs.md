# 源代码仓库调整 
* config.py缺参数，71行增加： 
`group.add_argument("--n-render-rollout-threads", type=int, default=1,
    help="number of parallel envs for rendering, could only be set as 1 for some environments")`
* train/train_*.py脚本修改参数(wandb可以不用，check文件路径，)
* 增加脚本：render/render_control.py
* termination_conditions包中增加README.md


# 环境相关
* I9-10980xe处理器，18核36线程
* 96G内存
* 3090显卡

# 下一步工作
* 专家数据？
* step()返回的info()对象中增加误差信息