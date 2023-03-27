# task定义

1. 智能体数量；
2. 与jsbsim交互的变量（core/catalog.py中定义的量）；
3. 状态空间、动作空间；
4. get_obs()方法，封装jsbsim返回的变量值，并做了normalization；
5. normalize_action()，对policy计算的action进行normalization（其实就是做一步转化，将转化后的值送进jsbsim）；
6. 任务相关的reset和step方法；
7. 奖励、中止条件；
8. **管理baseline_agent**；