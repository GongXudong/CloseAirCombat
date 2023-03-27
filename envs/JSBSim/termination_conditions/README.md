# unreach_heading.py

任务开始时，预设heading_check_time（ExtraCatalog.heading_check_time）。
任务执行（step）过程中，不断检查是否达到该时间。
如果达到了，检查heading的误差是否小于10度。
如果是，则随机重置目标heading、目标高度、和目标速度，设置新的heading_check_time，任务继续执行。
如果不是，任务结束（done=True）。

# timeout

step数超过配置文件中的max_steps，就结束任务（done=True）。

# low_altitude

飞机高度低于配置文件中的altitude_limit，就结束任务（done=True）。

# extreme_state

检测jsbsim中的属性detect_extreme_state（ExtraCatalog.detect_extreme_state），
速度过快，roll速度过快，高度过高，加速度过大，就结束任务（done=True）。

# overload

x,y,z三向加速度，如果有一个比配置文件中设定的值大，就结束任务（done=True）。

# safe_return

我机被击毁或坠机，或者所有敌机被击毁，结束任务。