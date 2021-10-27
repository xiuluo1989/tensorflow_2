import gym

def demo_gym():
    env = gym.make('CartPole-v1')  # 实例化一个游戏环境，参数为游戏名称
    state = env.reset()  # 初始化环境，获得初始状态
    while True:
        env.render()  # 对当前帧进行渲染，绘图到屏幕
        # action = model.predict()  # 假设我们有一个训练好的模型，能够通过当前状态预测出这时应该进行的动作
        # next_state, reward, done, info = env.step(action)  # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息9.
        # if done:  #	如果游戏结束则退出循环
        #   break

if __name__ == "__main__":
    demo_gym()
    pass
