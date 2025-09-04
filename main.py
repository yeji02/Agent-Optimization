# main.py
# Agent\Scripts\Activate
import time
import csv, os
import matplotlib.pyplot as plt

from simulator import EvacuationSimulator
from visualizer import Visualizer  
from statistics import mean

LOG_PATH = "train_log.csv"
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)
with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["episode","avg_steps","avg_reward","success_rate","avg_invalid","epsilon"])

logs = []   

HAS_VIS = True

MAP = [
    ".........................",
    "..####....................",
    "..........................",
    "......#.....#####.........",
    ".....###..................",
    "....######................",
    ".............###..........",
    ".....###..................",
    "..........####............",
    ".....................###..",
    "....####..................",
    "..............#####.......",
    "....................###...",
    "....###...................",
    ".........................",
]

AGENT_CONFIGS = [
    {"start": (0, 0),   "end": (14, 23)},
    {"start": (1, 24),  "end": (14, 0)},
    {"start": (2, 0),   "end": (10, 20)},
    {"start": (3, 24),  "end": (12, 5)},
    {"start": (4, 0),   "end": (0, 15)},
    {"start": (5, 24),  "end": (13, 8)},
    {"start": (6, 0),   "end": (1, 18)},
    {"start": (7, 24),  "end": (3, 3)},
    {"start": (8, 0),   "end": (11, 22)},
    {"start": (14, 24), "end": (0, 5)},
]


MAX_STEPS_PER_EPISODE = 500
NUM_EPISODES = 1500
EPSILON_INIT = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

if __name__ == "__main__":
    print(f"--- START TRAINING FOR {len(AGENT_CONFIGS)} AGENTS ---")
    start_time = time.time()

    # 시뮬레이터(학습용) 구성
    sim_model = EvacuationSimulator(
        map_layout=MAP,
        agent_configs=AGENT_CONFIGS,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # 각 에이전트별 학습 Q-Table 저장소
    learned_q_tables = [{} for _ in range(len(AGENT_CONFIGS))]

    epsilon = EPSILON_INIT

    for episode in range(NUM_EPISODES):
        # 에피소드 설정/리셋
        for ag in sim_model.get_agents():
            ag.reset()
            ag.set_episode_num(episode)
            # Agent 내부 epsilon 사용 시, 외부에서 주입
            ag.epsilon = epsilon
            ag.min_epsilon = MIN_EPSILON
            ag.epsilon_decay = EPSILON_DECAY
            ag.mode = "train"   

        sim_model.run_episode(train=True) 

        # 학습된 Q-Table 백업
        for i, ag in enumerate(sim_model.get_agents()):
            learned_q_tables[i] = dict(ag.q_table)  

        # Epsilon 감소
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            if epsilon < MIN_EPSILON:
                epsilon = MIN_EPSILON

            
        # === 학습 for episode in range(NUM_EPISODES): 루프의 '에피소드 종료' 직후 ===
        agents   = sim_model.get_agents()
        steps    = [ag.state["step_count"]    for ag in agents]
        rewards  = [ag.state["ep_reward"]     for ag in agents]
        success  = [1 if ag.state["reached_goal"] else 0 for ag in agents]
        invalids = [ag.state["invalid_moves"] for ag in agents]
        eps      = agents[0].epsilon if agents else 0.0

        row = {
            "episode": episode,
            "avg_steps":   float(mean(steps))   if steps   else 0.0,
            "avg_reward":  float(mean(rewards)) if rewards else 0.0,
            "success_rate":float(mean(success)) if success else 0.0,
            "avg_invalid": float(mean(invalids))if invalids else 0.0,
            "epsilon":     float(eps)
        }
        logs.append(row)

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row["episode"], f'{row["avg_steps"]:.3f}', f'{row["avg_reward"]:.3f}',
                        f'{row["success_rate"]:.3f}', f'{row["avg_invalid"]:.3f}', f'{row["epsilon"]:.5f}'])

        # 50 에피소드마다 콘솔 요약
        if (episode+1) % 50 == 0:
            print(f'[EP {episode+1:04d}] step={row["avg_steps"]:.1f} | '
                f'r={row["avg_reward"]:+.1f} | succ={row["success_rate"]*100:.1f}% | '
                f'inv={row["avg_invalid"]:.2f} | eps={row["epsilon"]:.3f}')

    end_time = time.time()
    print(f"--- TRAINING FINISHED in {end_time - start_time:.2f} seconds ---")

    # --- 테스트 ---
    print("\n--- START TESTING ---")
    final_test_model = EvacuationSimulator(
        map_layout=MAP,
        agent_configs=AGENT_CONFIGS,
        max_steps=MAX_STEPS_PER_EPISODE,
    )
    test_agents = final_test_model.get_agents()
    # 학습된 Q-Table 주입 + 탐색 끔
    for i, agent in enumerate(test_agents):
        agent.q_table = dict(learned_q_tables[i]) 
        agent.epsilon = 0.0  # 탐색 비활성화
        agent.state["mode"] = "test"
        agent.reset()

    final_test_model.run_episode(train=False) 
    print("--- TESTING FINISHED ---")

    #학습 완료 시각화
    if HAS_VIS:
        vis = Visualizer(MAP)
        vis.draw_final_paths(test_agents)
        for i, agent in enumerate(test_agents):
            print(f"Agent {i} Path Length: {agent.state['step_count']}")
        vis.show()
    else:
        for i, agent in enumerate(test_agents):
            print(f"Agent {i} Path Length: {agent.state['step_count']}")

    #학습 과정 시각화
    def moving_avg(xs, k=25):
        if k <= 1: return xs
        out, acc = [], 0.0
        for i, v in enumerate(xs):
            acc += v
            if i >= k: acc -= xs[i-k]
            out.append(acc / min(i+1, k))
        return out

    ep   = [r["episode"] for r in logs]
    rwd  = moving_avg([r["avg_reward"]   for r in logs], k=25)
    succ = moving_avg([r["success_rate"] for r in logs], k=25)
    #eps  = [r["epsilon"] for r in logs]

    plt.figure(figsize=(11,6))
    plt.plot(ep, rwd,  label="Avg reward (MA-25)")
    plt.plot(ep, [s*100 for s in succ], label="Success rate (%) (MA-25)")
    #plt.plot(ep, [e*100 for e in eps],  label="Epsilon (%)")
    plt.xlabel("Episode"); plt.ylabel("Value")
    plt.title("Training progress")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show()
