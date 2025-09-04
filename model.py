import math, random, numpy as np
from xdevs.models import Atomic, Port

DIRS_8 = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1),(0,0)]

class Agent(Atomic):
    def __init__(self, agent_id, start_pos, end_pos, building, max_steps=1000, mode='train'):
        super().__init__(f"Agent_{agent_id}")
        self.building = building

        # Ports
        self.in_tick = Port(bool, "in_tick");      self.add_in_port(self.in_tick)
        self.in_obs  = Port(tuple, "in_obs");      self.add_in_port(self.in_obs)   # (r,c,gr,gc,patch or misc)
        self.in_fb   = Port(dict,  "in_fb");       self.add_in_port(self.in_fb)    # {"reward":float, "done":bool}
        self.out_act = Port(int,   "out_action");  self.add_out_port(self.out_act)

        # RL/Q
        self.agent_id = agent_id
        self.mode = mode
        self.max_steps = max_steps
        self.actions = DIRS_8
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01

        self.ep_reward = 0.0
        self.invalid_moves = 0     # 벽/불가 이동 시 카운트
        self.reached_goal = False  # 도착 성공 여부
        
        # sim state
        self.start0 = tuple(start_pos)
        self.goal0  = tuple(end_pos)
        self.pos = self.start0
        self.goal = self.goal0
        self.other_agents = []      # 안전: 기본값
        self.episode_done = False
        self.step_count = 0
        self.path = [self.pos]

        # Q-table: key = (pos, goal) → [Q(a)]
        self.q_table = {}

        # event timing
        self.sigma = math.inf
        self._ticked = False

        # for TD update
        self.last_state = None      # (pos, goal)
        self.last_action = None     # action index
        self._last_obs = None
        self.episode_num = 0

        self.initial_start_pos = self.start0
        self.initial_end_pos   = self.goal0

        self.state = {
            "mode": self.mode,
            "path": self.path,
            "step_count": self.step_count,
            "ep_reward": self.ep_reward,
            "invalid_moves": self.invalid_moves,
            "reached_goal": self.reached_goal,
        }

    def set_other_agents(self, agents):
        self.other_agents = list(agents)

    def ta(self): 
        return self.sigma

    def initialize(self):
        self.sigma = math.inf

    def reset(self):
        self.pos = self.start0
        self.goal = self.goal0
        self.episode_done = False
        self.step_count = 0
        self.path = [self.pos]
        self.last_state = None
        self.last_action = None
        if self.mode == 'train' and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        # 에피소드 통계 초기화
        self.ep_reward = 0.0
        self.invalid_moves = 0
        self.reached_goal = False
        # state 동기화
        self.state.update({
            "path": self.path,
            "step_count": self.step_count,
            "ep_reward": self.ep_reward,
            "invalid_moves": self.invalid_moves,
            "reached_goal": self.reached_goal,
        })
        
    def set_episode_num(self, n): 
        self.episode_num = n

    # -------- Q helpers --------
    def _state_key(self, pos=None, goal=None):
        pos = pos if pos is not None else self.pos
        goal = goal if goal is not None else self.goal
        return (tuple(pos), tuple(goal))

    def _ensure_row(self, key):
        if key not in self.q_table:
            self.q_table[key] = [0.0]*len(self.actions)

    def get_q_value(self, key, a_idx):
        self._ensure_row(key)
        return self.q_table[key][a_idx]

    def set_q_value(self, key, a_idx, value):
        self._ensure_row(key)
        self.q_table[key][a_idx] = value

    def choose_action(self, key):
        self._ensure_row(key)
        if self.mode == 'train' and random.random() < self.epsilon:
            return random.randrange(len(self.actions))
        qs = self.q_table[key]
        return int(np.argmax(qs))

    # -------- DEVS transitions --------
    def deltext(self, e):
        # tick
        if not self.in_tick.empty():
            self._ticked = True
            self.in_tick.clear()

        # obs: (r,c,gr,gc,patch/anything)
        if not self.in_obs.empty():
            obs = None
            for o in self.in_obs.values:
                obs = o
            self.in_obs.clear()
            self._last_obs = obs

            # 관측으로 pos/goal 갱신 (World가 현재 위치를 관측으로 준다고 가정)
            if obs is not None:
                r, c, gr, gc = obs[0], obs[1], obs[2], obs[3]
                self.pos = (int(r), int(c))
                self.goal = (int(gr), int(gc))

        # feedback: {"reward": float, "done": bool}
        if not self.in_fb.empty():
            fb = None
            for f in self.in_fb.values:
                fb = f
            self.in_fb.clear()
            reward = float(fb.get("reward", 0.0))
            done   = bool(fb.get("done", False))

            # 누적 보상
            self.ep_reward += reward
            

            # TD(0) update using (last_state, last_action) → (current state from obs)
            if self.mode == 'train' and (self.last_state is not None) and (self.last_action is not None):
                s = self.last_state
                a = self.last_action
                s_next = self._state_key(self.pos, self.goal)
                self._ensure_row(s_next)
                target = reward + (0.0 if done else self.gamma * max(self.q_table[s_next]))
                new_q = (1 - self.alpha) * self.get_q_value(s, a) + self.alpha * target
                self.set_q_value(s, a, new_q)

            self.episode_done = done
            if done:
                # 도착 여부: 현재 pos == goal(혹은 fb에 success 플래그가 있다면 그걸 사용)
                self.reached_goal = (self.pos == self.goal)

            if not self.path or self.path[-1] != self.pos:
                self.path.append(self.pos)

            # 스텝 카운트 (feedback 도착 = 1 step 진행)
            if not self.episode_done:
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    self.episode_done = True

            self.state.update({
                "path": self.path,
                "step_count": self.step_count,
                "ep_reward": self.ep_reward,
                "invalid_moves": self.invalid_moves,
                "reached_goal": self.reached_goal,
            })

        # 액션을 낼 조건: tick + obs + not done
        if self._ticked and (self._last_obs is not None) and not self.episode_done:
            self.sigma = 0.0
        else:
            self.sigma = math.inf

    def lambdaf(self):
        # 현재 상태에서 액션 선택 → cache
        s = self._state_key(self.pos, self.goal)
        a_idx = self.choose_action(s)
        self.last_state = s
        self.last_action = a_idx
        self.out_act.add(a_idx)

    def deltint(self):
        self._ticked = False
        self.sigma = math.inf

    def deltcon(self, e):
        self.deltint(); 
        self.deltext(0.0)

    def exit(self): 
        pass

    def propose_action(self):
        # 현재 상태 → 행동 인덱스
        s = self._state_key(self.pos, self.goal)
        self._ensure_row(s)
        self.last_state = s

        # epsilon-greedy
        if self.mode == 'train' and random.random() < self.epsilon:
            a_idx = random.randrange(len(self.actions))
        else:
            qs = self.q_table[s]
            a_idx = int(np.argmax(qs))

        self.last_action = a_idx
        return a_idx

    def learn_from(self, reward, done, next_pos):
        # 1) 보상 누적
        self.ep_reward += float(reward)

        # 2) Q 업데이트
        if self.mode == 'train' and self.last_state is not None and self.last_action is not None:
            s_next = self._state_key(next_pos, self.goal)
            self._ensure_row(s_next)
            max_next = max(self.q_table[s_next])
            target = reward + (0.0 if done else self.gamma * max_next)
            old_q = self.get_q_value(self.last_state, self.last_action)
            new_q = (1 - self.alpha) * old_q + self.alpha * target
            self.set_q_value(self.last_state, self.last_action, new_q)

        # invalid 판정: next_pos가 '제자리'이고 보상이 음수일 때 =
        prev_pos = self.pos
        if tuple(next_pos) == prev_pos and reward < 0:
            self.invalid_moves += 1

        # 3) 좌표/경로/스텝 갱신
        self.pos = tuple(next_pos)
        if not self.path or self.path[-1] != self.pos:
            self.path.append(self.pos)
        self.step_count += 1
        if done and self.pos == self.goal:
            self.reached_goal = True

        # 4) state 동기화 (로그용)
        self.state.update({
            "step_count": self.step_count,
            "ep_reward": self.ep_reward,
            "reached_goal": self.reached_goal,
            "path": self.path,
            "invalid_moves": self.invalid_moves,
        })
