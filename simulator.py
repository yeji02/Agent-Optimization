# simulator.py
from xdevs.models import Coupled
from building import Building
from model import Agent  # <- 파일명이 agent.py 라면 이렇게

class EvacuationSimulator(Coupled):
    def __init__(self, map_layout, agent_configs, max_steps):
        """
        Coupled 모델 초기화
        :param map_layout: 건물 2D 구조 (list[str] 또는 list[list[str]])
        :param agent_configs: [{"start":(r,c), "end":(r,c)}, ...]
        :param max_steps: 에이전트 에피소드 최대 스텝
        """
        super().__init__("EvacuationSimulator")

        # 1) 환경(정적) 모델
        self.building = Building(map_layout=map_layout)
        self.add_component(self.building)

        # 2) 에이전트들
        self.agents = []
        for i, cfg in enumerate(agent_configs):
            ag = Agent(
                agent_id=i,
                start_pos=cfg["start"],
                end_pos=cfg["end"],
                building=self.building,
                max_steps=max_steps
            )
            self.agents.append(ag)
            self.add_component(ag)

        # 3) 충돌 체크 위해 서로 참조
        for ag in self.agents:
            others = [o for o in self.agents if o is not ag]
            ag.set_other_agents(others)


    def run_episode(self, train=True):
        # 초기화
        for ag in self.agents:
            ag.mode = 'train' if train else 'test'
            ag.reset()

        done_flags = [False] * len(self.agents)
        max_steps = max(a.max_steps for a in self.agents) if self.agents else 0

        BLOCK_PENALTY   = -1.0   # 충돌로 인해 그 스텝에 못 움직였을 때의 추가 패널티

        for _ in range(max_steps):
            # 1) 행동 수집 
            #    wanted_pos, base_reward, reached는 building.step_outcome이 결정
            intents = []  # [(i, cur, want, r0, reached, a_idx)]
            for i, (ag, done) in enumerate(zip(self.agents, done_flags)):
                if done:
                    intents.append((i, ag.pos, ag.pos, 0.0, False, None))
                    continue
                a_idx = ag.propose_action() # 행동 수집
                action = ag.actions[a_idx]
                want, r0, reached = self.building.step_outcome(ag.pos, action, ag.goal0) # 이동 결과 반환
                intents.append((i, ag.pos, want, r0, reached, a_idx))

            # 2) 동시 충돌 탐지 (한 타임스텝 한정)
            # 2-1) vertex: 같은 wanted_pos로 2명 이상
            target_map = {}
            for i, cur, want, r0, reached, a_idx in intents:
                target_map.setdefault(tuple(want), []).append(i)
            vertex_conflict = {pos for pos, ids in target_map.items() if len(ids) >= 2}

            # 2-2) edge swap: i는 j의 cur로, j는 i의 cur로 동시에 이동
            current_pos = {i: tuple(cur) for i, cur, want, r0, reached, a_idx in intents}
            wanted_pos  = {i: tuple(want) for i, cur, want, r0, reached, a_idx in intents}
            swap_block = set()
            for i in current_pos:
                for j in current_pos:
                    if i >= j: 
                        continue
                    if wanted_pos[i] == current_pos[j] and wanted_pos[j] == current_pos[i]:
                        swap_block.add(i); swap_block.add(j)

            # 3) 충돌 해소: 동시 충돌만 막고, 시간차 이동은 허용
            for i, (ag, done) in enumerate(zip(self.agents, done_flags)):
                if done:
                    continue

                cur, want, r0, reached = intents[i][1], intents[i][2], intents[i][3], intents[i][4]

                if tuple(want) in vertex_conflict or i in swap_block:
                    # 동시 충돌 → 제자리 + 소액 패널티, done=False
                    final_pos  = cur
                    final_rew  = r0 + BLOCK_PENALTY
                    final_done = False
                else:
                    # 정상 진행 (벽/맵 밖은 building.step_outcome이 이미 제자리+패널티로 반환)
                    final_pos  = want
                    final_rew  = r0
                    final_done = reached

                ag.learn_from(final_rew, final_done, final_pos)
                if final_done:
                    done_flags[i] = True

            if all(done_flags):
                break

    def get_agents(self):
        return self.agents
