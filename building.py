# building.py
import math
from xdevs.models import Atomic

class Building(Atomic):
    def __init__(self, map_layout, name="Building"):
        super(Building, self).__init__(name)
        self.state = {
            "map": [list(row) for row in map_layout],
            "rows": len(map_layout),
            "cols": len(map_layout[0])
        }
        # xDEVS 내부시간
        self.sigma = math.inf

    def initialize(self):
        # 정적 모델
        self.sigma = math.inf

    def ta(self):
        # 다음 내부 이벤트까지의 시간
        return self.sigma

    def lambdaf(self):
        # 출력 없음(정적 유틸리티)
        pass

    def deltint(self):
        # 내부 전이 없음
        self.sigma = math.inf

    def deltext(self, e):
        # 외부 입력 처리 없음(정적 유틸리티)
        # e: 경과시간. 필요시 내부 clock 갱신에 사용 가능
        pass

    def deltcon(self, e):
        # 동시 전이 규약: 내부 → 외부
        self.deltint()
        self.deltext(0.0)

    def exit(self):
        pass

    def is_valid_pos(self, pos):
        x, y = pos
        if 0 <= x < self.state["rows"] and 0 <= y < self.state["cols"]:
            return self.state["map"][x][y] != '#'
        return False

    def step_outcome(self, curr_pos, action, goal):
        """한 스텝 전이: (다음위치, 보상, 종료여부) 반환
           - 벽/맵 밖: 패널티 주고 제자리(원위치), done=False
           - 목적지 도착: 큰 보상, done=True
           - 그 외: 한 칸 이동, 작은 시간 패널티
        """
        dr, dc = action
        nr, nc = curr_pos[0] + dr, curr_pos[1] + dc

        # 벽/맵 밖 → 원위치 복귀 + 패널티
        if not self.is_valid_pos((nr, nc)):
            return curr_pos, -5.0, False 

        # 정상 이동
        next_pos = (nr, nc)
        if next_pos == goal:
            return next_pos, +100.0, True

        return next_pos, -1.0, False