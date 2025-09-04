# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class Visualizer:
    def __init__(self, map_layout):
        self.map_layout = map_layout
        self.rows = len(map_layout)
        self.cols = max(len(row) for row in map_layout if row)
        self.cmap = mcolors.ListedColormap(['white', 'black'])
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    # 함수 이름을 draw_grid -> draw_final_paths 로 변경하고 로직 수정
    def draw_final_paths(self, agents):
        self.ax.clear()
        
        # 맵 그리기
        grid = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(len(self.map_layout[r])):
                if self.map_layout[r][c] == '#':
                    grid[r, c] = 1
        self.ax.imshow(grid, cmap=self.cmap, interpolation='nearest')
        
        # 에이전트들의 최종 경로, 시작/도착점 그리기
        agent_colors = plt.cm.jet(np.linspace(0, 1, len(agents)))
        for i, agent in enumerate(agents):
            color = agent_colors[i]
            
            # 전체 경로
            path_x = [p[1] for p in agent.state["path"]]
            path_y = [p[0] for p in agent.state["path"]]
            self.ax.plot(path_x, path_y, color=color, linestyle='-', marker='o', markersize=4, label=f'Agent_{i}')

            # 시작점
            start = agent.initial_start_pos
            self.ax.plot(start[1], start[0], 'o', color='green', markersize=10)
            
            # 도착점
            end = agent.initial_end_pos
            self.ax.plot(end[1], end[0], '*', color='red', markersize=15)

        self.ax.set_xticks(np.arange(-.5, self.cols, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.rows, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_title("Final Learned Paths")

        handles, labels = self.ax.get_legend_handles_labels()

        # Figure 바깥 오른쪽에 legend 배치
        self.fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title="Agents"
        )
        self.fig.tight_layout(rect=[0, 0, 0.85, 1])  # 오른쪽 여백 확보


        
    def show(self):
        plt.show()