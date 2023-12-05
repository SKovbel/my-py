import time


class Stats:
    STEP = 10
    WINDOW = 100

    def __init__(self):
        self.start = int(time.time() * 1000)
        self.count = 0
        self.engines = {}
        self.statuses = []

    def add(self, game):
        won = game.status()
        code_x = game.engine_X.code
        code_0 = game.engine_0.code

        self.count += 1

        stats = {
            code_x: {'g': 1, 'w': 0, 'd': 0, 'l': 0},
            code_0: {'g': 1, 'w': 0, 'd': 0, 'l': 0}
        }

        self.engines[code_x] = code_x
        self.engines[code_0] = code_0

        if won:
            win, lost = (code_x, code_0) if won == 1 else (code_0, code_x)
            stats[win]['w'] = 1
            stats[lost]['l'] = 1
        else:
            stats[code_x]['d'] = 1
            stats[code_0]['d'] = 1

        self.statuses.append(stats)
        self.pop()

        if self.count % self.STEP == 0:
            self.print()

    def pop(self):
        if len(self.statuses) > self.WINDOW:
            self.statuses.pop(0)

    def header(self):
        print("Stats config: STEP =", self.STEP, "WINDOW =", self.WINDOW)

    def print(self):
        if self.count < 2 * self.STEP:
            self.header()

        loop = int(time.time() * 1000)
        print(f"\nGame #{self.count} | {((loop - self.start) / 1000)}sec", end=' |  ')
        for engine in self.engines.keys():
            cnt = win = draw = lost = 0
            for row in self.statuses:
                if engine in row:
                    cnt += row[engine]['g']
                    win += row[engine]['w']
                    draw += row[engine]['d']
                    lost += row[engine]['l']

            win_rate = '-' if cnt == 0 else str(round(100 * win / cnt))
            draw_rate = '-' if cnt == 0 else str(round(100 * draw / cnt))
            lost_rate = '-' if cnt == 0 else str(round(100 * lost / cnt))
            print(f"{engine.upper()}",
                  f"games {cnt} |",
                  f"{win} wins {win_rate}% |",
                  f"{draw} draws {draw_rate}% |",
                  f"{lost} losts {lost_rate}%",
                  end='  |  ')
