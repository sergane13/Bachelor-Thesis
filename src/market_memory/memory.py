import heapq

class Memory:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.short = []
        self.long = []

    def add_short_individual(self, ind, fitness):
        heapq.heappush(self.short, (-fitness, ind))
        if len(self.short) > self.max_size:
            heapq.heappop(self.short)

    def add_long_individual(self, ind, fitness):
        heapq.heappush(self.long, (-fitness, ind))
        if len(self.long) > self.max_size:
            heapq.heappop(self.long)

    def get_top_short(self):
        return [ind for _, ind in sorted(self.short, reverse=True)]

    def get_top_long(self):
        return [ind for _, ind in sorted(self.long, reverse=True)]

memory = Memory()