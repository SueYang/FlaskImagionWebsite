class Result(object):
    path = ""
    name = ""
    score = ""
    rank = ""

    def __init__(self, path, name, score, rank):
        self.path = path
        self.name = name
        self.score = score
        self.rank = rank
