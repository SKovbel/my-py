import os


# 1
PATH = os.path.join(os.getcwd(), f"../../")

# 2
DIR = os.path.dirname(os.path.abspath(__file__))

# 3
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keys.json')

# 4
path = lambda name: os.path.join(s.getcwd(), name)
path = lambda name: os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

# 5
for dir in os.listdir(DIR):
    dir = os.path.join(DIR, dir)
    if os.path.isdir(dir):
        pass
