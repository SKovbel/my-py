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


#
os.makedirs(DIR, exist_ok=True)


#
DIR_DATA = os.path.join(os.path.dirname(__file__), "../../tmp/torch-cv")
def path(name, create=False):
    dir = os.path.join(DIR_DATA, *name) if type(name) is list else os.path.join(DIR_DATA, name)
    if create:
        os.makedirs(dir, exist_ok=True)
    return dir
