import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
import libmod; libmod.build(); print("built via imported module")
