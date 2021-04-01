class VF(dict):
    "hashable wrapper over dict, can be used as a static argument for a jitted function"

    def __setattr__(self, key, value):
        raise RuntimeError("setting atttributes on vf not supported")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
