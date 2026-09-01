from itertools import product

def make_component_labels(dim, order) -> list[str]:

    dirs = ["x", "y", "z"][:dim]

    return [
        "".join(p)
        for p in product(dirs, repeat=order)
    ]
