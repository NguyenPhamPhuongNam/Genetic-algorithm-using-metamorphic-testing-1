# ——— Chromosome class: a variable‐length sequence of sub‐vectors ———
class Chromosome:
    def __init__(self, img_shape, transformations=None):
        self.img_shape = img_shape
        if transformations is None:
            # start with random length between 2 and 6
            length = random.randint(2,10)
            self.transformations = [
                generate_random_subvector(img_shape) for _ in range(length)
            ]
        else:
            # initialize from given list
            self.transformations = transformations
        self.fitness = None