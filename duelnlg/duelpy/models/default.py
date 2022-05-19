class Default:
    def get_duel(self, sample):
        return False

    def duel(self, sample):
        raise AssertionError(
            "Duel class should not be called for the default algorithm"
        )

    def start(self, env):
        return
