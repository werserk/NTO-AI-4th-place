import enchant
import numpy as np
from string import punctuation, digits, whitespace


not_letters = punctuation + digits + whitespace


class SpellCorrection:
    def __init__(self, dict_path, collides_path):
        self.dict = enchant.request_pwl_dict(dict_path)
        self.collides, self.alphabet = self.load_collides(collides_path)

    def __call__(self, word):
        candidates = self.get_candidates(word)
        variants = []
        for candidate in candidates:
            word1, word2 = self.sort_words(word, candidate)
            moves = self.calculate_moves(word1, word2)
            variants.append((candidate, len(moves)))
        variants.sort(key=lambda x: x[1])
        return variants

    #             print(candidate)
    #             print()

    def get_chance(self, a, b):
        ind1, ind2 = self.char2ind(a), self.char2ind(b)
        if ind1 == -1 or ind2 == -1:
            return 0
        return self.collides[ind1][ind2]

    def load_collides(self, path):
        f = np.load(path)
        return f['collides'], ''.join(f['alphabet'])

    def char2ind(self, x):
        return self.alphabet.find(x)

    def calculate_moves(self, a, b):
        moves = []
        matrix, steps = self.dist(a, b)
        path = self.recover_res(matrix, steps)
        for (i, j) in path:
            if steps[i][j] == 3:
                moves.append((i - 1, j - 1))
            if steps[i][j] == 1:
                moves.append(((i - 1, j), -1))
        return moves

    def get_candidates(self, word):
        candidates = self.dict.suggest(word)
        candidates.sort(key=lambda x: (len(x) != len(word)))
        return candidates

    def separate(self, word):
        prefix_end = self.get_prefix(word)
        postfix_start = self.get_postfix(word)
        prefix = word[:prefix_end]
        postfix = word[postfix_start:]
        word = word[prefix_end:postfix_start]
        return prefix, word, postfix

    def get_prefix(self, word):
        prefix_end = 0
        for char in word:
            if char in not_letters:
                prefix_end += 1
            else:
                break
        return prefix_end

    def get_postfix(self, word):
        postfix_start = len(word)
        for i in range(len(word) - 1, -1, -1):
            char = word[i]
            if char in not_letters:
                postfix_start -= 1
            else:
                break
        return postfix_start

    def sort_words(self, a, b):
        return sorted([a, b], key=lambda x: len(x))

    def dist(self, a, b):
        a, b = '#' + a, '#' + b

        # creating empty matrix
        matrix = [[0] * (len(a)) for _ in range(len(b))]
        matrix_steps = [[0] * (len(a)) for _ in range(len(b))]

        # filling borders
        for i in range(len(b)):
            matrix[i][0] = i
        for j in range(len(a)):
            matrix[0][j] = j

        # main process
        for i in range(1, len(b)):
            for j in range(1, len(a)):
                value = (a[j] != b[i])
                if matrix[i][j - 1] + 1 <= min(matrix[i - 1][j] + 1,
                                                   matrix[i - 1][j - 1] + value):
                    matrix[i][j] = matrix[i][j - 1] + 1
                    if not value:
                        matrix_steps[i][j] = 2
                    else:
                        matrix_steps[i][j] = 1
                elif matrix[i - 1][j] + 1 <= min(matrix[i][j - 1] + 1,
                                                     matrix[i - 1][j - 1] + value):
                    matrix[i][j] = matrix[i - 1][j] + value
                    if not value:
                        matrix_steps[i][j] = 2
                    else:
                        matrix_steps[i][j] = 1
                elif matrix[i - 1][j - 1] + value <= min(matrix[i - 1][j] + 1,
                                                         matrix[i][j - 1] + 1):
                    matrix[i][j] = matrix[i - 1][j - 1] + value
                    if not value:
                        matrix_steps[i][j] = 2
                    else:
                        matrix_steps[i][j] = 3
        return matrix, matrix_steps

    def recover_res(self, matrix, steps):
        i = len(matrix) - 1
        j = len(matrix[0]) - 1
        result = [(i, j)]
        while i != 1 and j != 1:
            variants = [(matrix[i - 1][j - 1], steps[i - 1][j - 1], (i - 1, j - 1)),
                        (matrix[i - 1][j], steps[i - 1][j], (i - 1, j)),
                        (matrix[i][j - 1], steps[i][j - 1], (i, j - 1))]
            variants = min(variants, key=lambda x: (x[0], -x[1]))
            result.append(variants[2])
            i, j = variants[2]
        return result[::-1]