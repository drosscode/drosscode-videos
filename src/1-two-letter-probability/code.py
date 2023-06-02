from itertools import combinations
from string import ascii_lowercase


def get_words_from_file():
    with open("words_alpha.txt", "r") as words_file:
        while word := words_file.readline().lower().strip():
            yield word


def generate_initial_count_dict():
    pairs_set = set()

    for c1 in ascii_lowercase:
        for c2 in ascii_lowercase:
            if c1 != c2:
                char_pair = tuple(
                    sorted(
                        (
                            c1,
                            c2,
                        )
                    )
                )
                pairs_set.add(char_pair)

    count_dict = {}

    for pair in pairs_set:
        count_dict[pair] = 0

    return count_dict


def compute_probability():
    single_count_dict = {c: 0 for c in ascii_lowercase}
    pair_count_dict = generate_initial_count_dict()
    total_word_count = 0

    for word in get_words_from_file():
        word_chars = set(c for c in word if c in ascii_lowercase)
        total_word_count += 1

        for c in word_chars:
            single_count_dict[c] += 1

        for c1, c2 in combinations(word_chars, 2):
            char_pair = tuple(
                sorted(
                    (
                        c1,
                        c2,
                    )
                )
            )
            pair_count_dict[char_pair] += 1

    for c in sorted(single_count_dict, key=single_count_dict.get, reverse=True):
        print(c, single_count_dict[c])

    for c in sorted(pair_count_dict, key=pair_count_dict.get, reverse=True):
        print(c, pair_count_dict[c])

    print(total_word_count)

    conditional_probability_dict = {}

    # compute first letter percentage (letter_count/total)
    for c1 in ascii_lowercase:
        p_a = single_count_dict[c1] / total_word_count
        # get intersection of two letters occurring (pair_count_dict)
        for c2 in ascii_lowercase:
            if c1 != c2:
                pair = tuple(
                    sorted(
                        (
                            c1,
                            c2,
                        )
                    )
                )
                p_ab = pair_count_dict[pair] / total_word_count
                p_b_given_a = p_ab / p_a
                conditional_probability_dict[
                    (
                        c1,
                        c2,
                    )
                ] = p_b_given_a

    for c in sorted(
        conditional_probability_dict,
        key=conditional_probability_dict.get,
        reverse=True,
    ):
        print(c, conditional_probability_dict[c])


if __name__ == "__main__":
    compute_probability()
