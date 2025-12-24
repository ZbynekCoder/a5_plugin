import itertools


def is_even_permutation(p):
    inv = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inv += 1
    return inv % 2 == 0


def compose(p, q):
    # (p ∘ q)(i) = p(q(i))
    return tuple(p[q[i]] for i in range(len(p)))


def generate_a5():
    perms = []
    for p in itertools.permutations(range(5)):
        if is_even_permutation(p):
            perms.append(tuple(p))
    if len(perms) != 60:
        raise ValueError("A5 size mismatch")

    perm_to_id = {p: i for i, p in enumerate(perms)}
    identity = tuple(range(5))
    id_id = perm_to_id[identity]

    mul = [[0] * 60 for _ in range(60)]
    for a in range(60):
        for b in range(60):
            c = compose(perms[a], perms[b])
            mul[a][b] = perm_to_id[c]

    return perms, mul, id_id
