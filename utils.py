import numpy as np

piece_name_map = {
    'P': 0,
    'N': 1,
    'B': 2,
    'R': 3,
    'Q': 4,
    'K': 5,
    'p': 6,
    'n': 7,
    'b': 8,
    'r': 9,
    'q': 10,
    'k': 11,
}


def fen_to_bitboards(fen: str):
    # example FEN
    # 8/5N2/4p2p/5p1k/1p4rP/1P2Q1P1/P4P1K/5q2 w - - 15 44

    # first 6 columns for white, next 6 for black
    bitboards = np.zeros(shape=(64, 12))
    fen = fen[:fen.find(' ')]  # ignore everything after the space
    square_idx = 0  # 0 - h1, 63 - a8
    for rank in fen.split(('/')):
        for c in rank:
            if c in piece_name_map:
                bitboards[square_idx, piece_name_map[c]] = 1
                square_idx += 1
            else:
                square_idx += int(c)
    return bitboards


def get_gamephase_batch(bitboards_batch):
    return (bitboards_batch.sum(axis=1) *
            [0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0]).sum(axis=1) / 24


reindex = np.array([s ^ 56 for s in range(64)])


def evaluate_position(psqt, bitboards):
    mg_psqt = psqt[:, [0, 2, 4, 6, 8, 10]]
    eg_psqt = psqt[:, [1, 3, 5, 7, 9, 11]]

    mg = bitboards[:, :6] * mg_psqt - bitboards[reindex, 6:] * mg_psqt
    eg = bitboards[:, :6] * eg_psqt - bitboards[reindex, 6:] * eg_psqt

    gamephase = get_gamephase_batch(np.array([bitboards]))
    return np.sum(mg) * gamephase + np.sum(eg) * (1-gamephase)


def evaluate_position_batch(psqt, bitboards_batch):
    w_mg_psqt = psqt[reindex, :6].reshape((1, 64, 6))
    w_eg_psqt = psqt[reindex, 6:].reshape((1, 64, 6))
    b_mg_psqt = psqt[:, :6].reshape((1, 64, 6))
    b_eg_psqt = psqt[:, 6:].reshape((1, 64, 6))

    mg = bitboards_batch[:, :, :6] * w_mg_psqt - \
        bitboards_batch[:, :, 6:] * b_mg_psqt
    eg = bitboards_batch[:, :, :6] * w_eg_psqt - \
        bitboards_batch[:, :, 6:] * b_eg_psqt

    gamephase = get_gamephase_batch(bitboards_batch)
    return np.sum(mg, axis=(1, 2)) * gamephase + np.sum(eg, axis=(1, 2)) * (1 - gamephase)


def get_wdl(psqt, fen_list):
    bitboards_list = []
    for fen in fen_list:
        bitboards_list.append(fen_to_bitboards(fen).reshape(1, 64, 12))
    bitboards_batch = np.concatenate(bitboards_list)

    evaluations = evaluate_position_batch(psqt, bitboards_batch)
    return 1 / (1 + np.exp(-evaluations/400))


def main():
    pass
    # from psqt import pesto_psqt
    # e = evaluate_position(pesto_psqt, fen_to_bitboards(
    #     'rnbqk2r/ppp2ppp/3b1n2/3p4/3P4/2NBP3/PP3PPP/R1BQK1NR b KQkq - 2 6'))
    # print(e)


if __name__ == '__main__':
    main()
