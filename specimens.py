import numpy as np
from sklearn.metrics import mean_squared_error as mse, log_loss
from psqt import *
from utils import *


class Specimen():

    __slots__ = ['genes']

    def __init__(self, genes) -> None:
        self.genes = genes

    def __str__(self) -> str:
        s = ""
        for k, v in self.genes.items():
            s += f"{k}: {v}\n"
        return s

    def evaluate(self):
        return 0

    def mutate(self):
        return None

    def copy(self):
        return Specimen(self.genes)

    @classmethod
    def crossover(cls, *parents):
        new_genes = {}
        n_parents = len(parents)
        for k in parents[0].genes.keys():
            parent_idx = np.random.randint(0, n_parents)
            new_genes[k] = parents[parent_idx].genes[k]
        return new_genes


class PSQTFactorisation():

    __slots__ = ['genes', 'n_components', 'n_weights', 'n_tables', 'piece_values'
                 'target_psqt', 'wdl_true', 'fen_list', 'target_relative_table',
                 'cw_ratio', 'component_range',
                 'weight_range', 'integer_weights',
                 'lock_full_component',
                 'loss_fn']

    @classmethod
    def use_target_psqts(cls, target_psqt):
        cls.target_psqt = target_psqt
        cls.n_tables = target_psqt.shape[1]
        cls.target_relative_table = cls.get_relative_table(target_psqt)

    @classmethod
    def use_piece_values(cls, piece_values):
        cls.piece_values = piece_values

    @classmethod
    def use_fens(cls, fen_list):
        cls.fen_list = fen_list
        cls.wdl_true = get_wdl(cls.target_psqt, fen_list)

    @classmethod
    def get_relative_table(cls, psqt):
        # relative_table = np.concatenate(
        #     [psqt.reshape(1, 64, 12).copy() for _ in range(64)])
        # for s in range(64):
        #     reference = psqt[s, :]
        #     diff = psqt - reference
        #     relative_table[s, :, :] = np.clip(diff.reshape((1, 64, 12)), -1, 1)
        # return relative_table

        reference = psqt.mean(axis=0)
        diff = psqt - reference
        return np.clip(diff, -1, 1)

    def __init__(self,
                 n_components,
                 component_range,
                 weight_range=(-np.inf, np.inf),
                 cw_ratio=1,
                 integer_weights=False,
                 lock_full_component=False,
                 loss_fn='mse',
                 genes=None) -> None:
        assert (n_components % cw_ratio == 0)
        self.n_components = n_components
        self.n_weights = n_components // cw_ratio
        self.cw_ratio = cw_ratio
        self.component_range = component_range
        self.weight_range = weight_range
        self.integer_weights = integer_weights
        self.lock_full_component = lock_full_component
        self.loss_fn = loss_fn
        if genes is None:
            self.genes = {
                'components': get_zero_components(n_components) * component_range[1],
                'weights': np.zeros(shape=(self.n_weights, self.n_tables))
                # np.random.randint(weight_range[0], weight_range[1], size=(self.n_weights, self.n_tables))

            }
            # self.genes['weights'][0, :] = self.weight_range[1]
            # self.genes['components'][:, 0] = self.component_range[1]

        else:
            self.genes = genes

    def get_phat(self):
        return np.floor(self.genes['components'] @ np.concatenate([self.genes['weights']] * self.cw_ratio)) + self.piece_values

    def evaluate(self):
        phat = self.get_phat()
        diff = np.abs(self.target_psqt - phat) * importance_mask
        if self.loss_fn == 'mse':
            return -np.mean(np.square(diff))
        elif self.loss_fn == 'mae':
            return -np.mean(diff)
        elif self.loss_fn == 'max_diff':
            return -np.max(diff) - 0.001 * np.mean(np.square(diff))
        elif self.loss_fn == 'wdl':
            wdl_hat = get_wdl(phat, self.fen_list)
            return -mse(self.wdl_true, wdl_hat)
        elif self.loss_fn == 'relative':
            relative_table = self.get_relative_table(phat)
            return np.sum(relative_table * self.target_relative_table) - \
                0.001 * np.mean(np.square(diff))

        else:
            raise ValueError('Incorrect Loss Function')

    def mutate(self, p_weight=0.05, p_component=0.01, m_weight=3, m_component=3):
        def get_deltas(size, p, m, integer):
            probs = np.random.random(size=size) < p
            if integer:
                magnitude = np.random.binomial(m * 2, p=0.5, size=size) - m
            else:
                magnitude = np.random.normal(0, m, size=size)
            return magnitude * probs

        self.genes['weights'] += get_deltas(
            (self.n_weights, self.n_tables), p_weight, m_weight, self.integer_weights)
        self.genes['weights'] = np.clip(
            self.genes['weights'], self.weight_range[0], self.weight_range[1])

        self.genes['components'] += get_deltas(
            (64, self.n_components), p_component, m_component, True)
        self.genes['components'] = np.clip(
            self.genes['components'], self.component_range[0], self.component_range[1])

    @classmethod
    def crossover(cls, *parents):
        return parents[0].copy()

    def copy(self):
        return PSQTFactorisation(
            n_components=self.n_components,
            component_range=self.component_range,
            weight_range=self.weight_range,
            cw_ratio=self.cw_ratio,
            integer_weights=self.integer_weights,
            loss_fn=self.loss_fn,
            genes={
                'components': self.genes['components'].copy(),
                'weights': self.genes['weights'].copy()
            }
        )
