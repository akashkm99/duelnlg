import numpy as np
import logging

DELTA = 1e-7


def my_argmax(ay):
    idx = np.nonzero(ay == ay.max())[0]
    return idx[np.random.randint(0, idx.shape[0])]


def my_argmin(ay):
    idx = np.nonzero(ay == ay.min())[0]
    return idx[np.random.randint(0, idx.shape[0])]


def KLBernoulli(p, q=0.5):
    p += 0.0
    q += 0.0
    if p > 1 or q > 1:
        return -1
    if p <= DELTA:
        return -np.log(1 - q)
    if p >= 1 - DELTA:
        return -np.log(q)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


class RMED1:
    def __init__(self, feedback_mechanism, kweight=0.3, random_state=None):

        self.feedback_mechanism = feedback_mechanism
        self.random_state = random_state
        self.arms = list(self.feedback_mechanism.get_arms())
        self.arms_dict = dict(zip(np.arange(len(self.arms)), self.arms))
        self.kweight = kweight
        self.n_arms = len(self.arms)
        self.func_k = self.kweight * (self.n_arms ** 1.01)  # f(k)
        self.threshold = self.func_k  # Eq. 4 log(t) + f(k)

        self.w = np.zeros((self.n_arms, self.n_arms))  # pair-wise observation
        np.fill_diagonal(self.w, 1)
        self.n = self.w + self.w.T
        self.u = 0.5 * np.ones((self.n_arms, self.n_arms))  # preference estimator
        self.t = 0  # time step
        self.log_likelihood = np.ones(self.n_arms)  # log likelihood

        self.arm_l = 0
        self.arm_r = 0
        self.LC = self.random_state.permutation(
            self.n_arms
        ).tolist()  # arms in current step
        self.LR = set(range(self.n_arms))  # remaining arms
        self.LN = set()  # next step

    def get_relative_arm(self):
        eps = 1e-9
        u = self.w[self.arm_l] / (self.w[self.arm_l] + self.w[:, self.arm_l] + eps)
        u[self.arm_l] = 1
        u_min = u.min()
        arm_r = self.arm_l if u_min > 0.5 else my_argmin(u)

        return arm_r

    def get_arms(self):
        if self.t < self.n_arms * (self.n_arms - 1) / 2.0:
            self.arm_r += 1
            if self.arm_r == self.n_arms:
                self.arm_l += 1
                self.arm_r = self.arm_l + 1
        else:
            self.arm_l = self.LC.pop()
            self.arm_r = self.get_relative_arm()
        return self.arm_l, self.arm_r

    def __update_likelihood(self):
        for k in [self.arm_l, self.arm_r]:
            o_hat = np.where(self.u[k] <= 0.5)[0]
            kl = [KLBernoulli(self.u[k][j], 0.5) for j in o_hat]
            self.log_likelihood[k] = np.dot(self.n[k][o_hat], kl)

    def __get_likelihood(self):
        for k in range(self.n_arms):
            o_hat = np.where(self.u[k] <= 0.5)[0]
            kl = [KLBernoulli(self.u[k][j], 0.5) for j in o_hat]
            self.log_likelihood[k] = np.dot(self.n[k][o_hat], kl)

    def step(self):
        if self.n_arms > 1:
            arm1, arm2 = self.get_arms()
            arm1, arm2, score = self.feedback_mechanism.get_duel(
                self.arms_dict[arm1], self.arms_dict[arm2]
            )
            if score is not None:
                self.update_scores(arm1, arm2, score)
            else:
                self.t += 1

    def update_scores(self, arm1, arm2, score):

        self.w[arm1][arm2] += score
        self.w[arm2][arm1] += 1 - score

        self.n[arm1][arm2] += 1
        self.n[arm2][arm1] += 1
        self.u[arm1][arm2] = self.w[arm1][arm2] / self.n[arm1][arm2]
        self.u[arm2][arm1] = 1 - self.u[arm1][arm2]

        if self.t >= self.n_arms * (self.n_arms - 1) / 2.0:
            # update list
            # Line 15
            self.LR.remove(self.arm_l)

            # Line 16
            self.__update_likelihood()
            new_items = np.where(
                self.log_likelihood
                <= self.log_likelihood.min() + np.log(self.t) + self.func_k
            )[0]
            if new_items.shape[0]:
                for item in new_items:
                    if item not in self.LR:
                        self.LN.add(item)

            if not self.LC:  # Line 19
                self.LC = list(self.LN)
                self.random_state.shuffle(self.LC)
                self.LR, self.LN = self.LN, set()

        self.t += 1
        if self.t == self.n_arms * (self.n_arms - 1) / 2.0:
            self.__get_likelihood()

    def get_winner(self):
        u = self.w / (self.w + self.w.T + DELTA)
        return self.arms_dict[my_argmax(np.sum(u > 0.5, axis=1))]
