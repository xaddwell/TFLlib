import numpy as np

from flcore.security.attack.privacy.inference.lia.base import BaseLabelInferenceAttack

# ref: https://github.com/googleinterns/learning-bag-of-words/blob/main/image_recognition/rlg.py
# can only predict labels appeared, without numbers and position
class RLGAttack(BaseLabelInferenceAttack):
    def __init__(self, Server, config):
        super().__init__('RLG', Server, config)
        self.recover_num = None
  
    def _solve_perceptron(self, X, y, fit_intercept=True, max_iter=1000, tol=1e-3, eta0=1.):
        from sklearn.linear_model import Perceptron
        clf = Perceptron(fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, eta0=eta0)
        clf.fit(X, y)
        if not fit_intercept:
            pass
        if clf.score(X, y) > 0.9:
            return True
        return False

    def _solve_lp(self, A, b, c):
        from cvxopt import matrix, solvers
        solvers.options['show_progress'] = False
        np.random.seed(None)
        for t in range(1):
            A, b, c = matrix(A), matrix(b), matrix(c)
            sol = solvers.lp(c, A, b)
            x = sol['x']
            if x is not None:
                ret = A * x
                # return np.count_nonzero(np.array(ret[1:]) <= 0) > 0.9 * len(ret)
                if ret[0] < -0.1 and np.max(ret[1:]) < 1e-2 and np.count_nonzero(np.array(ret[1:]) <= 0) > 0.5 * len(ret):
                    return True
        return False

    def label_inference(self, shared_info, epsilon=1e-8, t_model = None):
        print("Beginning RLG label inference attack")
        A = shared_info[-2]
        A = A.T
        m, n = A.shape
        B, s, C = np.linalg.svd(A, full_matrices=False)
        self.recover_num = np.linalg.matrix_rank(A)
        k = min(self.gt_k, self.recover_num)
        print("k", k)
        print("Predicted length of target sequence:", self.recover_num)
        print("Finding SVD of W...")
        C = C[:k, :].astype(np.double)

        # Find x: x @ C has only one positive element
        # Filter possible labels using perceptron algorithm
        bow = []
        if t_model == "ResNet50":
            bow = np.reshape(np.where(np.min(A, 0) < 0), -1).tolist()
        for i in range(n):
            if i in bow:
                continue
            indices = [j for j in range(n) if j != i]
            np.random.shuffle(indices)

            if self._solve_perceptron(np.concatenate([C[:, i:i + 1], C[:, indices[:self.class_num-1]]], 1).transpose(),
                    np.array([1 if j == 0 else -1 for j in range(self.class_num)]),
                    fit_intercept=True,
                    max_iter=1000,
                    tol=1e-3
            ):
                bow.append(i)
    
        # Get the final set with linear programming
        ret_bow = []
        for i in bow:
            if i in ret_bow:
                continue
            indices = [j for j in range(n) if j != i]
            D = np.concatenate([C[:, i:i + 1], C[:, indices]], 1)
            indices2 = np.argsort(np.linalg.norm(D[:, 1:], axis=0))[-199:]
            A = np.concatenate([D[:, 0:1], -D[:, 1 + indices2]], 1).transpose()
            if self._solve_lp(
                    A=A,
                    b=np.array([-epsilon] + [0] * len(indices2)),
                    c=np.array(C[:, i:i + 1])
            ):
                ret_bow.append(i)
        label_pred = ret_bow
        rec_instances = [1 if i in label_pred else 0 for i in range(self.class_num)]
        self.rec_labels = np.array(label_pred)
        self.rec_instances = np.array(rec_instances)
        return len(ret_bow), ret_bow        