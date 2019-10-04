import unittest as t
import numpy as np
import functional as F


class TestFunctional(t.TestCase):
    def test_affine_forword(self):
        X = np.random.randn(3, 9)
        W = np.random.randn(9, 2)
        b = np.random.randn(2)
        self.assertTrue(
            (F.affine_forword(
                X.reshape(3, 3, 3), W, b)[0] == (X.dot(W) + b)).all())
        out = (F.affine_forword(X.reshape(3, 3, 3),
                                np.zeros_like(W),
                                b)[0]
               ==
               (np.zeros((3, 2)) + b)).all()
        self.assertTrue(out)

    def test_affine_backword(self):
        X = np.random.randn(3, 9)
        X = X.reshape(3, 3, 3)
        W = np.random.randn(9, 2)
        b = np.random.randn(2)

        out, cache = F.affine_forword(X, W, b)
        dX, dW, db = F.affine_backword(np.ones((3, 2)), cache)

        dX_num = numerical_gradient(lambda x: F.affine_forword(x, W, b)[0], X)
        dW_num = numerical_gradient(lambda x: F.affine_forword(X, x, b)[0], W)
        db_num = numerical_gradient(lambda x: F.affine_forword(X, W, x)[0], b)

        self.assertLess(np.mean(np.abs(dX - dX_num)), 0.0001)
        self.assertLess(np.mean(np.abs(dW - dW_num)), 0.0001)
        self.assertLess(np.mean(np.abs(db - db_num)), 0.0001)

    def test_relu_forward(self):
        X = np.linspace(1, 10, num=27)
        self.assertTrue((F.relu_forward(X)[0] == X).all())
        self.assertTrue((F.relu_forward(-X)[0] == np.zeros_like(X)).all())
        X = X.reshape((3, 9))
        self.assertTrue((F.relu_forward(X)[0] == X).all())
        self.assertTrue((F.relu_forward(-X)[0] == np.zeros_like(X)).all())
        X = X.reshape((3, 3, 3))
        self.assertTrue((F.relu_forward(X)[0] == X).all())
        self.assertTrue((F.relu_forward(-X)[0] == np.zeros_like(X)).all())

    def test_relu_backward(self):
        X = np.random.randn(3, 9)
        out, cache = F.relu_forward(X)
        dX = F.relu_backward(np.ones_like(X), cache)
        dX_num = numerical_gradient(lambda x: F.relu_forward(x)[0], X)
        self.assertLess(np.mean(np.abs(dX - dX_num)), 0.0001)


def numerical_gradient(f, x, h=0.00001):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        x[i] += h
        a = f(x)
        x[i] -= 2 * h
        b = f(x)
        x[i] += h
        grad[i] = np.sum(a - b) / (2 * h)
        it.iternext()
    return grad


if __name__ == '__main__':
    t.main()
