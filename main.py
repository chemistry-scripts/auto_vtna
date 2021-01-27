import numpy as np

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def frechet_distance(cloud1, cloud2):
    # Compute Frechet Distance between two curves
    frechet = DiscreteFrechet()
    return frechet.distance(cloud1, cloud2)


class DiscreteFrechet(object):
    """
    Calculates the discrete Fréchet distance between two poly-lines using the
    original recursive algorithm

    Extracted from
    https://github.com/joaofig/discrete-frechet/blob/ac710990ce452515f4ecec4b527c7ed761fe286a/distances/discrete.py
    """

    def __init__(self):
        """
        Initializes the instance with a pairwise distance function.
        :param dist_func: The distance function. It must accept two NumPy
        arrays containing the point coordinates (x, y), (lat, long)
        """
        self.dist_func = DiscreteFrechet.euclidean
        self.ca = np.array([0.0])

    def euclidean(p: np.ndarray, q: np.ndarray) -> float:
        d = p - q
        return np.linalg.norm(d)

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines p and q
        This function implements the algorithm described by Eiter & Mannila
        :param p: Poly-line p
        :param q: Poly-line q
        :return: Distance value
        """

        def calculate(i: int, j: int) -> float:
            """
            Calculates the distance between p[i] and q[i]
            :param i: Index into poly-line p
            :param j: Index into poly-line q
            :return: Distance value
            """
            if self.ca[i, j] > -1.0:
                return self.ca[i, j]

            d = self.dist_func(p[i], q[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i - 1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j - 1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(min(calculate(i - 1, j),
                                        calculate(i - 1, j - 1),
                                        calculate(i, j - 1)), d)
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = p.shape[0]
        n_q = q.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.set_printoptions(precision=4)

    p = np.array(
        [
            [0.2, 2.0],
            [1.5, 2.8],
            [2.3, 1.6],
            [2.9, 1.8],
            [4.1, 3.1],
            [5.6, 2.9],
            [7.2, 1.3],
            [8.2, 1.1],
        ]
    )
    q = np.array(
        [
            [0.3, 1.6],
            [3.2, 3.0],
            [3.8, 1.8],
            [5.2, 3.1],
            [6.5, 2.8],
            [7.0, 0.8],
            [8.9, 0.6],
        ]
    )

    distance = frechet_distance(p, q)
    print(distance)
