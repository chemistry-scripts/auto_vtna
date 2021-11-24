import argparse
import logging
import csv
import sys
from pathlib import Path
import numpy as np


def main():
    """Main function that does the job"""
    setup_logging()
    args = get_input_arguments()
    data = list()
    for file in args["input_file"]:
        with open(file) as csv_file:
            csv.reader(csv_file)

    # data is a list of ndarrays.


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
        Initializes the instance with an empty ca table.
        """
        self.ca = np.array([0.0])

    @staticmethod
    def euclidean(point_1: np.ndarray, point_2: np.ndarray) -> float:
        d = point_1 - point_2
        return np.linalg.norm(d)

    def distance(self, polyline_1: np.ndarray, polyline_2: np.ndarray) -> float:
        """
        Calculates the Fréchet distance between poly-lines 1 and 2
        This function implements the algorithm described by Eiter & Manilla
        :param polyline_1: Poly-line 1
        :param polyline_2: Poly-line 2
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

            d = DiscreteFrechet.euclidean(polyline_1[i], polyline_2[j])
            if i == 0 and j == 0:
                self.ca[i, j] = d
            elif i > 0 and j == 0:
                self.ca[i, j] = max(calculate(i - 1, 0), d)
            elif i == 0 and j > 0:
                self.ca[i, j] = max(calculate(0, j - 1), d)
            elif i > 0 and j > 0:
                self.ca[i, j] = max(
                    min(
                        calculate(i - 1, j),
                        calculate(i - 1, j - 1),
                        calculate(i, j - 1),
                    ),
                    d,
                )
            else:
                self.ca[i, j] = np.infty
            return self.ca[i, j]

        n_p = polyline_1.shape[0]
        n_q = polyline_2.shape[0]
        self.ca = np.zeros((n_p, n_q))
        self.ca.fill(-1.0)
        return calculate(n_p - 1, n_q - 1)


def setup_logging():
    """Setup logging for module"""
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)


def get_input_arguments():
    """Check command line options and accordingly set computation parameters."""
    logger = logging.getLogger()

    # List of values to extract
    values = dict.fromkeys(["input_files"])

    # Basic parser setup
    parser = argparse.ArgumentParser(
        description=help_description(), epilog=help_epilog()
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    # Add arguments to parser
    parser.add_argument(
        "-i",
        "--input_files",
        type=str,
        nargs="+",
        help="List of files for which a single point is necessary",
    )
    try:
        args = parser.parse_args()
    except argparse.ArgumentError as error:
        print(str(error))  # Print something like "option -a not recognized"
        sys.exit(2)

    # Setup file names
    values["input_files"] = [Path(i) for i in args.input_files]
    logger.debug("Input files: %s", values["input_files"])

    # All values are retrieved, return the table
    return values


def help_description():
    pass


def help_epilog():
    pass


if __name__ == "__main__":
    main()
