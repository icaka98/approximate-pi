import logging

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def approximate_pi_geom(
    circle_radius: float = 1.0, number_of_iterations: int = 10_000
) -> float:
    """
    Pi approximation (<4) implementation using the approach during the interview.
    """
    step = circle_radius / number_of_iterations

    def _f(x: float) -> float:
        return (circle_radius**2 - x**2) ** 0.5

    pi_estimate = 0.0
    for iteration_step in tqdm(
        range(number_of_iterations), desc="Approximating Pi using rectangles"
    ):
        x = iteration_step * step
        y = _f(x)
        pi_estimate += y * step

    return 4 * pi_estimate


def approximate_pi_monte_carlo(
    circle_radius: float = 1.0, number_of_iterations: int = 10_000
) -> float:
    """
    An alternative Pi approximation (<4) using the Monte Carlo method
    """
    random_points = np.random.uniform(
        -circle_radius, circle_radius, size=(number_of_iterations, 2)
    ).tolist()

    point_inside_or_on_circle = sum(
        x**2 + y**2 <= circle_radius**2
        for x, y in tqdm(random_points, desc="Approximating Pi using Monte Carlo")
    )

    return 4 * point_inside_or_on_circle / number_of_iterations


if __name__ == "__main__":
    pi_approx_geom = approximate_pi_geom(number_of_iterations=1_000_000)
    pi_approx_monte_carlo = approximate_pi_monte_carlo(number_of_iterations=1_000_000)

    logging.info(
        f"Pi approximation using rectangles (1,000,000 iterations): {pi_approx_geom}"
    )
    logging.info(
        f"Pi approximation using Monte Carlo (1,000,000 points): {pi_approx_monte_carlo}"
    )
