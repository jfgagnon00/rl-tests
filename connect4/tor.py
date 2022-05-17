import torch
import numpy as np


if __name__ == "__main__":
    print(f"Torch: {torch.__version__}")

    x = torch.tensor([1., 2.], requires_grad=True)
    y = torch.tensor([3., 4.], requires_grad=True)

    x_prime = torch.tensor(x**2, requires_grad=True)

    z = x_prime + y

    # partial dz / dx == 2x
    # partial dz / dy == 2y

    # dx / dt = 0 -> c'est une constante
    # dy / dt = 0 -> c'est une constante

    # dz / dt == partial_dz_dx * dx_dt + partial_dz_dy * dy_dt

    ones = torch.tensor([1., 1.])
    z.backward(ones)

    print("x_prime")
    print(x_prime)
    print(x_prime.grad)
    print()

    print("y")
    print(y)
    print(y.grad)
    print()

    print("z")
    print(z)
    print()