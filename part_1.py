import numpy as np


def svd_manual(A):
    ATA = A.T @ A

    eigenvalues, V = np.linalg.eigh(ATA)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    singular_values = np.sqrt(eigenvalues)
    Sigma = np.diag(singular_values)

    U = A @ V @ np.linalg.inv(Sigma)

    return U, Sigma, V.T


# Перевірка
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, Sigma, VT = svd_manual(A)
A_reconstructed = U @ Sigma @ VT

print("Original Matrix:\n", A)
print("Reconstructed Matrix:\n", A_reconstructed)


