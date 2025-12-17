import os
from typing import List, Tuple


def get_directory_size(path: str) -> int:
    """
    Calcula o tamanho total (em bytes) de um diretório de forma recursiva.
    """
    total_size = 0

    for root, _, files in os.walk(path):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
            except (FileNotFoundError, PermissionError):
                continue

    return total_size


def get_folders_sizes(base_path: str) -> List[Tuple[str, int]]:
    """
    Retorna uma lista de tuplas (nome_da_pasta, tamanho_em_bytes),
    ordenada do maior para o menor.
    """
    folders_sizes = []

    for entry in os.scandir(base_path):
        if entry.is_dir():
            size = get_directory_size(entry.path)
            folders_sizes.append((entry.name, size))

    folders_sizes.sort(key=lambda x: x[1], reverse=True)
    return folders_sizes


if __name__ == "__main__":
    base_directory = "C:\\"

    result = get_folders_sizes(base_directory)

    for folder, size in result:
        print(f"{folder:<40} {size / (1024**2):>10.2f} MB")
