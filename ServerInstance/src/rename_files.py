import os

# Ścieżki do folderów 'test' i 'train'
data_folders = ['../data/test', '../data/train']

# Przechodzenie przez każdy folder w 'data_folders'
for folder in data_folders:
    for subfolder in ['CNV', 'DME', 'DRUSEN', 'NORMAL']:
        full_path = os.path.join(folder, subfolder)
        if os.path.exists(full_path):
            # Pobieranie listy plików JPEG w folderze, sortowanie nie jest wymagane, ale może być pomocne
            files = [f for f in os.listdir(full_path) if f.endswith('.jpeg')]
            # Sortowanie może być użyteczne, jeśli chcesz zachować oryginalną kolejność plików w jakimś sensie
            files.sort()
            # Zmiana nazwy plików
            for i, file in enumerate(files, start=1):
                new_file_name = f"{i}.jpeg"
                os.rename(os.path.join(full_path, file), os.path.join(full_path, new_file_name))
            print(f"Zaktualizowano nazwy plików w {full_path}")
