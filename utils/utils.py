import os
import pickle

def load_all_pickles_in_folder(folder_path):
    # List all files in the directory
    files = os.listdir(folder_path)
    
    # Filter pickle files
    pickle_files = [file for file in files if file.endswith('.pkl')]
    
    # Load pickle files
    data = {}
    for file in pickle_files:
        with open(os.path.join(folder_path, file), 'rb') as f:
            data[file] = pickle.load(f)
    
    return data

