import pandas as pd
import argparse
import re
import numpy as np

def augment_descriptions(description, command, num_variations=3):
    variations = []
    
    templates = [
        "How to {}?",
        "Command to {}",
        "Terminal command for {}",
        "Bash command to {}",
        "How can I {} in terminal?",
        "What's the command to {}?",
        "I need to {} using terminal"
    ]
    
    action_words = extract_action_from_command(command)
    
    for template in np.random.choice(templates, num_variations):
        if action_words:
            variation = template.format(action_words)
        else:
            variation = template.format(description)
        variations.append(variation)
    
    return variations

def extract_action_from_command(command):
    action_map = {
        'ls': 'list files',
        'grep': 'search text',
        'find': 'find files',
        'cp': 'copy files',
        'mv': 'move files',
        'rm': 'remove files',
        'chmod': 'change permissions',
        'mkdir': 'create directory',
        'cat': 'view file content',
        'ps': 'show processes'
    }
    
    cmd_base = command.split()[0]
    return action_map.get(cmd_base, None)

def delete_cringe(command):
    return re.sub(r"['\"`\(\)\[\]]","", command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augmentate data')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with commands')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--command_column', type=str, default='command', help='Name of the command column')
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    augmented = []
    for i,r in frame.iterrows():
        variands = augment_descriptions(r["description"], r["command"])
    frame = 




