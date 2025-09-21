import subprocess
import re
import pandas as pd
import argparse
import os
from tqdm import tqdm

def get_man_description(command):
    try:
        man_output = subprocess.run(
            ["man", command], 
            capture_output=True, 
            text=True, 
            timeout=1
        )
        
        if man_output.returncode != 0:
            return None
            
        man_text = man_output.stdout
        
        description_match = re.search(r'NAME\s*\n\s*(\w+)\s*-\s*(.*?)(?:\n\n|\n[A-Z]|\Z)', 
                                     man_text, re.DOTALL | re.IGNORECASE)
        
        if description_match:
            description = description_match.group(2).strip()
            description = re.sub(r'\s+', ' ', description)
            return command, description
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
        
    return None

def parse_man_pages(input_csv, output_csv, command_column='command', max_workers=10):
    df = pd.read_csv(input_csv)
    commands = df[command_column].unique().tolist()
    
    print(f"Найдено {len(commands)} новых команд для обработки")
    
    results = []
        
    for i, c in tqdm(enumerate(commands),total= len(commands)):
        result = get_man_description(str(c))
        if result:
            command, description = result
            results.append({command_column: command, 'description': description})
            print(f"Обработано {i+1}/{len(commands)}: {command}")
    
    final_df = pd.DataFrame(results)
    
    final_df.to_csv(output_csv, index=False)
    print(f"Сохранено {len(results)} новых описаний в {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Парсинг man-страниц для получения описаний команд')
    parser.add_argument('--input', type=str, required=True, help='Путь к входному CSV файлу с командами')
    parser.add_argument('--output', type=str, required=True, help='Путь к выходному CSV файлу')
    parser.add_argument('--command_column', type=str, default='command', help='Название колонки с командами')
    parser.add_argument('--max_workers', type=int, default=4, help='Количество потоков для параллельной обработки')
    
    args = parser.parse_args()
    parse_man_pages(args.input, args.output, args.command_column, args.max_workers)