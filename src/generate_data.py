import requests
import json
import pandas as pd
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import pydantic
from langchain_core.runnables.base import RunnableLambda
from tqdm import tqdm
import re
from langchain.schema.output_parser import StrOutputParser
from json_repair import repair_json

BASE_URL = "http://localhost:1234/v1"

def clean_json_output(text):
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    r = repair_json(text.strip())
    try:
        js = json.loads(r)
        if "properties" in js:
            return json.dumps(js["properties"])
    except Exception:
        pass
    return r

class CommandDescription(pydantic.BaseModel):
    reasoning: str = pydantic.Field(description="Detailed technical analysis of the command")
    description: str = pydantic.Field(description="Concise summary of what the command does")
    is_command: bool =  pydantic.Field(description="command is valid and exists and no contains other text")

def generate_descriptions(input_csv, output_csv, command_column='command', max_workers=5, limit=None):
    # Initialize LLM with structured output
    llm = ChatOpenAI(
        api_key="unused", 
        base_url=BASE_URL, 
        temperature=0.1, 
        max_completion_tokens=2048,
        model="local-model" 
    )
    
    parser = JsonOutputParser(pydantic_object=CommandDescription)

    prompt_template = PromptTemplate(
        template="""You are a bash command explanation system. Analyze the given bash command and provide a clear, accurate description in action format.

## Input
- Receive a bash command string
- Commands may include flags, arguments, and complex syntax

## Output Requirements
- Provide a JSON object with exactly two fields: "reasoning" and "description"
- "reasoning": Detailed technical analysis of the command, explaining each component (flags, arguments, syntax)
- "description": Concise, human-readable action of what the command does (1-2 sentences), always note paths, urls, ips, and important content
- Use English only for all output
- Check command is valid, exists and could be executed and contains no excess text or anything else, mark it in is_command field as true, false otherwise, follow the next instuctions:
### MARK is_command true if:
- command is valid and exists
- command could be executed without errors
- command not contains excess text or something else
### MARK is_command false if:
- command does not exists
- command is'n complete or contains excess text

IF COMMAND CONTAINS FILES OR IP ADDRESSES LET IT CORRECT AND EXISTS, VERFY ONLY SYNTAXIS IN COMMANDS

## Examples
Input: "cd /var/log"
Output: {{
  "reasoning": "Command 'cd' (change directory) with argument '/var/log' (absolute path to directory)",
  "description": "Change current working directory to /var/log",
  is_command:true
}}

Input: "tar -czvf archive.tar.gz /home/user"
Output: {{
  "reasoning": "Command 'tar' (tape archive) with flags: -c (create archive), -z (gzip compression), -v (verbose output), -f (specify filename). Arguments: 'archive.tar.gz' (output filename), '/home/user' (directory to archive)",
  "description": "Create a gzip-compressed tar archive of /home/user directory",
  is_command: true
}}

Input: "//192.168.2.101/ShareForVMs /media/share/ cifs username=toeknee,password=2dog$hit3"
Output: {{
"reasoning": "No commands provided, it could be line in log file or part of arguments. command is incorrect",
"description": "no command is provided",
is_command: false
}}


Input: "git clone git://projects.archlinux.org/archiso.git && cd archiso"
Output:{{ 
"reasoning": "The command uses `git clone` to create a copy of the repository from 'git://projects.archlinux.org/archiso.git'. It then changes the current working directory to 'archiso' using `cd`. No files or IP addresses are involved.",
"description": "Clones the Git repository from the projects.archlinux.org/archiso.git and changes the current working directory to archiso.",
"is_command": true
}}


## Security Note
- Do not execute or simulate execution of any commands
- Provide analysis based on command syntax and documentation only
- Flag potentially destructive commands in your reasoning

## Output Format
{format_instructions}

YOU SHOULD ALWAYS FOLLOW OUTPUT FORMAT, NO SKIPS, NO OTHER FIELDS, NO ANYTHING ELSE, ONLY reasoning, description and is_command JSON

Describe this command: {command}""",
        input_variables=["command"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # Create chain
    chain = prompt_template | llm | StrOutputParser() | clean_json_output | parser
    
    # Read input data
    df = pd.read_csv(input_csv)
    commands = df[command_column].unique().tolist()
    
    # Apply limit if specified
    if limit:
        commands = commands[:limit]
    
    # Check existing results
    existing_data = pd.DataFrame()
    if os.path.exists(output_csv):
        existing_data = pd.read_csv(output_csv)
        existing_commands = existing_data[command_column].tolist()
        commands = [cmd for cmd in commands if cmd not in existing_commands]
    
    print(f"Found {len(commands)} new commands to process")
    
    # Create a lock for thread-safe file writing
    file_lock = threading.Lock()
    
    def process_command(command):
        """Process a single command and return the result"""
        try:
            result = chain.invoke({"command": command})
            r = {
                "command": command,
                "reasoning": result.get("reasoning", ""),
                "description": result.get("description", ""),
                "is_command": result.get("is_command", False)
            }
            return r
        except Exception as e:
            print(e)
            return {
                "command": command,
                "reasoning": "",
                "description": "",
                "is_command": False
            }
    
    def save_results(batch_results):
        """Save a batch of results to CSV (thread-safe)"""
        with file_lock:
            # Create DataFrame from results
            new_df = pd.DataFrame(batch_results)
            
            # Append to existing file or create new
            if os.path.exists(output_csv):
                existing_df = pd.read_csv(output_csv)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df
            
            # Save to CSV
            final_df.to_csv(output_csv, index=False)
    
    # Process commands with ThreadPoolExecutor
    results = []
    batch_size = 50
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all commands for processing
        future_to_command = {
            executor.submit(process_command, command): command 
            for command in commands
        }
        
        # Process completed futures
        for i, future in tqdm(enumerate(as_completed(future_to_command)), total=len(commands)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # print(f"Processed {i+1}/{len(commands)}: {result[command_column]}")
                # Save batch every batch_size results
                if len(results) >= batch_size:
                    save_results(results)
                    results = []
                    
            except Exception as e:
                pass
    
    # Save any remaining results
    if results:
        save_results(results)
        print(f"Saved final batch of {len(results)} results")
    
    print(f"Processing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate command descriptions using LMstudio')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file with commands')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--command_column', type=str, default='command', help='Name of the command column')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of commands to process')
    
    args = parser.parse_args()
    generate_descriptions(
        args.input, 
        args.output, 
        args.command_column, 
        args.max_workers, 
        args.limit
    )