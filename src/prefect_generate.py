import json
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import pydantic
from tqdm import tqdm
import re
from langchain.schema.output_parser import StrOutputParser
from json_repair import repair_json
import itertools
import csv
import mlflow
from typing import List

from prefect import flow, task
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

mlflow.set_tracking_uri("http://192.168.100.222:5000")

def clean_json_output(text):
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub("<think>.*</think>", "", text)
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

class ServerPoolManager:
    def __init__(self, base_urls, timeout=120):
        self.base_urls = base_urls
        self.timeout = timeout
        self.url_cycle = itertools.cycle(base_urls)
        self.lock = threading.Lock()
        
    def get_next_server(self):
        with self.lock:
            return next(self.url_cycle)
    
    def create_llm(self, base_url=None):
        if base_url is None:
            base_url = self.get_next_server()
            
        return ChatOpenAI(
            api_key="unused", 
            base_url=base_url, 
            temperature=0.1, 
            max_completion_tokens=2048,
            model="local-model",
            timeout=self.timeout
        )

def process_single_command_internal(command, server_pool):
    """Internal function to process single command without Prefect logging"""
    try:
        parser = JsonOutputParser(pydantic_object=CommandDescription)
        prompt = mlflow.genai.load_prompt("prompts:/generate-bash-description/12").template
        
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["command"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template_format="mustache"
        )
        
        llm = server_pool.create_llm()
        chain = prompt_template | llm | StrOutputParser() | clean_json_output | parser
        
        result = chain.invoke({"command": command})
        return {
            "command": command,
            "reasoning": result.get("reasoning", ""),
            "description": re.sub(r"i want to", "", result.get("description", ""), flags=re.IGNORECASE),
            "is_command": result.get("is_command", False)
        }
    except ConnectionError as ce:
        print(f"Connection error for command '{command}': {str(ce)}")
        return None
    except Exception as e:
        print(f"Error processing command '{command}': {str(e)}")
        return None

@task(
    name="process-command-batch",
    description="Обработка батча команд",
    log_prints=False,
    cache_policy=NO_CACHE
)
def process_command_batch(batch_commands, batch_number, server_pool, total_commands, processed_before_batch, max_workers):
    """Subflow для обработки батча команд"""
    logger = get_run_logger()
    logger.info(f"🔄 Обрабатываю батч {batch_number} ({len(batch_commands)} команд)")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_command = {
            executor.submit(process_single_command_internal, command, server_pool): command 
            for command in batch_commands
        }
        
        for future in as_completed(future_to_command):
            command = future_to_command[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                 logger.error(f"❌ Ошибка при обработке команды '{command}': {str(e)}")
    
    processed_so_far = processed_before_batch + len(results)
    remaining = total_commands - processed_so_far
    logger.info(f"📊 Батч {batch_number} завершен. Обработано: {processed_so_far}/{total_commands}, осталось: {remaining}")
    
    return results

@task(
    name="save-batch-results",
    description="Сохранение батча результатов в CSV",
    cache_policy=NO_CACHE
)
def save_batch_results(batch_results, output_csv):
    logger = get_run_logger()
    """Save a batch of results to CSV (thread-safe)"""
    if not batch_results:
        return
        
    valid_results = [r for r in batch_results if r is not None]
    if not valid_results:
        return
        
    lock = threading.Lock()
    with lock:
        try:
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["command", "reasoning", "description", "is_command"])
                if f.tell() == 0:
                    writer.writeheader()
                for result in valid_results:
                    writer.writerow(result)
            logger.info(f"✅ Saved batch of {len(valid_results)} results")
        except Exception as e2:
            logger.info(f"❌ Save failed: {e2}")

@flow(
    name="command-description-pipeline",
    description="Генерация описаний bash команд через LLM с пулом серверов",
    task_runner=ThreadPoolTaskRunner(max_workers=5),
    log_prints=True,
    persist_result=True
)
def command_description_flow(
    input_csv: str,
    output_csv: str,
    base_urls: List[str],
    command_column: str = "command",
    max_workers: int = 5,
    limit: int|None = None,
    timeout: int = 120,
    batch_size: int = 15
):
    """
    Prefect flow для генерации описаний команд через пул LLM серверов
    """
    print(f"🚀 Starting command description pipeline")
    print(f"   Input: {input_csv}")
    print(f"   Output: {output_csv}")
    print(f"   Servers: {base_urls}")
    print(f"   Max workers: {max_workers}")  # Теперь правильно отображается
    print(f"   Timeout: {timeout}s")
    print(f"   Batch size: {batch_size}")
    
    # Инициализация пула серверов
    server_pool = ServerPoolManager(base_urls, timeout)
    
    # Чтение входных данных
    df = pd.read_csv(input_csv)
    commands = df[command_column].unique().tolist()
    
    if limit:
        commands = commands[:limit]
        print(f"Limited to {limit} commands")
    
    # Исключаем уже обработанные команды
    if os.path.exists(output_csv):
        existing_data = pd.read_csv(output_csv)
        existing_commands = existing_data[command_column].tolist()
        commands = [cmd for cmd in commands if cmd not in existing_commands]
        print(f"Found {len(existing_commands)} existing commands, {len(commands)} new to process")
    else:
        print(f"Found {len(commands)} commands to process")
    
    if not commands:
        print("✅ No new commands to process")
        return {"status": "completed", "processed": 0, "reason": "No new commands"}
    
    # Разбиваем команды на батчи
    batches = [commands[i:i + batch_size] for i in range(0, len(commands), batch_size)]
    print(f"📦 Created {len(batches)} batches")
    
    # Обработка батчей
    all_results = []
    processed_before_batch = 0
    
    for batch_num, batch_commands in enumerate(batches, 1):
        print(f"🔄 Starting batch {batch_num}/{len(batches)}")
        
        # Обрабатываем батч в subflow
        batch_results = process_command_batch(
            batch_commands=batch_commands,
            batch_number=batch_num,
            server_pool=server_pool,
            total_commands=len(commands),
            processed_before_batch=processed_before_batch,
            max_workers=max_workers
        )
        
        if batch_results:
            save_batch_results(batch_results, output_csv)
            all_results.extend(batch_results)
            processed_before_batch += len(batch_results)
    
    if os.path.exists(output_csv):
        final_df = pd.read_csv(output_csv)
        total_commands = len(final_df)
        successful = len(final_df[final_df['is_command'] == True])
        
        print(f"\n🎉 Processing complete!")
        print(f"📊 Total commands in output: {total_commands}")
        print(f"✅ Successful descriptions: {successful}")
        print(f"📈 Success rate: {successful/total_commands:.1%}" if total_commands > 0 else "N/A")
        print(f"💾 Results saved to: {output_csv}")
        
        return {
            "status": "completed",
            "output_file": output_csv,
            "total_commands": total_commands,
            "successful_descriptions": successful,
            "success_rate": successful/total_commands if total_commands > 0 else 0
        }
    else:
        return {
            "status": "failed",
            "error": "No output file created"
        }

if __name__ == "__main__":
    command_description_flow.serve(
        name="command-description-pipeline",
        parameters={
            "input_csv": "bash_commands.csv",
            "output_csv": "generated_data.csv",
            "base_urls": ["http://192.168.100.228:8000", "http://192.168.100.219:8000"], 
            "max_workers": 5,
            "batch_size": 15
        }
    )