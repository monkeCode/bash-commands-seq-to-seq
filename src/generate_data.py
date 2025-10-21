import json
import pandas as pd
import argparse
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

mlflow.set_tracking_uri("./mlruns")


def clean_json_output(text):
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
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
    reasoning: str = pydantic.Field(
        description="Detailed technical analysis of the command"
    )
    description: str = pydantic.Field(
        description="Concise summary of what the command does"
    )
    is_command: bool = pydantic.Field(
        description="command is valid and exists and no contains other text"
    )


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
            timeout=self.timeout,
        )


def generate_descriptions(
    input_csv,
    output_csv,
    base_urls,
    command_column="command",
    max_workers=5,
    limit=None,
    timeout=120,
):
    server_pool = ServerPoolManager(base_urls, timeout)

    parser = JsonOutputParser(pydantic_object=CommandDescription)
    prompt = mlflow.genai.load_prompt("prompts:/generate-bash-description/12").template

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["command"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template_format="mustache",
    )

    # Read input data
    df = pd.read_csv(input_csv)
    commands = df[command_column].unique().tolist()

    if limit:
        commands = commands[:limit]

    existing_data = pd.DataFrame()
    if os.path.exists(output_csv):
        existing_data = pd.read_csv(output_csv)
        existing_commands = existing_data[command_column].tolist()
        commands = [cmd for cmd in commands if cmd not in existing_commands]

    print(f"Found {len(commands)} new commands to process")
    print(f"Using server pool: {base_urls}")
    print(f"Request timeout: {timeout} seconds")

    file_lock = threading.Lock()

    def process_command(command):
        """Process a single command and return the result"""
        try:
            llm = server_pool.create_llm()
            chain = (
                prompt_template | llm | StrOutputParser() | clean_json_output | parser
            )

            result = chain.invoke({"command": command})
            r = {
                "command": command,
                "reasoning": result.get("reasoning", ""),
                "description": re.sub(
                    r"i want to", "", result.get("description", ""), flags=re.IGNORECASE
                ),
                "is_command": result.get("is_command", False),
            }
            return r
        except ConnectionError:
            print("Connection error")
            return None
        except Exception as e:
            print(f"Error processing command '{command}': {str(e)}")
            return None
            return {
                "command": command,
                "reasoning": "",
                "description": "",
                "is_command": False,
            }

    def save_results(batch_results):
        """Save a batch of results to CSV (thread-safe)"""
        with file_lock:
            new_df = pd.DataFrame(batch_results)

            if os.path.exists(output_csv):
                existing_df = pd.read_csv(output_csv)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            final_df.to_csv(output_csv, index=False)

    def save_single_result(result):
        with file_lock:
            try:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "command",
                            "reasoning",
                            "description",
                            "is_command",
                        ],
                    )
                    writer.writerow(result)
            except Exception as e:
                print(f"Error saving result for command '{result['command']}': {e}")

    def save_batch_results(batch_results):
        with file_lock:
            try:
                with open(output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "command",
                            "reasoning",
                            "description",
                            "is_command",
                        ],
                    )
                    writer.writerows(batch_results)
            except Exception as e:
                print(f"Error saving batch of {len(batch_results)} results: {e}")

    # Process commands with ThreadPoolExecutor
    results = []
    batch_size = 15

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all commands for processing
        future_to_command = {
            executor.submit(process_command, command): command for command in commands
        }

        for i, future in tqdm(
            enumerate(as_completed(future_to_command)), total=len(commands)
        ):
            try:
                result = future.result(timeout=timeout + 10)
                if result:
                    results.append(result)
                if len(results) >= batch_size:
                    save_batch_results(results)
                    results = []

            except Exception as e:
                print(f"Error waiting for future: {str(e)}")
                if results:
                    save_batch_results(results)
                    results = []

    # Save any remaining results
    if results:
        save_batch_results(results)
        print(f"Saved final batch of {len(results)} results")

    print(f"Processing complete. Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate command descriptions using LMstudio"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input CSV file with commands"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output CSV file"
    )
    parser.add_argument(
        "--base_urls",
        type=str,
        required=True,
        help='Comma-separated list of base URLs for server pool (e.g., "http://server1:1234/v1,http://server2:1234/v1")',
    )
    parser.add_argument(
        "--command_column",
        type=str,
        default="command",
        help="Name of the command column",
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of commands to process"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Timeout for API requests in seconds"
    )

    args = parser.parse_args()

    # Parse base URLs
    base_urls = [url.strip() for url in args.base_urls.split(",")]

    generate_descriptions(
        args.input,
        args.output,
        base_urls,
        args.command_column,
        args.max_workers,
        args.limit,
        args.timeout,
    )
