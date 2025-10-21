import json
import pandas as pd
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import pydantic
from tqdm import tqdm
import re
from langchain.schema.output_parser import StrOutputParser
from json_repair import repair_json

import mlflow
from datetime import datetime

os.environ["OPENAI_API_KEY"] = "noway"


def clean_json_output(text):
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
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


def evaluate_prompt(
    prompt_uri,
    dataset_path,
    command_column="command",
    model="local-model",
    max_workers=1,
):
    """
    Evaluate a prompt from MLflow against a dataset and log results to MLflow
    """
    # Set up MLflow experiment
    mlflow.set_experiment("bash-prompt-evaluation")

    # Start MLflow run
    with mlflow.start_run():
        # Load prompt from MLflow
        try:
            prompt_template = mlflow.genai.load_prompt(prompt_uri).template
            print(f"✅ Successfully loaded prompt from: {prompt_uri}")
        except Exception as e:
            print(f"❌ Failed to load prompt from {prompt_uri}: {e}")
            return

        # Initialize LLM
        llm = ChatOpenAI(
            api_key="unused",
            base_url="http://localhost:1234/v1",
            temperature=0.1,
            max_completion_tokens=2048,
            model=model,
        )

        mlflow.log_param("model-name", llm.model_name)
        mlflow.log_param("prompt", prompt_uri)

        # Set up parser and chain
        parser = JsonOutputParser(pydantic_object=CommandDescription)

        prompt_template_obj = PromptTemplate(
            template=prompt_template,
            input_variables=["command"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            template_format="mustache",
        )

        chain = (
            prompt_template_obj | llm | StrOutputParser() | clean_json_output | parser
        )

        # mlflow.langchain.log_model(chain, prompts=[prompt_template])

        # Read dataset
        try:
            df = pd.read_csv(dataset_path)
            commands = df[command_column].unique().tolist()

            mlflow.log_param("total_commands", len(commands))
            print(f"📊 Loaded dataset with {len(commands)} commands")

        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            mlflow.log_param("error", f"Dataset loading failed: {e}")
            return

        results = []

        def process_command(command):
            """Process a single command and return the result"""

            try:
                result = chain.invoke({"command": command})

                command_result = {
                    "command": command,
                    "reasoning": result.get("reasoning", ""),
                    "description": re.sub(
                        r"i want to",
                        "",
                        result.get("description", ""),
                        flags=re.IGNORECASE,
                    ),
                    "is_command": result.get("is_command", False),
                    "processing_success": True,
                    "error": "",
                }

                return command_result

            except Exception as e:
                print(f"❌ Error processing command '{command[:50]}...': {e}")

                return {
                    "command": command,
                    "reasoning": "",
                    "description": "",
                    "is_command": False,
                    "processing_success": False,
                    "error": str(e),
                }

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all commands
            future_to_command = {
                executor.submit(process_command, command): command
                for command in commands
            }

            # Process results as they complete
            for future in tqdm(as_completed(future_to_command), total=len(commands)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"❌ Future error: {e}")

        # Save results as artifact
        results_df = pd.DataFrame(results)
        d = pd.merge(
            left=results_df,
            right=df[["command", "real_is_command"]],
            how="left",
            on="command",
        )
        dataset = mlflow.data.from_pandas(
            d, predictions="is_command", targets="real_is_command"
        )
        mlflow.log_input(dataset, context="data")
        results_csv_path = f"evaluations/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        d.to_csv(results_csv_path, index=False)
        mlflow.log_artifact(results_csv_path, "model-results")

        # Log summary statistics
        mlflow.evaluate(data=dataset, model_type="classifier")

        # Print summary
        print(f"   Results saved to: {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a prompt from MLflow against a dataset"
    )
    parser.add_argument(
        "--prompt-uri",
        type=str,
        required=True,
        help='MLflow prompt URI (e.g., "prompts:/generate-bash-description/2")',
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to input CSV dataset"
    )
    parser.add_argument(
        "--command-column",
        type=str,
        default="command",
        help="Name of the command column",
    )
    parser.add_argument(
        "--max-workers", type=int, default=5, help="Number of parallel workers"
    )
    parser.add_argument(
        "--model", type=str, required=False, default="local-model", help="model"
    )

    args = parser.parse_args()

    evaluate_prompt(
        prompt_uri=args.prompt_uri,
        dataset_path=args.dataset,
        command_column=args.command_column,
        max_workers=args.max_workers,
        model=args.model,
    )
