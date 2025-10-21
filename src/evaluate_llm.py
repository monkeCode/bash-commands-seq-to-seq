import pandas as pd
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm
import re
from langchain.schema.output_parser import StrOutputParser
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import mlflow
from datetime import datetime

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

os.environ["OPENAI_API_KEY"] = "noway"


def clean_command_output(text):
    text = re.sub(r"^```bash\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    text = re.sub(r'^["\'](.*)["\']$', r"\1", text.strip())

    return text.strip()


def calculate_bleu_scores(generated, reference):
    gen_tokens = generated.split()
    ref_tokens = reference.split()

    smoothing = SmoothingFunction().method1

    bleu2 = sentence_bleu(
        [ref_tokens], gen_tokens, weights=(0.5, 0.5), smoothing_function=smoothing
    )

    bleu4 = sentence_bleu(
        [ref_tokens],
        gen_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )

    return bleu2, bleu4


def evaluate_model(
    prompt_uri,
    dataset_path,
    intent_column="intent",
    command_column="command",
    model="local-model",
    max_workers=1,
):
    mlflow.set_experiment("bash-command-llm-eval")

    with mlflow.start_run():
        try:
            prompt_template = mlflow.genai.load_prompt(prompt_uri).template
            print(f"Successfully loaded prompt from: {prompt_uri}")
        except Exception as e:
            print(f"Failed to load prompt from {prompt_uri}: {e}")
            return

        llm = ChatOpenAI(
            api_key="unused",
            base_url="http://localhost:1234/v1",
            temperature=0,
            max_completion_tokens=2048,
            model=model,
        )

        mlflow.log_param("model_name", model)
        mlflow.log_param("prompt_uri", prompt_uri)

        prompt_template_obj = PromptTemplate(
            template=prompt_template,
            input_variables=["description"],
            template_format="mustache",
        )

        chain = prompt_template_obj | llm | StrOutputParser() | clean_command_output

        try:
            df = pd.read_csv(dataset_path)
            if intent_column not in df.columns or command_column not in df.columns:
                raise ValueError(
                    f"Dataset must contain '{intent_column}' and '{command_column}' columns"
                )

            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("intent_column", intent_column)
            mlflow.log_param("command_column", command_column)
            print(f"📊 Loaded dataset with {len(df)} examples")

        except Exception as e:
            print(f"Failed to load dataset: {e}")
            mlflow.log_param("error", f"Dataset loading failed: {e}")
            return

        results = []
        exact_matches = 0

        def process_example(row):
            intent = row[intent_column]
            reference_command = row[command_column]

            try:
                generated_command = chain.invoke({"description": intent})

                exact_match = generated_command.strip() == reference_command.strip()
                bleu2, bleu4 = calculate_bleu_scores(
                    generated_command, reference_command
                )

                result = {
                    "intent": intent,
                    "reference_command": reference_command,
                    "generated_command": generated_command,
                    "exact_match": exact_match,
                    "bleu2": bleu2,
                    "bleu4": bleu4,
                    "processing_success": True,
                    "error": "",
                }

                return result

            except Exception as e:
                print(f"Error processing intent '{intent[:50]}...': {e}")

                return {
                    "intent": intent,
                    "reference_command": reference_command,
                    "generated_command": "",
                    "exact_match": False,
                    "bleu2": 0.0,
                    "bleu4": 0.0,
                    "processing_success": False,
                    "error": str(e),
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_example = {
                executor.submit(process_example, row): row for _, row in df.iterrows()
            }

            for future in tqdm(as_completed(future_to_example), total=len(df)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if result["exact_match"]:
                            exact_matches += 1
                except Exception as e:
                    print(f"Future error: {e}")

        if results:
            successful_results = [r for r in results if r["processing_success"]]
            exact_match_rate = (
                exact_matches / len(successful_results) if successful_results else 0
            )
            avg_bleu2 = (
                sum(r["bleu2"] for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            )
            avg_bleu4 = (
                sum(r["bleu4"] for r in successful_results) / len(successful_results)
                if successful_results
                else 0
            )

            mlflow.log_metric("exact_match_rate", exact_match_rate)
            mlflow.log_metric("bleu2_score", avg_bleu2)
            mlflow.log_metric("bleu4_score", avg_bleu4)
            mlflow.log_metric(
                "successful_processing_rate", len(successful_results) / len(results)
            )

            print("Evaluation Results:")
            print(f"Exact Match Rate: {exact_match_rate:.4f}")
            print(f"Average BLEU-2: {avg_bleu2:.4f}")
            print(f"Average BLEU-4: {avg_bleu4:.4f}")
            print(f"Successful Processing: {len(successful_results)}/{len(results)}")

        results_df = pd.DataFrame(results)
        results_csv_path = f"evaluations/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_csv_path, index=False)

        mlflow.log_artifact(dataset_path, "dataset")
        mlflow.log_artifact(results_csv_path, "evaluation_results")

        dataset = mlflow.data.from_pandas(df)
        mlflow.log_input(dataset, context="evaluating")

        print(f"💾 Results saved to: {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate bash command recovery prompt from MLflow against a dataset"
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
        "--intent-column",
        type=str,
        default="description",
        help="Name of the intent/description column",
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
        "--model", type=str, required=False, default="local-model", help="Model name"
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri("http://mlflow.k3s.home")

    evaluate_model(
        prompt_uri=args.prompt_uri,
        dataset_path=args.dataset,
        intent_column=args.intent_column,
        command_column=args.command_column,
        max_workers=args.max_workers,
        model=args.model,
    )
