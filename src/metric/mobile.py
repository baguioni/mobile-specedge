import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.table import Table

# dataset offset (req_idx boundaries between task subsets)
TRANSLATION_OFFSET = 80
SUMMARIZATION_OFFSET = 160
QUESTION_REASONING_OFFSET = 240
MATHEMATICAL_REASONING_OFFSET = 320
RETRIEVAL_OFFSET = 400


def load_data(data_folder_path: Path):
    # Find all client files and the server file
    client_files = list(data_folder_path.glob("client_*.jsonl"))
    server_file = data_folder_path / "server.jsonl"

    if not client_files:
        print(
            "Error: No data files (client_*.jsonl or server.jsonl) "
            f"found in {data_folder_path}"
        )
        sys.exit(1)

    if not server_file.exists():
        print(f"Error: Server file not found: {server_file}")
        sys.exit(1)

    client_raw_data = []
    for file_path in client_files:
        try:
            with open(file_path, "r") as f:
                # Read and parse each line as JSON
                file_data = [json.loads(line) for line in f.readlines()]
                client_raw_data.extend(file_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            # Decide how to handle errors - for now, skip the file
            continue
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    if not client_raw_data:
        print("Error: No valid data records found in the specified files.")
        sys.exit(1)

    with open(server_file, "r") as f:
        try:
            server_raw_data = [json.loads(line) for line in f.readlines()]
        except json.JSONDecodeError as e:
            print(f"Error: Error decoding JSON from {server_file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error: Error reading file {server_file}: {e}")
            sys.exit(1)

    # Rest of the processing remains the same, operating on the combined raw_data
    client_df = pl.json_normalize(client_raw_data)
    server_df = pl.json_normalize(server_raw_data)

    return client_df, server_df


def overall_analysis(server_df: pl.DataFrame, client_df: pl.DataFrame, subset: str):
    server_start_time = datetime.fromisoformat(
        server_df.select(pl.first("timestamp")).item()
    )
    server_end_time = datetime.fromisoformat(
        server_df.select(pl.last("timestamp")).item()
    )

    match subset:
        case "multi_turn":
            client_df = client_df.filter(pl.col("req_idx") < TRANSLATION_OFFSET)
        case "translation":
            client_df = client_df.filter(
                (TRANSLATION_OFFSET <= pl.col("req_idx"))
                & (pl.col("req_idx") < SUMMARIZATION_OFFSET)
            )
        case "summarization":
            client_df = client_df.filter(
                (SUMMARIZATION_OFFSET <= pl.col("req_idx"))
                & (pl.col("req_idx") < QUESTION_REASONING_OFFSET)
            )
        case "question_answering":
            client_df = client_df.filter(
                (QUESTION_REASONING_OFFSET <= pl.col("req_idx"))
                & (pl.col("req_idx") < MATHEMATICAL_REASONING_OFFSET)
            )
        case "mathematical_reasoning":
            client_df = client_df.filter(
                (MATHEMATICAL_REASONING_OFFSET <= pl.col("req_idx"))
                & (pl.col("req_idx") < RETRIEVAL_OFFSET)
            )
        case "retrieval":
            client_df = client_df.filter(RETRIEVAL_OFFSET <= pl.col("req_idx"))
    server_time = server_end_time - server_start_time

    return {
        "draft": {
            "end_to_end": {
                "non-prefill": (
                    client_df.filter(pl.col("step_idx") != 0)
                    .select("draft.end_to_end")
                    .mean()
                    .item(),
                    client_df.filter(pl.col("step_idx") != 0)
                    .select("draft.end_to_end")
                    .std()
                    .item(),
                ),
                "prefill": (
                    client_df.filter(pl.col("step_idx") == 0)
                    .select("draft.end_to_end")
                    .mean()
                    .item(),
                    client_df.filter(pl.col("step_idx") == 0)
                    .select("draft.end_to_end")
                    .std()
                    .item(),
                ),
            }
        },
        "target": {
            "end_to_end": {
                "non-prefill": (
                    client_df.filter(pl.col("target.prefill") == 0)
                    .select("target.end_to_end")
                    .mean()
                    .item(),
                    client_df.filter(pl.col("target.prefill") == 0)
                    .select("target.end_to_end")
                    .std()
                    .item(),
                ),
                "prefill": (
                    client_df.filter(pl.col("target.prefill") != 0)
                    .select("target.end_to_end")
                    .mean()
                    .item(),
                    client_df.filter(pl.col("target.prefill") != 0)
                    .select("target.end_to_end")
                    .std()
                    .item(),
                ),
            },
            "server": {
                "non-prefill": (
                    server_df.filter(pl.col("target.prefill") == 0)
                    .select("target.server_end_to_end_t")
                    .mean()
                    .item(),
                    server_df.filter(pl.col("target.prefill") == 0)
                    .select("target.server_end_to_end_t")
                    .std()
                    .item(),
                ),
                "prefill": (
                    server_df.filter(pl.col("target.prefill") != 0)
                    .select("target.server_end_to_end_t")
                    .mean()
                    .item(),
                    server_df.filter(pl.col("target.prefill") != 0)
                    .select("target.server_end_to_end_t")
                    .std()
                    .item(),
                ),
            },
        },
        "overall": {
            "non-prefill": (
                client_df.filter(pl.col("target.prefill") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .mean()
                .item(),
                client_df.filter(pl.col("target.prefill") == 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .std()
                .item(),
            ),
            "prefill": (
                client_df.filter(pl.col("target.prefill") != 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .mean()
                .item(),
                client_df.filter(pl.col("target.prefill") != 0)
                .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
                .std()
                .item(),
            ),
        },
        "tokens": {
            "generated": client_df.select("num_accepted_tokens").sum().item(),
            "accepted": (
                client_df.select("num_accepted_tokens").mean().item(),
                client_df.select("num_accepted_tokens").std().item(),
            ),
        },
        "latency": {
            "value": client_df.filter(
                (pl.col("step_idx") != 0) & (pl.col("target.prefill") == 0)
            )
            .select(pl.col("draft.end_to_end") + pl.col("target.end_to_end"))
            .sum()
            .item()
            / client_df.filter(
                (pl.col("step_idx") != 0) & (pl.col("target.prefill") == 0)
            )
            .select(pl.col("num_accepted_tokens"))
            .sum()
            .item(),
        },
        "running_time": {
            "edge": client_df.select(
                pl.col("draft.end_to_end") + pl.col("target.end_to_end")
            )
            .sum()
            .item(),
            "server": server_time.total_seconds(),
        },
        "throughput": {
            "value": client_df.select("num_accepted_tokens").sum().item()
            / server_time.total_seconds()
        },
    }


def print_table(client_df: pl.DataFrame, server_df: pl.DataFrame, subset: str):
    console = Console()

    overall_table = Table(title="Overall")

    overall_table.add_column("Metric", justify="left")
    overall_table.add_column("Value", justify="right", min_width=20)
    overall_table.add_column("Std", justify="right", min_width=20)

    metrics = overall_analysis(server_df, client_df, subset)

    overall_table.add_row(
        "Draft (prefill)",
        f"{metrics['draft']['end_to_end']['prefill'][0]:.3f} ms",
        f"{metrics['draft']['end_to_end']['prefill'][1]:.3f}ms",
    )
    overall_table.add_row(
        "Draft (non-prefill)",
        f"{metrics['draft']['end_to_end']['non-prefill'][0]:.3f} ms",
        f"{metrics['draft']['end_to_end']['non-prefill'][1]:.3f}ms",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Target (prefill)",
        f"{metrics['target']['end_to_end']['prefill'][0]:.3f} ms",
        f"{metrics['target']['end_to_end']['prefill'][1]:.3f}ms",
    )
    overall_table.add_row(
        "Target (non-prefill)",
        f"{metrics['target']['end_to_end']['non-prefill'][0]:.3f} ms",
        f"{metrics['target']['end_to_end']['non-prefill'][1]:.3f}ms",
    )

    overall_table.add_section()
    overall_table.add_row(
        "Target (server, prefill)",
        f"{metrics['target']['server']['prefill'][0]:.3f} ms",
        f"{metrics['target']['server']['prefill'][1]:.3f}ms",
    )
    overall_table.add_row(
        "Target (server, non-prefill)",
        f"{metrics['target']['server']['non-prefill'][0]:.3f} ms",
        f"{metrics['target']['server']['non-prefill'][1]:.3f}ms",
    )

    overall_table.add_section()
    overall_table.add_row(
        "Overall (prefill)",
        f"{metrics['overall']['prefill'][0]:.3f} ms",
        f"{metrics['overall']['prefill'][1]:.3f}ms",
    )
    overall_table.add_row(
        "Overall (non-prefill)",
        f"{metrics['overall']['non-prefill'][0]:.3f} ms",
        f"{metrics['overall']['non-prefill'][1]:.3f}ms",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Accept Tokens",
        f"{metrics['tokens']['accepted'][0]:.2f}",
        f"{metrics['tokens']['accepted'][1]:.2f}",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Inter token latency",
        f"{metrics['latency']['value']:.3f} ms/tok",
    )
    overall_table.add_section()
    overall_table.add_row(
        "Server Running Time",
        f"{metrics['running_time']['server']:.3f} s",
    )
    overall_table.add_row(
        "Edge Running Time",
        f"{metrics['running_time']['edge'] / 1000:.3f} s",
    )
    overall_table.add_row(
        "Generated tokens",
        f"{metrics['tokens']['generated']}",
    )
    overall_table.add_row(
        "Throughput",
        f"{metrics['throughput']['value']:.3f} tokens/s",
    )

    console.print(overall_table)


def plain_text_print(client_df: pl.DataFrame, server_df: pl.DataFrame, subset: str):
    """
    Print the metrics in a plain text format.
    Some metrics are not printed in plain text format due to simplification.
    """
    metrics = overall_analysis(server_df, client_df, subset)

    values = [
        # Client Draft Latency (ms)
        f"{metrics['draft']['end_to_end']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['draft']['end_to_end']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['draft']['end_to_end']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['draft']['end_to_end']['non-prefill'][1]:.3f}",  # non-prefill_std
        # Client Target Latency (ms)
        f"{metrics['target']['end_to_end']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['target']['end_to_end']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['target']['end_to_end']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['target']['end_to_end']['non-prefill'][1]:.3f}",  # non-prefill_std
        # Server Target Latency (ms)
        f"{metrics['target']['server']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['target']['server']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['target']['server']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['target']['server']['non-prefill'][1]:.3f}",  # non-prefill_std
        # Client Overall Latency (ms)
        f"{metrics['overall']['prefill'][0]:.3f}",  # prefill_mean
        f"{metrics['overall']['prefill'][1]:.3f}",  # prefill_std
        f"{metrics['overall']['non-prefill'][0]:.3f}",  # non-prefill_mean
        f"{metrics['overall']['non-prefill'][1]:.3f}",  # non-prefill_std
        # Accepted Tokens per step (tokens): mean, std
        f"{metrics['tokens']['accepted'][0]:.2f}",  # mean
        f"{metrics['tokens']['accepted'][1]:.2f}",  # std
        # Client Inter-token Latency (non-prefill) (ms/tok)
        f"{metrics['latency']['value']:.3f}",
        # Server Total Running Time (s)
        f"{metrics['running_time']['server']:.3f}",
        # Client Total Processing Time (s)
        f"{metrics['running_time']['edge'] / 1_000:.3f}",
        # Total Accepted Tokens (tokens)
        f"{metrics['tokens']['generated']}",
        # Throughput (tokens/s)
        f"{metrics['throughput']['value']:.3f}",
    ]

    print("\t".join(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Path to the data folder")
    parser.add_argument(
        "-s",
        "--subset",
        type=str,
        choices=[
            "multi_turn",
            "translation",
            "summarization",
            "question_answering",
            "mathematical_reasoning",
            "retrieval",
            "overall",
        ],
        default="overall",
    )
    parser.add_argument("--plain", action="store_true", help="Use plain text data")
    args = parser.parse_args()

    data_folder_path = Path(args.data)
    subset = args.subset

    if not data_folder_path.is_dir():
        raise ValueError(f"Data path '{data_folder_path}' is not a valid directory")

    client_df, server_df = load_data(data_folder_path)

    if args.plain:
        plain_text_print(client_df, server_df, subset)
    else:
        print_table(client_df, server_df, subset)
