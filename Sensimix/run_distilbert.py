import os
import subprocess

PYTHONPATH = os.getcwd()
GLUE_DIR = "/media/shared/Devshree/edgellm3/glue_data"

os.environ["PYTHONPATH"] = PYTHONPATH

datasets = [
    ("cola",  f"{GLUE_DIR}/CoLA"),
    ("wnli",  f"{GLUE_DIR}/WNLI"),
    ("sts-b", f"{GLUE_DIR}/STS-B"),
    ("rte",   f"{GLUE_DIR}/RTE"),
    ("sst-2", f"{GLUE_DIR}/SST-2"),
    ("qnli",  f"{GLUE_DIR}/QNLI"),
    ("qqp",   f"{GLUE_DIR}/QQP"),
    ("mnli",  f"{GLUE_DIR}/MNLI"),
]

for task, data_dir in datasets:
    print(f"==========================================")
    print(f"Running: DistilBERT on {task}")
    print(f"==========================================")

    # Start GPU logger
    gpu_log = f"distilbert_{task}_gpu_log.csv"
    gpu_logger = subprocess.Popen(
        f"nvidia-smi --query-gpu=timestamp,power.draw,memory.used,utilization.gpu "
        f"--format=csv,noheader -l 1 > {gpu_log}",
        shell=True
    )

    # Run training and evaluation
    cmd = [
        "python", "src/examples/run_glue_new.py",
        "--model_type", "distilbert",
        "--model_name_or_path", "distilbert-base-uncased",
        "--task_name", task,
        "--do_train",
        "--do_eval",
        "--data_dir", data_dir,
        "--max_seq_length", "128",
        "--per_gpu_train_batch_size", "16",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "3",
        "--output_dir", f"outputs/distilbert_{task}",
        "--quantized_model_dir", f"quantized_models/distilbert_{task}",
        "--save_quantized_model",
        "--overwrite_output_dir",
    ]
    subprocess.run(cmd)

    # Stop GPU logger
    gpu_logger.terminate()
    print(f"Done: DistilBERT on {task}")

print("All DistilBERT datasets completed.")
