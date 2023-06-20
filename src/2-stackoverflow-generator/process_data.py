import json
import os
import random
import shutil
from urllib.parse import unquote
from html import unescape
import pathlib

sizer = 100


def process_data():
    i = 0

    choice_array = []

    for k, v in file_split.items():
        choice_array += [k] * int(v * sizer)
        pathlib.Path(f"{processed_folder}/tags/{k}").mkdir(parents=True)
        pathlib.Path(f"{processed_folder}/titles/{k}").mkdir(parents=True)
        pathlib.Path(f"{processed_folder}/questions/{k}").mkdir(parents=True)
        pathlib.Path(f"{processed_folder}/answers/{k}").mkdir(parents=True)

    random.shuffle(choice_array)

    for file_name in os.listdir(data_folder):
        with open(f"{data_folder}/{file_name}", "r") as f:
            content = json.load(f)

        subfolder = random.choice(choice_array)

        for item in content["items"]:
            tags = " ".join(item["tags"])
            title = unescape(unquote(item["title"]))
            question_body = unescape(unquote(item["body_markdown"]))

            with open(f"{processed_folder}/tags/{subfolder}/{i}.txt", "w") as f:
                f.write(f"{tags}\n")

            with open(f"{processed_folder}/titles/{subfolder}/{i}.txt", "w") as f:
                f.write(f"{title}\n")

            with open(f"{processed_folder}/questions/{subfolder}/{i}.txt", "w") as f:
                f.write(f"{question_body}\n")

            if "answers" in item:
                with open(f"{processed_folder}/answers/{subfolder}/{i}.txt", "w") as f:
                    f.write("\n".join(a["body_markdown"] for a in item["answers"]) + "\n")

            i += 1

            if i >= config["limit"]:
                return


if __name__ == "__main__":
    data_folder = os.path.expanduser(
        "~/Documents/code-data/2-stackoverflow-generator/raw-data"
    )
    processed_folder = "processed"

    config_file = "config.json"
    file_split = {}

    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            file_split = config.get("file_split", {
                "train": 0.8,
                "validate": 0.2
            })

    if os.path.exists(processed_folder) and os.path.isdir(processed_folder):
        shutil.rmtree(processed_folder)

    process_data()
