import json
import os
import shutil
from urllib.parse import unquote
from html import unescape


def process_data():
    i = 0
    for file_name in os.listdir(data_folder):
        with open(f"{data_folder}/{file_name}", "r") as f:
            content = json.load(f)

        for item in content["items"]:
            tags = " ".join(item["tags"])
            title = unescape(unquote(item["title"]))
            question_body = unescape(unquote(item["body_markdown"]))

            with open(f"{processed_folder}/tags/{i}.txt", "w") as f:
                f.write(tags)

            with open(f"{processed_folder}/titles/{i}.txt", "w") as f:
                f.write(title)

            with open(f"{processed_folder}/questions/{i}.txt", "w") as f:
                f.write(question_body)

            if "answers" in item:
                with open(f"{processed_folder}/answers/{i}.txt", "w") as f:
                    f.write("\n".join(a["body_markdown"] for a in item["answers"]))

            i += 1

        break


if __name__ == "__main__":
    data_folder = "data"
    processed_folder = "processed"

    if os.path.exists(processed_folder) and os.path.isdir(processed_folder):
        shutil.rmtree(processed_folder)

    os.mkdir(processed_folder)
    os.mkdir(f"{processed_folder}/tags")
    os.mkdir(f"{processed_folder}/titles")
    os.mkdir(f"{processed_folder}/questions")
    os.mkdir(f"{processed_folder}/answers")

    process_data()
