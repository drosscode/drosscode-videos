import json
import os
import time

from requests import Session


def check_backoff(response, **kwargs):
    response_body = response.json()
    if backoff := response_body.get("backoff"):
        print(f"Backoff: {backoff}")
        time.sleep(backoff + 1)


request_session = Session()

request_session.hooks["response"].append(check_backoff)


def get_data():
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    page = prev_result.get("last_page", 0) + 1
    request_url = "https://api.stackexchange.com/2.3/questions"

    params = {
        "pagesize": 100,
        "order": "asc",
        "sort": "creation",
        "site": "stackoverflow",
        "filter": "!gaGFJ4-CdHdnU*LYnQ(vB6sD9w1JlKafQt81vI-CB1mZuXM",
        "page": page,
    }

    if key:
        params["key"] = key

    if access_token:
        params["access_token"] = access_token

    loop_flag = True

    try:
        while loop_flag:
            print(f"Page: {page}")
            params["page"] = page
            response = request_session.get(
                request_url,
                params=params,
            )

            response_body = response.json()

            with open(f"{data_folder}/page-{page}.txt", "w") as df:
                json.dump(
                    response_body,
                    df,
                )

            quota_remaining = response_body["quota_remaining"]

            print(f"Quota remaining: {quota_remaining}")
            loop_flag = quota_remaining >= min_quota
            page += 1
    finally:
        with open(result_file, "w") as rf:
            json.dump(
                {"last_page": page - 1},
                rf,
            )


if __name__ == "__main__":
    data_folder = os.path.expanduser(
        "~/Documents/code-data/2-stackoverflow-generator/raw-data"
    )
    config_file = "config.json"
    result_file = "results.json"
    min_quota = 10
    access_token = None
    key = None
    config = None
    prev_result = {}

    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
            key = config["key"]
            access_token = config["access_token"]

    if os.path.exists(result_file):
        with open(result_file) as f:
            prev_result = json.load(f)

    get_data()
