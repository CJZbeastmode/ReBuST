"""Module for img copying."""

import json


def copying(json_path: str, category: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in data:
        if data[i] == "KEEP":
            from_path = from_dest + i + "-" + category + ".svs"
            to_path = to_dest + i + "-" + category + ".svs"
            print(f"Copying {from_path} to {to_path}")
            with open(from_path, "rb") as fr:
                content = fr.read()
            with open(to_path, "wb") as fw:
                fw.write(content)
            print("Done.")


if __name__ == "__main__":
    from_dest = "/Volumes/Xbox_HD/data/med_img/"
    to_dest = "/Volumes/Xbox_HD/data/med_img_supervised_data_collection/"
    category = "STAD"
    json_path = f"TCGA-{category}.json"
    copying(json_path, category)
