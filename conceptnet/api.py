import time
import requests


def query_word_sentences(word):
    sentences = []
    object = requests.get(f"http://api.conceptnet.io/c/en/{word}").json()
    for edge in object["edges"]:
        # only get relations between English words
        if (
            edge["end"].get("language", "") != "en"
            or edge["start"].get("language", "") != "en"
        ):
            continue

        sentence = edge["surfaceText"]
        if sentence:
            sentence = sentence.replace("[", "").replace("]", "").replace("*", "")
            sentences.append(sentence)

    return sentences


def query_word_relations(word):
    relations = []
    base_url = "http://api.conceptnet.io"
    concept_url = f"/c/en/{word}?limit=1000"
    while concept_url:
        time.sleep(0.3)
        object = requests.get(base_url + concept_url).json()
        for edge in object["edges"]:
            if (
                edge["end"].get("language", "") != "en"
                or edge["start"].get("language", "") != "en"
            ):
                continue

            sentence = edge["surfaceText"]
            start = edge["start"]["label"]
            end = edge["end"]["label"]
            if sentence:
                sentence = sentence.replace("[", "").replace("]", "").replace("*", "")
                relations.append((start, end, sentence))

        concept_url = object.get("view", {}).get("nextPage", None)

    return relations
