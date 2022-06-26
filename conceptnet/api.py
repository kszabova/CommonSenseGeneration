import requests


def query_word(word):
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
            sentence = sentence.replace("[", "").replace("[", "").replace("*", "")
            sentences.append(sentence)

    return sentences

