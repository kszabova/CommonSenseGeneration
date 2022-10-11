import time
import requests

BASE_URL = "http://api.conceptnet.io/"
QUERY_URL = "/c/en/{word}?limit=1000"
ALLOWED_RELATIONS = set(
    [
        "/r/IsA",
        "/r/PartOf",
        "/r/HasA",
        "/r/UsedFor",
        "/r/CapableOf",
        "/r/AtLocation",
        "/r/Causes",
        "/r/HasSubservent",
        "/r/HasFirstSubservent",
        "/r/HasLastSubservent",
        "/r/HasPrerequisite",
        "/r/HasProperty",
        "/r/MotivatedByGoal",
        "/r/ObstructedBy",
        "/r/Desires",
        "/r/CreatedBy",
        "/r/MannerOf",
        "/r/LocatedNear",
        "/r/CausesDesire",
        "/r/MadeOf",
        "/r/ReceivesAction",
    ]
)


class Conceptnet:
    def __init__(
        self,
        base_url=BASE_URL,
        query_url=QUERY_URL,
        allowed_relations=ALLOWED_RELATIONS,
    ):
        self.base_url = base_url
        self.query_url = query_url
        self.allowed_relations = allowed_relations

    def query(self, word):
        relations = []
        query_url = self.query_url.format(word=word)
        while query_url:
            time.sleep(0.3)
            response = requests.get(self.base_url + query_url).json()
            for edge in response["edges"]:
                # check if we have a meaningful relation
                if not edge.get("rel", None).get("@id", None) in self.allowed_relations:
                    continue
                # check if both ends of the edge are English
                if (
                    edge["end"].get("language", "") != "en"
                    or edge["start"].get("language", "") != "en"
                ):
                    continue

                sentence = edge.get("surfaceText", None)
                start = edge["start"]["label"]
                end = edge["end"]["label"]
                if sentence:
                    # remove unnecessary tokens from the sentence
                    sentence = (
                        sentence.replace("[", "").replace("]", "").replace("*", "")
                    )
                    relations.append({"start": start, "end": end, "sentence": sentence})

            query_url = response.get("view", {}).get("nextPage", None)
        return relations
