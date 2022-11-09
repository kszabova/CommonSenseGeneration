import time
import requests

import networkx as nx

from typing import NamedTuple

BASE_URL = "http://api.conceptnet.io/"
QUERY_URL = "/c/en/{word}?limit=1000"
DATA_DIR = "./data/"
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
    class Resources(NamedTuple):
        concept2id: dict
        id2concept: dict
        relation2id: dict
        id2relation: dict

    def __init__(
        self,
        base_url=BASE_URL,
        query_url=QUERY_URL,
        allowed_relations=ALLOWED_RELATIONS,
    ):
        self.base_url = base_url
        self.query_url = query_url
        self.allowed_relations = allowed_relations

        self.graph = None
        self.resources = None

    def query_api(self, word):
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

    def query_local(self, **kwargs):
        # load resources and graph
        if not self.resources:
            self._load_resources()

        assert self.resources.concept2id is not None

        graph_path = DATA_DIR + "conceptnet.graph"
        if not self.graph:
            self.graph = nx.read_gpickle(graph_path)

        if kwargs["mode"] == "shortest_path":
            return self._local_shortest_path(kwargs["start"], kwargs["end"])
        if kwargs["mode"] == "edge":
            return self._local_edge(kwargs["start"], kwargs["end"])
        if kwargs["mode"] == "convert":
            return self._local_convert(kwargs["resource"], kwargs["item"])
        else:
            raise RuntimeError("Unknown mode for local Conceptnet query")

    def _load_resources(self):
        concept_path = DATA_DIR + "concept.txt"
        relation_path = DATA_DIR + "relation.txt"

        concept2id = {}
        id2concept = {}
        with open(concept_path, "r", encoding="utf8") as f:
            for w in f.readlines():
                concept2id[w.strip()] = len(concept2id)
                id2concept[len(id2concept)] = w.strip()
        print("Finished loading 'concept' resources")

        id2relation = {}
        relation2id = {}
        with open(relation_path, "r", encoding="utf8") as f:
            for w in f.readlines():
                id2relation[len(id2relation)] = w.strip()
                relation2id[w.strip()] = len(relation2id)
        print("Finished loading 'relation' resources")

        self.resources = self.Resources(
            concept2id, id2concept, relation2id, id2relation
        )

    def _local_shortest_path(self, start, end):
        start_id, end_id = (
            self.resources.concept2id.get(start),
            self.resources.concept2id.get(end),
        )
        if not start_id or not end_id:
            return None, None
        if not start_id in self.graph or not end_id in self.graph:
            return None, None
        end_to_start, start_to_end = None, None
        if nx.has_path(self.graph, start_id, end_id):
            end_to_start = nx.shortest_path(self.graph, start_id, end_id)
        if nx.has_path(self.graph, end_id, start_id):
            start_to_end = nx.shortest_path(self.graph, end_id, start_id)
        return end_to_start, start_to_end

    def _local_edge(self, start, end):
        return self.graph.get_edge_data(start, end).values()

    def _local_convert(self, resource, item):
        resource_name = {
            "concept2id": self.resources.concept2id,
            "id2concept": self.resources.id2concept,
            "relation2id": self.resources.relation2id,
            "id2relation": self.resources.id2relation,
        }
        resource = resource_name[resource]
        return resource.get(item)

