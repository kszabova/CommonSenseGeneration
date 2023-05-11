import json
import random


class InputEnhancer:
    allowed_enhancements = [
        "basic",
        "all_keywords",
        "pair",
        "subgraph",
        "spantree",
        "gnn",
    ]

    def __init__(
        self, enhancement_type, enhancement_file, cls_token, sep_token
    ) -> None:
        self.enhancement_type = enhancement_type
        self.enhancement_file = enhancement_file
        self.enhancement = self.setup_enhancement()

        self.cls_token = cls_token
        self.sep_token = sep_token

    def setup_enhancement(self):
        if self.enhancement_type in self.allowed_enhancements:
            with open(self.enhancement_file, "r") as file:
                return json.load(file)
        return {}

    def get_enhanced_input(self, concepts):
        enhancement_fct_map = {
            None: self.get_no_enhancement_input,
            "basic": self.get_basic_enhanced_input,
            "all_keywords": self.get_all_keywords_enhanced_input,
            "pair": self.get_pair_enhanced_input,
            "subgraph": self.get_subgraph_enhanced_input,
            "spantree": self.get_spantree_enhanced_input,
            "gnn": self.get_gnn_enhanced_input,
        }
        return enhancement_fct_map[self.enhancement_type](concepts)

    def get_no_enhancement_input(self, concepts):
        return " ".join([str(concept) for concept in concepts])

    def get_basic_enhanced_input(self, concepts):
        # select a random sentence for a random concept
        input = " ".join(concepts)
        sentences = []
        for concept in concepts:
            concept_sentences = self.enhancement.get(concept, {}).get("sentences", [])
            if concept_sentences:
                sentences.append(random.choice(concept_sentences))
        input += (
            " "
            + self.cls_token
            + " "
            + (random.choice(sentences) if sentences else "")
            + " "
            + self.sep_token
        )
        return input

    def get_all_keywords_enhanced_input(self, concepts):
        # select a random sentence for each of the concepts
        input = " ".join(concepts)
        sentences = []
        for concept in concepts:
            concept_sentences = self.enhancement.get(concept, {}).get("sentences", [])
            if concept_sentences:
                sentences.append(random.choice(concept_sentences))
        input += (
            "".join([f" {self.cls_token} {sentence} " for sentence in sentences])
            + self.sep_token
        )
        return input

    def get_pair_enhanced_input(self, concepts):
        # select a sentence that contains at least two of the concepts
        input = " ".join(concepts)
        sentences = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                intersection = (
                    self.enhancement.get(concepts[i], {})
                    .get(concepts[j], {})
                    .get("sentences", [])
                )
                sentences.extend(intersection)
        sentence = random.choice(sentences) if sentences else ""
        input += " " + self.cls_token + " " + sentence + " " + self.sep_token
        return input

    def get_subgraph_enhanced_input(self, concepts):
        input = " ".join(concepts)
        sentences = []
        concept_subgraph = self.enhancement.get(" ".join(concepts), {})
        for edge in concept_subgraph.values():
            path = random.choice(edge)
            sentences.append(" ".join(path))
        input += (
            "".join([f" {self.cls_token} {sentence} " for sentence in sentences])
            + self.sep_token
        )
        return input

    def get_spantree_enhanced_input(self, concepts):
        input = " ".join(concepts)
        sentences = self.enhancement.get(input, [])
        input += f" {self.sep_token} " + f" {self.sep_token} ".join(sentences)
        return input

    def get_gnn_enhanced_input(self, concepts):
        input = " ".join(concepts)
        input += f" {self.sep_token}"
        for concept in concepts:
            if concept in self.enhancement:
                input += " " + " ".join(self.enhancement[concept])
        return input

