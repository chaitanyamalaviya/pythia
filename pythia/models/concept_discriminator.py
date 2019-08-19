# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F

import itertools

from pythia.common.registry import registry
from pythia.modules.layers import WeightNormClassifier


@registry.register_model("concept_discriminator")
class ConceptDiscriminator:
    def __init__(self, config, word_embedding):
        super(ConceptDiscriminator, self).__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self.word_embedding = word_embedding

    def build(self):
        self._init_classifier()

    def _init_classifier(self):
        num_hidden = self.config["text_embedding"]["num_hidden"]
        dropout = self.config["classifier"]["dropout"]
        self.classifier = WeightNormClassifier(
            num_hidden * 2, 1, num_hidden * 2, dropout
        )

    def sample_plausible_concepts(self, object_ids, gold_concepts=None):
        plausible_concepts = []
        targets = None
        threshold = 0.6
        final_objects = []
        for obj in object_ids:
            if obj[1] > threshold and final_objects.count(obj[0]) < 2:
                final_objects.append(obj[0])

        for obj_pair in itertools.product(final_objects, repeat=2):
            relations = self.dataset.scene_graphs_db.find_seen_relations(*obj_pair)
            attr1 = self.dataset.scene_graphs_db.find_seen_attributes(obj_pair[0])
            attr2 = self.dataset.scene_graphs_db.find_seen_attributes(obj_pair[1])
            plausible_concepts += [obj_pair[0] + " " + rel + " " + obj_pair[1] for rel in relations]
            plausible_concepts += [attr + " " + obj_pair[0] for attr in attr1]
            plausible_concepts += [attr + " " + obj_pair[1] for attr in attr2]
            # TODO: additionally sample plausible concepts with GloVe embedding similarity?

        plausible_concepts = list(set(plausible_concepts))

        # Gold concepts available during training
        if gold_concepts:
            targets = [1 if concept in gold_concepts else 0 for concept in plausible_concepts]

        return plausible_concepts, targets

    def forward(self, sample_list, plausible_concepts):

        v = sample_list.image_feature_0
        num_assertions = [elem.size(0) for i, elem in enumerate(plausible_concepts)]
        concept_input = torch.cat(plausible_concepts, dim=0).cuda()
        concept_emb = self.get_embs_batchwise(concept_input)

        # split_c_emb = concept_emb.split(num_assertions, dim=0)

        # More efficient way to do below
        repeat_v = []
        for a_, size in zip(v, num_assertions):
            repeat_v.append(v_.repeat(size))
        repeat_v = torch.cat(repeat_v)

        classifier_input = torch.cat((repeat_v, concept_emb), dim=1)

        logits = self.classifier(classifier_input, dim=1)

        return {"concept_scores": logits}

    def get_embs_batchwise(self, c, batch_size=128):

        c_embs = []
        for i in range(0, c.size(0), batch_size):
            c_emb = self.word_embedding.forward(c[i : i+batch_size])
            c_embs.append(c_emb)

        return torch.cat(c_embs, dim=0)

    def filter_topk_concepts(self, scores, k=100):
        return torch.topk(scores, k, dim=1)
