# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import torch.nn.functional as F

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.models.concept_discriminator import ConceptDiscriminator
from pythia.modules.embeddings import BiLSTMTextEmbedding
from pythia.modules.layers import (BCNet, BiAttention, FCNet,
                                   WeightNormClassifier)


@registry.register_model("ban_with_concepts")
class BAN_with_Concepts(BaseModel):
    def __init__(self, config):
        super(BAN_with_Concepts, self).__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    def build(self):
        self._build_word_embedding()
        self._init_text_embedding()
        self._init_classifier()
        self._init_bilinear_attention()
        if self.config["concept_attention"]["attn_type"] == "bilinear":
            num_hidden = self.config["text_embedding"]["num_hidden"]
            self.concept_bilinear = nn.Bilinear(num_hidden*2, num_hidden*2, num_hidden)
        self.concept_discriminator = ConceptDiscriminator(self.config, self.word_embedding)

    def _build_word_embedding(self):
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embedding(self):
        module_config = self.config["text_embedding"]
        q_mod = BiLSTMTextEmbedding(
            module_config["num_hidden"],
            module_config["emb_size"],
            module_config["num_layers"],
            module_config["dropout"],
            module_config["bidirectional"],
            module_config["rnn_type"],
        )
        self.q_emb = q_mod

    def _init_bilinear_attention(self):
        module_config = self.config["bilinear_attention"]
        num_hidden = self.config["text_embedding"]["num_hidden"]
        v_dim = module_config["visual_feat_dim"]

        v_att = BiAttention(v_dim, num_hidden, num_hidden, module_config["gamma"])

        b_net = []
        q_prj = []

        for i in range(module_config["gamma"]):
            b_net.append(
                BCNet(
                    v_dim, num_hidden, num_hidden, None, k=module_config["bc_net"]["k"]
                )
            )

            q_prj.append(
                FCNet(
                    dims=[num_hidden, num_hidden],
                    act=module_config["fc_net"]["activation"],
                    dropout=module_config["fc_net"]["dropout"],
                )
            )

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.v_att = v_att

    def _init_classifier(self):
        num_hidden = self.config["text_embedding"]["num_hidden"]
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        dropout = self.config["classifier"]["dropout"]
        self.classifier = WeightNormClassifier(
            num_hidden*2, num_choices, num_hidden * 2, dropout
        )

    def forward(self, sample_list, print_concepts=False):

        plausible_concepts, targets = self.concept_discriminator.sample_plausible_concepts(sample_list.object_classes,
                                                                                           sample_list.scene_graph)
        plausible_concept_scores = self.concept_discriminator.forward(sample_list, plausible_concepts=plausible_concepts)
        # If sample_list.scene_graph is specified, use that. Else, use plausible concepts

        v = sample_list.image_feature_0
        q = self.word_embedding(sample_list.text)

        if print_concepts:
            scene_graph_text = self.get_text(sample_list.scene_graph, scene_graph=True)
            question_words = self.get_text(sample_list.text)

        scene_graph_input = [torch.stack(elem).cuda() if elem else torch.LongTensor([]).cuda()
                             for elem in sample_list.scene_graph]
        num_assertions = [elem.size(0) for i, elem in enumerate(scene_graph_input)]
        scene_graph_input = torch.cat(scene_graph_input, dim=0).cuda()

        c = self.word_embedding(scene_graph_input)

        q_emb = self.q_emb.forward_all(q)
        c_emb = self.get_embs_batchwise(c)

        b_emb = [0] * self.config["bilinear_attention"]["gamma"]
        att, logits = self.v_att.forward_all(v, q_emb)

        for g in range(self.config["bilinear_attention"]["gamma"]):
            g_att = att[:, g, :, :]
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, g_att)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        q_emb_sum = q_emb.sum(1)
        split_c_emb = c_emb.split(num_assertions, dim=0)
        topk_idxs = []

        if self.config["concept_attention"]["attn_type"] == "dot":
            c_emb_weighted_sum = []
            for i, curr_assertions in enumerate(split_c_emb):
                attn_scores = torch.mm(curr_assertions, q_emb_sum[i].unsqueeze(1))
                attn_probs = F.softmax(attn_scores, dim=0)
                topk_idxs.append(torch.topk(attn_probs, k=min(5, num_assertions[i]), dim=0)[1])
                c_emb_weighted = attn_probs * curr_assertions
                c_emb_weighted_sum.append(c_emb_weighted.sum(0))

        elif self.config["concept_attention"]["attn_type"] == "bilinear":
            attn_scores = self.concept_bilinear(q_emb_sum, c_emb)

        #c_emb_weighted = torch.bmm(attn_probs, c_emb)

        c_emb_weighted_sum = torch.stack(c_emb_weighted_sum, dim=0)
        logits = self.classifier(torch.cat((q_emb_sum, c_emb_weighted_sum), dim=1))

        if print_concepts:
            self.print_top_concepts(question_words, scene_graph_text, topk_idxs)

        return {"scores": logits, "concept_scores": plausible_concept_scores["concept_scores"], "concept_targets": targets}

    def get_embs_batchwise(self, c, batch_size=128):

        c_embs = []
        for i in range(0, c.size(0), batch_size):
            c_emb = self.q_emb.forward(c[i : i+batch_size])
            c_embs.append(c_emb)

        return torch.cat(c_embs, dim=0)

    def forward_alt(self, sample_list):

        v = sample_list.image_feature_0
        q = self.word_embedding(sample_list.text)

        scene_graph_input = [torch.stack(elem).cuda() if elem else torch.LongTensor([]).cuda() for elem in
                             sample_list.scene_graph]
        num_assertions = [elem.size(0) for i, elem in enumerate(scene_graph_input)]
        # idxs_sorted =  sorted(num_assertions, key=lambda x: x[0])
        scene_graph_input = torch.cat(scene_graph_input, dim=0).cuda()

        c = self.word_embedding(scene_graph_input)

        q_emb = self.q_emb.forward_all(q)
        c_emb = self.get_embs_batchwise(c)

        b_emb = [0] * self.config["bilinear_attention"]["gamma"]
        att, logits = self.v_att.forward_all(v, q_emb)

        for g in range(self.config["bilinear_attention"]["gamma"]):
            g_att = att[:, g, :, :]
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, g_att)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        q_emb_sum = q_emb.sum(1)
        split_c_emb = c_emb.split(num_assertions, dim=0)

        if self.config["concept_attention"]["attn_type"] == "dot":
            c_emb_weighted_sum = []
            for i, curr_assertions in enumerate(split_c_emb):
                attn_scores = torch.mm(curr_assertions, q_emb_sum[i].unsqueeze(1))
                attn_probs = F.softmax(attn_scores, dim=0)
                c_emb_weighted = attn_probs * curr_assertions
                c_emb_weighted_sum.append(c_emb_weighted.sum(0))

        elif self.config["concept_attention"]["attn_type"] == "bilinear":
            attn_scores = self.concept_bilinear(q_emb_sum, c_emb)

        # c_emb_weighted = torch.bmm(attn_probs, c_emb)

        c_emb_weighted_sum = torch.stack(c_emb_weighted_sum, dim=0)
        logits = self.classifier(torch.cat((q_emb_sum, c_emb_weighted_sum), dim=1))

        return {"scores": logits}

    def get_text(self, word_ids, scene_graph=False):

        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        idx2word = {v:k for k,v in vocab.vocab.word_dict.items()}
        all_words = []
        for word_seq in word_ids:
            curr_words = []
            if scene_graph:
                for assertion in word_seq:
                    words = [idx2word[elem.cpu().item()] for elem in assertion if idx2word[elem.cpu().item()]!="<pad>"]
                    curr_words.append(words)
                all_words.append(curr_words)
            else:
                words = [idx2word[elem.cpu().item()] for elem in word_seq if elem.nelement()!=0 and idx2word[elem.cpu().item()]!="<pad>"]
                all_words.append(words)
        return all_words

    def print_top_concepts(self, question_words, scene_graph_words, topk_idxs):
        for i, question in enumerate(question_words):
            print("\nQuestion:")
            print(" ".join(question))
            for idx in topk_idxs[i]:
                print(" ".join(scene_graph_words[i][idx.cpu().item()]))
