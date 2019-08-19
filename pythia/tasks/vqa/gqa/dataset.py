import os
import json

import numpy as np
import torch

from PIL import Image

from pythia.common.registry import registry
from pythia.common.sample import Sample
from pythia.tasks.base_dataset import BaseDataset
from pythia.utils.general import get_pythia_root
from pythia.tasks.features_dataset import FeaturesDataset
from pythia.tasks.image_database import ImageDatabase
from pythia.tasks.scene_graph_dataset import SceneGraphDatabase
from pythia.tasks.object_class_dataset import ObjectClassDatabase
from pythia.utils.text_utils import VocabFromText, tokenize
from pythia.utils.distributed_utils import is_main_process, synchronize


_CONSTANTS = {
    "questions_folder": "questions",
    "dataset_key": "gqa",
    "empty_folder_error": "GQA dataset folder is empty.",
    "questions_key": "questions",
    "question_key": "question",
    "answer_key": "answer",
    "train_dataset_key": "train",
    "images_folder": "allImages",
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for GQA is not present."
}


class GQADataset(BaseDataset):
    """Dataset for GQA.

    Args:
        dataset_type (str): type of dataset, train|val|test
        config (ConfigNode): Configuration Node representing all of the data necessary
                             to initialize GQA dataset class
        data_folder: Root folder in which all of the data will be present if passed
                     replaces default based on data_root_dir and data_folder in config.

    """

    def __init__(self, dataset_type, imdb_file_index, config, *args, **kwargs):
        super().__init__("gqa", dataset_type, config)
        imdb_files = self.config.imdb_files

        if dataset_type not in imdb_files:
            raise ValueError(
                "Dataset type {} is not present in "
                "imdb_files of dataset config".format(dataset_type)
            )

        self.imdb_file = imdb_files[dataset_type][imdb_file_index]
        self.imdb_file = self._get_absolute_path(self.imdb_file)
        self.imdb = ImageDatabase(self.imdb_file)

        self.kwargs = kwargs

        self.image_depth_first = self.config.image_depth_first
        self._should_fast_read = self.config.fast_read

        self._use_features = False
        if hasattr(self.config, "image_features"):
            self._use_features = True
            self.features_max_len = self.config.features_max_len
            all_image_feature_dirs = self.config.image_features["spatial"]
            curr_image_features_dir = all_image_feature_dirs
            curr_image_features_dir = self._get_absolute_path(curr_image_features_dir)

            self.features_db = FeaturesDataset(
                "coco",
                directories=curr_image_features_dir,
                depth_first=self.image_depth_first,
                max_features=self.features_max_len,
                fast_read=self._should_fast_read,
                imdb=self.imdb,
            )

        self._use_scene_graphs = False
        if self.config["use_scene_graphs"]:
            self._use_scene_graphs = True
            if dataset_type == "test": dataset_type="val"
            path = self.config.scene_graphs[dataset_type][0]
            path = self._get_absolute_path(path)
            self.scene_graphs_db = SceneGraphDatabase(
                path
            )

        if hasattr(self.config, "object_classes"):
            path = self.config.object_classes[0]
            path = self._get_absolute_path(path)
            self.object_classes = ObjectClassDatabase(
                path
            )

    def _get_absolute_path(self, paths):
        if isinstance(paths, list):
            return [self._get_absolute_path(path) for path in paths]
        elif isinstance(paths, str):
            if not os.path.isabs(paths):
                pythia_root = get_pythia_root()
                paths = os.path.join(pythia_root, self.config.data_root_dir, paths)
            return paths
        else:
            raise TypeError(
                "Paths passed to dataset should either be " "string or list"
            )

    def __len__(self):
        return len(self.imdb)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            self.writer.write(
                "Starting to fast read {} {} dataset".format(
                    self._name, self._dataset_type
                )
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.imdb)), miniters=100, disable=not is_main_process()
            ):
                self.cache[idx] = self.load_item(idx)

    def get_item(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.imdb[idx]
        current_sample = Sample()

        text_processor_argument = {"tokens": sample_info["question_tokens"]}
        processed_question = self.text_processor(text_processor_argument)
        current_sample.text = processed_question["text"]

        #q_id = int(sample_info["question_id"])
        #current_sample.question_id = torch.tensor(
        #    q_id, dtype=torch.int
        #)
        current_sample.question_id = sample_info["question_id"]

        #if str.isdigit(sample_info["image_id"]):
        #    img_id = int(sample_info["image_id"])
        #    current_sample.image_id = torch.tensor(
        #        img_id, dtype=torch.int
        #    )
        #else:
        current_sample.image_id = sample_info["image_id"]

        current_sample.text_len = torch.tensor(
            len(sample_info["question_tokens"]), dtype=torch.int
        )

        if self._use_features is True:
            features = self.features_db[idx]
            current_sample.update(features)

        if self._use_scene_graphs:
            scene_graph = self.scene_graphs_db[sample_info["image_id"]]
            current_sample["scene_graph"] = []
            for assertion in scene_graph:
                text_processor_argument = {"tokens": assertion}
                processed_question = self.text_processor(text_processor_argument)
                current_sample["scene_graph"].append(processed_question["text"])
            #current_sample["scene_graph"] = torch.cat(current_sample["scene_graph"], dim=0)

        current_sample["object_classes"] = self.object_classes[sample_info["image_id"]]
        current_sample["rel_dict"] = self.scene_graphs_db.rel_dict
        current_sample["attr_dict"] = self.scene_graphs_db.attr_dict

        # # Depending on whether we are using soft copy this can add
        # # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)

        #import pdb; pdb.set_trace()
        #order_vectors = torch.eye(self.config.context_max_len)
        #order_vectors[self.config.context_max_len :] = 0
        #current_sample.order_vectors = order_vectors

        return current_sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            sample.answers = processed_soft_copy_answers["answers"]
            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)


    def format_for_evalai(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == self.text_processor.PAD_TOKEN:
                answer = "unanswerable"

            predictions.append({"questionId": question_id, "prediction": answer})

        return predictions
