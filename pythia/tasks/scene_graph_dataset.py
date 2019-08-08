# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

import json

class SceneGraphDatabase(torch.utils.data.Dataset):
    """
    Dataset for SceneGraphs used in Pythia
    """

    def __init__(self, scene_graph_path):
        super().__init__()

        if not scene_graph_path.endswith(".json"):
            raise ValueError("Unknown file format for scene graph file")

        with open(scene_graph_path) as f:
            data = f.readlines()
            self.scene_graphs_db = json.loads(data[0])

        self.data = self.process_scene_graphs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, image_id):
        return self.data[image_id]

    def process_scene_graphs(self):
        image_assertions = {}
        for img_id in self.scene_graphs_db.keys():
            image_assertions[img_id] = []
            obj_id2name = {}
            objects = self.scene_graphs_db[img_id]["objects"]
            for obj_id in objects.keys():
                obj_id2name[obj_id] = objects[obj_id]["name"]
            for obj_id in objects.keys():
                obj_name = objects[obj_id]["name"]
                for rel in objects[obj_id]["relations"]:
                    rel_name = rel["name"]
                    target_obj = obj_id2name[rel["object"]]
                    assertion = obj_name + " " + rel_name + " " + target_obj
                    image_assertions[img_id].append(assertion.split(" "))

        return image_assertions
