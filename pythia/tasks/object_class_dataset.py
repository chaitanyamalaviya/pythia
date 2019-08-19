# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import json

class ObjectClassDatabase(torch.utils.data.Dataset):
    """
    Dataset for Object Classes used in Pythia
    """

    def __init__(self, file_path):
        super().__init__()

        if not file_path.endswith(".json"):
            raise ValueError("Unknown file format for object class file")

        with open(file_path) as f:
            data = f.readlines()
            self.data = json.loads(data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, image_id):
        return self.data[image_id]
