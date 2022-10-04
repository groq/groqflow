from abc import ABC, abstractmethod
from functools import partial
from itertools import zip_longest
import os
import random
import tarfile
from typing import Dict, List, Optional
import stat
import shutil
import math
import zipfile

from datasets import load_dataset
from datasets.utils.file_utils import cached_path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.ops import box_convert, nms
import torchvision.transforms as T
from tqdm import tqdm
from groqflow.common.build import DEFAULT_CACHE_DIR
from demo_helpers.misc import suppress_stdout


GROQFLOW_DATASETS_PATH = os.path.join(DEFAULT_CACHE_DIR, "datasets")
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_PATH = os.path.join(GROQFLOW_DATASETS_PATH, "imagenette2-320")
DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
IMAGENETTE_CLASS_MAP = {
    0: 0,
    1: 217,
    2: 482,
    3: 491,
    4: 497,
    5: 566,
    6: 569,
    7: 571,
    8: 574,
    9: 701,
}

MODELNET10_URL = (
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
)
MODELNET10_PATH = os.path.join(GROQFLOW_DATASETS_PATH, "ModelNet10")


random.seed(42)


def zero_pad(data, length=None):
    """
    Zero pad inner dimension of data to size `length`.
    Useful for making data be exact multiple of 320 before passing to chip.
    """
    x, y = data
    if length is not None:
        x = np.array(
            [
                np.pad(x_, (0, length - len(x_)), "constant", constant_values=0)
                for x_ in x
            ]
        )
    else:
        x = np.array(list(zip_longest(*x, fillvalue=0))).T
    y = np.array(y, dtype=np.float32)
    return x, y


def binarize_labels(data):
    """Convert probability output to 0/1 label prediction."""
    x, y = data
    y = (y > 0.5).astype(np.int64)
    return x, y


class Dataset(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name
        self.raw_data = None
        self.x = None
        self.y = None

    @abstractmethod
    def preprocess(self):
        pass

    def postprocess(self, output):
        return output


class BasicDataset(Dataset):
    def __init__(
        self, name: str, x: List[Dict[str, torch.Tensor]], y: np.ndarray
    ) -> None:
        super().__init__(name)
        self.name: str = name
        self.x = x
        self.y = y

    def preprocess(self) -> None:
        return


class LanguageDataset(Dataset):
    def __init__(self, name: str, tokenizer, max_seq_length) -> None:
        super().__init__(name)
        self.raw_data = load_dataset(self.name, split="test")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.x, self.y = self.preprocess()

    def preprocess(self):
        feature_names = self._get_data_features()
        x, y = (self.raw_data[feature_names[0]], self.raw_data[feature_names[1]])
        x = list(map(lambda x: self.tokenizer.encode(x, add_special_tokens=True), x))
        x, y = binarize_labels(zero_pad((x, y), length=self.max_seq_length))
        x = [
            {
                "input_ids": torch.unsqueeze(torch.from_numpy(x_), dim=0),
                "attention_mask": torch.unsqueeze(torch.from_numpy(x_ != 0), dim=0),
            }
            for x_ in x
        ]
        return x, y

    def _get_data_features(self) -> List[str]:
        feature_names = self.raw_data.features.keys()
        assert len(feature_names) >= 2
        feature_names = list(feature_names)[:2]
        return feature_names


class CoNLL2003Dataset(Dataset):
    def __init__(self, name: str, tokenizer, max_seq_length):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # load dataset. tokenize input and align labels to tokenized inputs
        self.raw_data = load_dataset(self.name, split="test").map(
            self.tokenize_and_align_labels, batched=True, load_from_cache_file=False
        )
        self.x, self.y = self.preprocess()

    def preprocess(self):
        # package as list of dictionaries
        x = [
            {
                "input_ids": torch.unsqueeze(torch.LongTensor(input_id), dim=0),
                "attention_mask": torch.unsqueeze(
                    torch.FloatTensor(attention_mask), dim=0
                ),
            }
            for input_id, attention_mask in zip(
                self.raw_data["input_ids"], self.raw_data["attention_mask"]
            )
        ]
        # get ground truth labels
        y = [label for label in self.raw_data["labels"]]
        return x, y

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=False,
            max_length=self.max_seq_length,
            padding="max_length",
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set
                # the label to -100 so they are automatically ignored
                # in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to
                # either the current label or -100, depending on the
                # label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            # update the labels to only include
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


class QuestionAnsweringDataset(Dataset):
    def __init__(self, name: str, tokenizer, max_seq_length) -> None:
        super().__init__(name)
        self.raw_data = load_dataset(self.name, split="validation")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.x, self.y, self.ids = self.preprocess()

    def preprocess(self):
        data = self.raw_data
        contexts, questions, answers = (
            data["context"],
            data["question"],
            data["answers"],
        )
        questions = [q.strip() for q in questions]
        answer_starts = []
        answer_ends = []
        answer_texts = []
        for context, answer in zip(contexts, answers):
            answer_texts.append(answer["text"])
            gold_text = answer["text"][0]
            start_idx = answer["answer_start"][0]
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer_starts.append(start_idx)
                answer_ends.append(end_idx)
            elif context[start_idx - 1 : end_idx - 1] == gold_text:
                # When the gold label is off by one character
                answer_starts.append(start_idx - 1)
                answer_ends.append(end_idx - 1)
            elif context[start_idx - 2 : end_idx - 2] == gold_text:
                # When the gold label is off by two characters
                answer_starts.append(start_idx - 2)
                answer_ends.append(end_idx - 2)
            else:
                raise ValueError(
                    "Gold Label off by more than two characters, check dataset for corruption"
                )

        return self.tokenize_qa(
            contexts, questions, answer_starts, answer_ends, answer_texts
        )

    def tokenize_qa(
        self,
        contexts,
        questions,
        answer_starts,
        answer_ends,
        answer_texts,
    ):
        """Convert raw Question Answering input data into tokenized format."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        x, y, texts, ids_list = [], [], [], []
        answer_ids = None

        print(f"Tokenizing Dataset {self.name} for Question Answering Task")
        for context, question, astart, aend, atext in tqdm(
            zip(contexts, questions, answer_starts, answer_ends, answer_texts),
            total=len(contexts),
        ):
            cxt = self.tokenizer(context)
            has_tt = "token_type_ids" in cxt
            qst = self.tokenizer(question, max_length=64, truncation=True)
            ids_qst = qst.input_ids[1:]
            answer_ids = self.tokenizer(atext).input_ids[0][1:-1]

            def get_tokens(start_char, end_char, cxt):
                start_tkn = cxt.char_to_token(start_char)
                end_tkn = cxt.char_to_token(end_char)
                if end_tkn is None:
                    end_tkn = start_tkn + len(answer_ids)

                if cxt.input_ids[start_tkn:end_tkn] != answer_ids:
                    raise ValueError
                assert start_tkn is not None and end_tkn is not None
                return start_tkn, end_tkn

            try:
                start, end = get_tokens(astart, aend, cxt)

                while end >= self.max_seq_length - len(ids_qst):
                    limit = cxt.token_to_chars(128).start
                    context = context[limit:]
                    astart = astart - limit
                    aend = aend - limit

                    cxt = self.tokenizer(context)
                    start, end = get_tokens(astart, aend, cxt)
            except ValueError:
                continue

            ids_cxt = self.tokenizer(
                context,
                max_length=self.max_seq_length - len(ids_qst),
                truncation=True,
            ).input_ids
            ids = ids_cxt + ids_qst
            assert len(ids) <= self.max_seq_length
            pad_amount = self.max_seq_length - len(ids)
            if pad_amount > 0:
                ids += [0] * (pad_amount)
            assert len(ids) == self.max_seq_length
            tt = [0] * len(ids_cxt) + [1] * len(ids_qst) + [0] * pad_amount
            assert len(tt) == self.max_seq_length
            mask = [1] * (len(ids_cxt) + len(ids_qst)) + [0] * pad_amount
            assert len(mask) == self.max_seq_length

            assert 0 < start < self.max_seq_length
            assert 0 < end < self.max_seq_length

            assert sum(np.array(ids) == 102) == 2
            assert sum(np.array(ids) == 101) == 1

            ids_list.append(ids)
            ids = torch.tensor(ids, dtype=torch.long)
            tt = torch.tensor(tt, dtype=torch.bool)
            mask = torch.tensor(mask, dtype=torch.bool)
            start = torch.tensor(start, dtype=torch.long)
            end = torch.tensor(end, dtype=torch.long)
            expand_dim0 = partial(torch.unsqueeze, dim=0)
            ids, tt, mask = tuple(map(expand_dim0, (ids, tt, mask)))

            if has_tt:
                x.append(
                    {"input_ids": ids, "attention_mask": mask, "token_type_ids": tt}
                )
            else:
                x.append({"input_ids": ids, "attention_mask": mask})
            y.append((start, end))
            texts.append(atext)

        return x, texts, ids_list


class ImagenetteDataset(Dataset):
    def __init__(
        self,
        name: str,
        input_names: Optional[List[str]] = None,
        num_examples: Optional[int] = None,
    ) -> None:
        """
        A sampled version of the Imagenet dataset containing only 10 classes.

        Original source from fast.ai here: https://github.com/fastai/imagenette
        """
        super().__init__(name)
        self.num_examples = num_examples
        self.raw_data = self._load_dataset()
        self.input_key = "x" if input_names is None else input_names[0]
        self.x, self.y = self.preprocess()

    def preprocess(self):
        data_loader = DataLoader(
            self.raw_data,
            batch_size=1,
            shuffle=False,
        )

        x = []
        y = []
        for i, item in enumerate(data_loader):
            if self.num_examples is not None and i >= self.num_examples:
                break
            x_, y_ = item
            x_ = {self.input_key: x_}
            x.append(x_)
            y.append(np.array(IMAGENETTE_CLASS_MAP[int(y_.item())]))

        return x, np.array(y, dtype=np.float32)

    def _validate_data(self, imagenette_path):
        devkit_file_name = DEVKIT_URL.split("/")[-1]
        devkit_path = os.path.join(imagenette_path, devkit_file_name)

        # Check devkit exists
        if not os.path.exists(devkit_path):
            return False

        # Check devkit permissions
        devkit_permissions = os.stat(devkit_path).st_mode
        if devkit_permissions & stat.S_IEXEC != stat.S_IEXEC:
            return False

        # Check val data exists
        val_data_path = os.path.join(imagenette_path, "val")
        if not os.path.exists(val_data_path):
            return False

        # Ensure 10 classes
        subdirs = os.listdir(val_data_path)
        if len(subdirs) != 10:
            return False

        # Ensure >= 300 samples per classes
        for subdir in subdirs:
            if len(os.listdir(os.path.join(val_data_path, subdir))) < 300:
                return False
        return True

    def _download_dataset(self):
        if os.path.exists(IMAGENETTE_PATH):
            if self._validate_data(IMAGENETTE_PATH):
                return IMAGENETTE_PATH
            else:
                # Data is corrupted, delete and re-download
                shutil.rmtree(IMAGENETTE_PATH)

        import wget  # pylint: disable=import-error

        tarfile_path = cached_path(IMAGENETTE_URL)
        os.makedirs(GROQFLOW_DATASETS_PATH, exist_ok=True)
        with tarfile.open(tarfile_path) as f:
            f.extractall(GROQFLOW_DATASETS_PATH)
        wget.download(DEVKIT_URL, out=IMAGENETTE_PATH)
        devkit_file_name = DEVKIT_URL.split("/")[-1]
        os.system(f"chmod +x {IMAGENETTE_PATH}/{devkit_file_name}")
        return IMAGENETTE_PATH

    def _load_dataset(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize]
        )
        imagenet_dir = self._download_dataset()
        images = ImageNet(root=imagenet_dir, split="val", transform=transform)

        return images


class SemanticSegmentationDataset(Dataset):
    def __init__(self, name: str, feature_extractor, num_examples=None) -> None:
        super().__init__(name)
        self.raw_data = load_dataset(name, split="validation")
        self.feature_extractor = feature_extractor
        self.num_examples = num_examples

        self.x, self.y = self.preprocess()

    def preprocess(self):
        feature_names = self._get_data_features()
        x, y = (self.raw_data[feature_names[0]], self.raw_data[feature_names[1]])

        inputs = [
            self.feature_extractor(x_, y_, return_tensors="pt")
            for (x_, y_) in zip(x, y)
        ]

        x = [
            {"pixel_values": self._resize_input(inp["pixel_values"])}
            for inp in inputs
        ]
        y = [inp["labels"] for inp in inputs]
        if self.num_examples:
            x = x[: self.num_examples]
            y = y[: self.num_examples]

        return x, y

    def _get_data_features(self) -> List[str]:
        feature_names = self.raw_data.features.keys()
        assert len(feature_names) >= 2
        feature_names = list(feature_names)[:2]
        return feature_names

    # TODO: generalize this.
    def _resize_input(self, image):
        return F.interpolate(image, size=[224, 224])


class SentenceSimilarityDataset(Dataset):
    def __init__(
        self, name: str, tokenizer, max_seq_length, num_examples=None
    ) -> None:
        super().__init__(name)
        self.raw_data = load_dataset(name, "en", split="test")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_examples = num_examples

        self.x, self.y = self.preprocess()

    def preprocess(self):
        dataset = self.raw_data.rename_columns({"similarity_score": "label"})
        dataset = dataset.map(lambda x: {"label": x["label"] / 5.0})

        x = []
        y = []

        for item in dataset:
            encoded_input = self.tokenizer(
                [item["sentence1"], item["sentence2"]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded_input = dict(**encoded_input)
            if encoded_input["input_ids"].shape[1] > self.max_seq_length:
                continue
            encoded_input = self._zero_pad(encoded_input)
            encoded_input["attention_mask"] = encoded_input["attention_mask"].bool()
            x.append(encoded_input)
            y.append(item["label"])
            if self.num_examples is not None and len(x) == self.num_examples:
                break

        return x, y

    def _zero_pad(self, encoded_input):
        for k, v in encoded_input.items():
            encoded_input[k] = F.pad(
                v, (0, self.max_seq_length - v.shape[1]), "constant", 0
            )
        return encoded_input


class SpeechCommandsDataset(Dataset):
    def __init__(self, name: str):
        super().__init__(name)

        from torchaudio.datasets import SPEECHCOMMANDS  # pylint: disable=import-error

        # load dataset
        self.raw_data = SPEECHCOMMANDS(
            root=(os.path.join(os.path.dirname(__file__), "datasets")),
            download=True,
            subset="testing",
        )

        # generate input and labels
        self.x, self.y = self.preprocess()

    def preprocess(self):
        # create word to index lookup list
        labels_list = sorted(list(set(data[2] for data in self.raw_data)))

        # generate max length to pad all shorter input data
        max_len = max([data[0].shape[-1] for data in self.raw_data])

        x = [
            {
                "x": torch.unsqueeze(
                    torch.cat(
                        (data[0], torch.zeros((1, max_len - data[0].shape[-1]))),
                        dim=-1,
                    ),
                    dim=0,
                )
            }
            for data in self.raw_data
        ]
        y = [labels_list.index(data[2]) for data in self.raw_data]

        return x, y


class PointSampler:
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = self.triangle_area(
                verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]]
            )

        sampled_faces = random.choices(
            faces, weights=areas, cum_weights=None, k=self.output_size
        )

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(
                verts[sampled_faces[i][0]],
                verts[sampled_faces[i][1]],
                verts[sampled_faces[i][2]],
            )

        return sampled_points


class Normalize:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud


class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)


class RandRotationZ:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        theta = random.random() * 2.0 * math.pi
        rot_matrix = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))

        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class PointCloudDataset(Dataset):
    """https://github.com/nikitakaraevv/pointnet"""

    def __init__(self, name: str, num_examples: Optional[int] = None) -> None:
        super().__init__(name)
        self.num_examples = num_examples
        root_dir = self._download_dataset()
        folders = [
            dir
            for dir in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, dir))
        ]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = T.Compose(
            [
                PointSampler(1024),
                Normalize(),
                RandRotationZ(),
                RandomNoise(),
                ToTensor(),
            ]
        )

        self.files = []
        for category in self.classes.keys():
            new_dir = os.path.join(root_dir, category, "test")
            for file in os.listdir(new_dir):
                if file.endswith(".off"):
                    sample = {}
                    sample["pcd_path"] = os.path.join(new_dir, file)
                    sample["category"] = category
                    self.files.append(sample)

        self.x, self.y = self.preprocess()

    def preprocess(self):
        x = []
        y = []
        for sample in self.files:
            pcd_path = sample["pcd_path"]
            with open(pcd_path, "r", encoding="utf-8") as file:
                verts, faces = self._read_off(file)
                pointcloud = self.transforms((verts, faces))
                # ndarray needs to be C-contiguous
                pointcloud = np.asarray(
                    np.transpose(np.expand_dims(pointcloud, axis=0), (0, 2, 1)),
                    order="c",
                )

            x.append({"input": torch.tensor(pointcloud, dtype=torch.float)})
            y.append(self.classes[sample["category"]])
            if self.num_examples is not None and len(x) == self.num_examples:
                break

        return x, np.array(y, dtype=np.float32)

    def _read_off(self, file):
        if "OFF" != file.readline().strip():
            raise Exception("Not a valid OFF header")
        n_verts, n_faces, __ = tuple(
            [int(s) for s in file.readline().strip().split(" ")]
        )
        verts = [
            [float(s) for s in file.readline().strip().split(" ")]
            for i_vert in range(n_verts)
        ]
        faces = [
            [int(s) for s in file.readline().strip().split(" ")][1:]
            for i_face in range(n_faces)
        ]
        return verts, faces

    def _download_dataset(self):
        if os.path.exists(MODELNET10_PATH):
            if self._validate_data(MODELNET10_PATH):
                return MODELNET10_PATH
            else:
                # Data is corrupted, delete and re-download
                shutil.rmtree(MODELNET10_PATH)

        zipfile_path = cached_path(MODELNET10_URL)

        with zipfile.ZipFile(zipfile_path, "r") as f:
            f.extractall(GROQFLOW_DATASETS_PATH)
        return MODELNET10_PATH

    def _validate_data(self, modelnet10_path):
        root_dir = modelnet10_path

        # Ensure there are 10 class folders
        folders = [
            dir
            for dir in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, dir))
        ]
        if len(folders) != 10:
            return False

        # Ensure every class has a test folder
        for folder in folders:
            test_dir = os.path.join(root_dir, folder, "test")
            if not os.path.exists(test_dir):
                return False

        # Ensure >= 50 test samples per classes
        for folder in folders:
            test_dir = os.path.join(root_dir, folder, "test")
            if len(os.listdir(test_dir)) < 50:
                return False

        return True


class CocoTransform:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        import cv2

        cv_img = np.array(img)

        h0, w0 = cv_img.shape[:2]
        r = self.target_size / max(h0, w0)
        if r != 1:
            cv_img = cv2.resize(
                cv_img,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR,
            )
            h0, w0 = cv_img.shape[:2]

        top = (self.target_size - h0) // 2
        bottom = self.target_size - h0 - top
        left = (self.target_size - w0) // 2
        right = self.target_size - w0 - left

        cv_img = cv2.copyMakeBorder(
            cv_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        cv_img = cv_img.transpose((2, 0, 1)) / 255
        return torch.tensor(cv_img).float()

    def __repr__(self):
        return f"CocoTransform({self.target_size})"


class COCODataset(Dataset):
    def __init__(self, target_size=640):
        super().__init__("coco")
        self.dataset_path = os.path.join(GROQFLOW_DATASETS_PATH, "coco")
        self.anno_path = os.path.join(
            self.dataset_path, "annotations/instances_val2017.json"
        )
        self.imgs_path = os.path.join(self.dataset_path, "val2017")
        self._download_dataset()

        from torchvision.datasets.coco import CocoDetection

        transform = CocoTransform(target_size)
        with suppress_stdout():
            self.coco = CocoDetection(
                self.imgs_path, self.anno_path, transform=transform
            )
        self.x = self.preprocess()
        self.y = None
        self.target_size = target_size

        skip_classes = set([0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83])
        self.class_conversion_map = [i for i in range(91) if i not in skip_classes]

    def _validate_dataset(self):
        if not os.path.exists(self.anno_path):
            return False

        if not os.path.exists(self.imgs_path):
            return False

        if len(os.listdir(self.imgs_path)) != 5000:
            return False
        return True

    def _download_dataset(self):
        if self._validate_dataset():
            return
        os.makedirs(self.dataset_path, exist_ok=True)

        base_url = "http://images.cocodataset.org"
        anno_url = f"{base_url}/annotations/annotations_trainval2017.zip"
        imgs_url = f"{base_url}/zips/val2017.zip"

        anno_zip = cached_path(anno_url)
        imgs_zip = cached_path(imgs_url)

        with zipfile.ZipFile(anno_zip, "r") as f:
            f.extractall(self.dataset_path)

        with zipfile.ZipFile(imgs_zip, "r") as f:
            f.extractall(self.dataset_path)

    def preprocess(self):
        return [
            {"image_arrays": self.coco[i][0].unsqueeze(0).numpy()}
            for i in range(len(self.coco))
        ]

    def non_max_suppression(self, prediction):
        if len(prediction) == 0 or len(prediction[0]) == 0:
            return np.zeros((0, 6))

        conf_thres = 0.03
        iou_thres = 0.65
        max_det = 300

        x = prediction[0]
        x[:, 5:] *= x[:, 4:5]
        conf, class_idx = x[:, 5:].max(axis=1, keepdim=True)
        mask = (conf > conf_thres).view(-1)
        x = x[mask]
        class_idx = class_idx[mask].float()
        conf = conf[mask]
        if len(x) == 0:
            return np.zeros((0, 6))
        boxes = box_convert(x[:, :4], "cxcywh", "xyxy")
        x = torch.cat((boxes, conf, class_idx), 1)

        # Add offsets nms is done separately for each class
        offset_boxes = boxes + class_idx * 4096
        keep_idx = nms(offset_boxes, conf.view(-1).float(), iou_thres)
        keep_idx = keep_idx[:max_det]
        return x[keep_idx]

    def _postprocess_single(self, output, image_id, img_meta):
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)
        output = self.non_max_suppression(output)
        if len(output) == 0:
            return np.zeros((0, 7))
        src_h, src_w = img_meta["height"], img_meta["width"]
        scale = self.target_size / (max(src_h, src_w))
        w_pad = (self.target_size - round(src_w * scale)) // 2
        h_pad = (self.target_size - round(src_h * scale)) // 2

        output[:, 0].sub_(w_pad)
        output[:, 1].sub_(h_pad)
        output[:, 2].sub_(w_pad)
        output[:, 3].sub_(h_pad)

        output[:, :4].div_(scale)

        output[:, 0].clamp_(0, src_w)
        output[:, 1].clamp_(0, src_h)
        output[:, 2].clamp_(0, src_w)
        output[:, 3].clamp_(0, src_h)

        boxes = box_convert(output[:, :4], "xyxy", "xywh")
        n = output.shape[0]

        out = np.zeros((n, 7))
        out[:, 0] = image_id
        out[:, 1:5] = boxes
        out[:, 5] = output[:, 4]
        out[:, 6] = [self.class_conversion_map[int(output[i][5])] for i in range(n)]
        return out

    def postprocess(self, outputs):
        results = []
        for i, output in enumerate(outputs):
            img_id = self.coco.ids[i]
            results.append(
                self._postprocess_single(output, img_id, self.coco.coco.imgs[img_id])
            )
        return results


def create_dataset(
    name: str,
    tokenizer=None,
    max_seq_length=None,
    feature_extractor=None,
    input_names=None,
) -> Dataset:
    if name == "sst":
        return LanguageDataset(name, tokenizer, max_seq_length)
    elif name == "squad":
        return QuestionAnsweringDataset(name, tokenizer, max_seq_length)
    elif name == "sampled_imagenet":
        return ImagenetteDataset(name, input_names=input_names)
    elif name == "conll2003":
        return CoNLL2003Dataset(
            name, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
    elif name == "scene_parse_150":
        return SemanticSegmentationDataset(
            name, feature_extractor=feature_extractor, num_examples=100
        )
    elif name == "stsb_multi_mt":
        return SentenceSimilarityDataset(
            name, tokenizer=tokenizer, max_seq_length=max_seq_length
        )
    elif name == "speechcommands":
        return SpeechCommandsDataset(name)
    elif name == "point_cloud":
        return PointCloudDataset(name)
    elif name == "coco":
        return COCODataset()
    else:
        raise ValueError("Unknown dataset: " + name)
