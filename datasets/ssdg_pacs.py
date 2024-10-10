import os.path as osp
import random
from collections import defaultdict
import numpy as np

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, read_json, write_json

from .utils import random_numbers, exp_imbalance_l, exp_imbalance_u, count_classes


@DATASET_REGISTRY.register(force=True)
class SSDGPACS(DatasetBase):
    """PACS.

    Statistics:
        - 4 domains: Photo (1,670), Art (2,048), Cartoon
        (2,344), Sketch (3,929).
        - 7 categories: dog, elephant, giraffe, guitar, horse,
        house and person.

    Reference:
        - Li et al. Deeper, broader and artier domain generalization.
        ICCV 2017.
        - Zhou et al. Semi-Supervised Domain Generalization with
        Stochastic StyleMatch. ArXiv preprint, 2021.
    """

    dataset_dir = "pacs"
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")
        self.split_ssdg_dir = osp.join(self.dataset_dir, "splits_ssdg")
        mkdir_if_missing(self.split_ssdg_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "pacs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS)

        seed = cfg.SEED
        num_labeled = cfg.DATASET.NUM_LABELED
        src_domains = cfg.DATASET.SOURCE_DOMAINS
        tgt_domain = cfg.DATASET.TARGET_DOMAINS[0]

        print(cfg.DATASET.SOURCE_DOMAINS)
        print(cfg.DATASET.TARGET_DOMAINS)
        print(cfg.DATASET.ONE_SOURCE_L)

        split_ssdg_path = osp.join(
            self.split_ssdg_dir, f"{tgt_domain}_nlab{num_labeled}_{cfg.TRAINER.FBASA.IMBALANCE}_seed{seed}.json"
        )
        if not osp.exists(split_ssdg_path):
            train_x, train_u = self._read_data_train(
                    cfg.DATASET.SOURCE_DOMAINS,
                    num_labeled,
                    cfg.TRAINER.FBASA.IMBALANCE,
                    gamma=cfg.TRAINER.FBASA.GAMMA,
                    one_source_l=cfg.DATASET.ONE_SOURCE_L
                )
        else:
            train_x, train_u = self.read_json_train(
                split_ssdg_path, src_domains, self.image_dir
            )
        
        val = self._read_data_test(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, "all")

        if cfg.DATASET.ALL_AS_UNLABELED:
            train_u = train_u + train_x

        # Print class distribution
        print("Class distribution in the labeled training set:")
        print(count_classes(train_x))
        print("Class distribution in the unlabeled training set:")
        print(count_classes(train_u))

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    @staticmethod
    def read_json_train(filepath, src_domains, image_dir):
        """
        The latest office_home_dg dataset's class folders have
        been changed to only contain the class names, e.g.,
        000_Alarm_Clock/ is changed to Alarm_Clock/.
        """

        def _convert_to_datums(items):
            out = []
            for impath, label, dname in items:
                if dname not in src_domains:
                    continue
                domain = src_domains.index(dname)
                impath2 = osp.join(image_dir, impath)
                if not osp.exists(impath2):
                    impath = impath.split("/")
                    if impath[-2].startswith("0"):
                        impath[-2] = impath[-2][4:]
                    impath = "/".join(impath)
                    impath2 = osp.join(image_dir, impath)
                item = Datum(impath=impath2, label=int(label), domain=domain)
                out.append(item)
            return out

        print(f'Reading split from "{filepath}"')
        split = read_json(filepath)
        train_x = _convert_to_datums(split["train_x"])
        train_u = _convert_to_datums(split["train_u"])

        return train_x, train_u

    # @staticmethod
    # def write_json_train(filepath, src_domains, image_dir, train_x, train_u):
    #     def _convert_to_list(items):
    #         out = []
    #         for item in items:
    #             impath = item.impath
    #             label = item.label
    #             domain = item.domain
    #             dname = src_domains[domain]
    #             impath = impath.replace(image_dir, "")
    #             if impath.startswith("/") or impath.startswith("\\"):
    #                 impath = impath[1:]
    #             out.append((impath, label, dname))
    #         return out

    #     train_x = _convert_to_list(train_x)
    #     train_u = _convert_to_list(train_u)
    #     output = {"train_x": train_x, "train_u": train_u}

    #     write_json(output, filepath)
    #     print(f'Saved the split to "{filepath}"')

    def _read_data_train(self, input_domains, num_labeled, imbalance, gamma=None, one_source_l=None):
        num_domains = len(input_domains)
        items_x, items_u = [], []

        num_labeled_per_cd = None
        num_unlabeled_per_cd = None
        num_labeled_per_class = None

        # Labelled samples come from all source domains
        if one_source_l is None:
            # Get number of labels
            file = osp.join(self.split_dir, input_domains[0] + "_train_kfold.txt")
            impath_label_list = self._read_split_pacs(file)

            impath_label_dict = defaultdict(list)

            for impath, label in impath_label_list:
                impath_label_dict[label].append((impath, label))

            labels = list(impath_label_dict.keys())

            # Original implementation
            if imbalance == "original":
                num_labeled_per_cd = np.ones((num_domains, len(labels))) * num_labeled // (num_domains * len(labels))

            # Randomly assign number of labeled samples per class and domain
            elif imbalance == "random":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = []
                for d in range(num_domains):
                    num_labeled_per_cd.append(random_numbers(num_labeled_per_domain, len(labels)))

            # Exponential (long-tail) imbalance on both labeled and unlabeled samples
            elif imbalance == "exp":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = exp_imbalance_l(num_labeled_per_domain, len(labels), gamma)

                random.shuffle(labels) # randomize the majority class
                m1 = len(impath_label_dict[labels[0]]) - num_labeled_per_cd[labels[0]]*num_domains
                num_unlabeled_per_cd = exp_imbalance_u(m1, len(labels), gamma)

                num_labeled_per_cd = [[num_labeled_per_cd[label] for label in labels] for _ in range(num_domains)]
                num_unlabeled_per_cd = [[num_unlabeled_per_cd[label] for label in labels] for _ in range(num_domains)]

            # Exponential (long-tail) imbalance on labeled samples only
            elif imbalance == "exp_l_only":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = exp_imbalance_l(num_labeled_per_domain, len(labels), gamma)
                random.shuffle(labels) # randomize the majority class
                num_labeled_per_cd = [[num_labeled_per_cd[label] for label in labels] for _ in range(num_domains)]

            # Uniform distribution with the same number of unlabeled samples as in the exponential imbalance to compare both settings
            elif imbalance == "uniform_exp_like":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = exp_imbalance_l(num_labeled_per_domain, len(labels), gamma)

                random.shuffle(labels) # randomize the majority class
                m1 = len(impath_label_dict[labels[0]]) - num_labeled_per_cd[labels[0]]*num_domains
                num_unlabeled_per_cd = exp_imbalance_u(m1, len(labels), gamma)

                num_labeled_per_cd = np.ones((num_domains, len(labels))) * num_labeled // (num_domains * len(labels))
                num_unlabeled_per_cd = np.ones((num_domains, len(labels))) * np.sum(num_unlabeled_per_cd) // (num_domains * len(labels))

            else:
                raise ValueError(f"Unknown imbalance type for all sources labelled: {imbalance}")
            
            for domain, dname in enumerate(input_domains):
                file = osp.join(self.split_dir, dname + "_train_kfold.txt")
                impath_label_list = self._read_split_pacs(file)

                impath_label_dict = defaultdict(list)

                for impath, label in impath_label_list:
                    impath_label_dict[label].append((impath, label))

                labels = list(impath_label_dict.keys())            

                for label in labels:
                    pairs = impath_label_dict[label]
                    assert len(pairs) >= num_labeled_per_cd[domain][label], "Not enough labeled data for class {} in domain {}".format(label, dname)
                    random.shuffle(pairs)

                    for i, (impath, label) in enumerate(pairs):
                        item = Datum(impath=impath, label=label, domain=domain)
                        if (i + 1) <= num_labeled_per_cd[domain][label]:
                            items_x.append(item)
                        elif num_unlabeled_per_cd is not None:
                            if (i + 1) <= num_labeled_per_cd[domain][label] + num_unlabeled_per_cd[domain][label]:
                                items_u.append(item)
                        else:
                            items_u.append(item)

            return items_x, items_u
            
        else:
            assert one_source_l in input_domains, "Labelled source domain not in the input domains"

            # Get number of labels
            file = osp.join(self.split_dir, input_domains[0] + "_train_kfold.txt")
            impath_label_list = self._read_split_pacs(file)

            impath_label_dict = defaultdict(list)

            for impath, label in impath_label_list:
                impath_label_dict[label].append((impath, label))

            labels = list(impath_label_dict.keys())

            # Original implementation
            if imbalance == "original":
                num_labeled_per_class = np.ones(len(labels)) * num_labeled // len(labels)

            # Randomly assign number of labeled samples per class and domain
            elif imbalance == "random":
                num_labeled_per_class = random_numbers(num_labeled, len(labels))

            # Exponential (long-tail) imbalance on labeled samples only
            elif imbalance == "exp_l_only":
                num_labeled_per_class = exp_imbalance_l(num_labeled, len(labels), gamma)
                random.shuffle(labels) # randomize the majority class
                num_labeled_per_class = [num_labeled_per_class[label] for label in labels]

            else:
                raise ValueError(f"Unknown imbalance type for one_source_l: {imbalance}")
            
            for domain, dname in enumerate(input_domains):
                file = osp.join(self.split_dir, dname + "_train_kfold.txt")
                impath_label_list = self._read_split_pacs(file)

                impath_label_dict = defaultdict(list)

                for impath, label in impath_label_list:
                    impath_label_dict[label].append((impath, label))

                labels = list(impath_label_dict.keys())            

                for label in labels:
                    pairs = impath_label_dict[label]
                    if dname == one_source_l:
                        assert len(pairs) >= num_labeled_per_class[label], "Not enough labeled data for class {} in domain {}".format(label, dname)
                    random.shuffle(pairs)

                    for i, (impath, label) in enumerate(pairs):
                        item = Datum(impath=impath, label=label, domain=domain)
                        if dname == one_source_l and (i + 1) <= num_labeled_per_class[label]:
                            items_x.append(item)
                        else:
                            items_u.append(item)

            return items_x, items_u
        

    def _read_data_test(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                file_train = osp.join(self.split_dir, dname + "_train_kfold.txt")
                impath_label_list = self._read_split_pacs(file_train)
                file_val = osp.join(self.split_dir, dname + "_crossval_kfold.txt")
                impath_label_list += self._read_split_pacs(file_val)
            else:
                file = osp.join(self.split_dir, dname + "_" + split + "_kfold.txt")
                impath_label_list = self._read_split_pacs(file)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items

    def _read_split_pacs(self, split_file):
        items = []

        with open(split_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                if impath in self._error_paths:
                    continue
                impath = osp.join(self.image_dir, impath)
                label = int(label) - 1
                items.append((impath, label))

        return items