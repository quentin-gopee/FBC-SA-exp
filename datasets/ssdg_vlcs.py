import os.path as osp
import glob
import random
from collections import defaultdict
import numpy as np

from dassl.utils import listdir_nohidden
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .utils import random_numbers, exp_imbalance_l, count_classes



@DATASET_REGISTRY.register(force=True)
class SSDGVLCS(DatasetBase):
    """VLCS.

    Statistics:
        - 4 domains: CALTECH, LABELME, PASCAL, SUN
        - 5 categories: bird, car, chair, dog, and person.

    Reference:
        - Torralba and Efros. Unbiased look at dataset bias. CVPR 2011.
    """

    dataset_dir = "VLCS"
    domains = ["caltech", "labelme", "pascal", "sun"]
    data_url = "https://drive.google.com/uc?id=1r0WL5DDqKfSPp9E3tRENwHaXNs1olLZd"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_ssdg_dir = osp.join(self.dataset_dir, "splits_ssdg")
        mkdir_if_missing(self.split_ssdg_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "vlcs.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        seed = cfg.SEED
        num_labeled = cfg.DATASET.NUM_LABELED
        src_domains = cfg.DATASET.SOURCE_DOMAINS
        tgt_domain = cfg.DATASET.TARGET_DOMAINS[0]

        split_ssdg_path = osp.join(
            self.split_ssdg_dir, f"{tgt_domain}_nlab{num_labeled}_{cfg.TRAINER.FBASA.IMBALANCE}_seed{seed}.json"
        )
        if not osp.exists(split_ssdg_path):
            train_x, train_u = self._read_data_train(
                    cfg.DATASET.SOURCE_DOMAINS,
                    "full",
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
        test = self._read_data_test(cfg.DATASET.TARGET_DOMAINS, "full")

        # Print class distribution
        print("Class distribution in the labeled training set:")
        print(count_classes(train_x))
        print("Class distribution in the unlabeled training set:")
        print(count_classes(train_u))

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)

    # def _read_data(self, input_domains, split):
    #     items = []

    #     for domain, dname in enumerate(input_domains):
    #         dname = dname.upper()
    #         path = osp.join(self.dataset_dir, dname, split)
    #         folders = listdir_nohidden(path)
    #         folders.sort()

    #         for label, folder in enumerate(folders):
    #             impaths = glob.glob(osp.join(path, folder, "*.jpg"))

    #             for impath in impaths:
    #                 item = Datum(impath=impath, label=label, domain=domain)
    #                 items.append(item)

    #     return items


    def _read_data_train(self, input_domains, split, num_labeled, imbalance, gamma=None, one_source_l=None):
        num_domains = len(input_domains)
        items_x, items_u = [], []

        # Labelled samples come from all source domains
        if one_source_l is None:
            # Get number of labels
            path = osp.join(self.dataset_dir, input_domains[0].upper(), split)
            folders = listdir_nohidden(path, sort=True)
            labels = np.arange(len(folders))

            # Get the number of samples in each category/domain
            min_distribution = [
                min(
                    [
                        len(glob.glob(osp.join(self.dataset_dir, domain.upper(), split, folder, "*.jpg")))
                    for domain in input_domains]
                )
            for folder in folders]

            # Original implementation
            if imbalance == "original":
                num_labeled_per_cd = np.ones((num_domains, len(labels))) * num_labeled // (num_domains * len(labels))

            # Randomly assign number of labeled samples per class and domain
            elif imbalance == "random":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = []
                for d in range(num_domains):
                    num_labeled_per_cd.append(random_numbers(num_labeled_per_domain, len(labels)))

            # Exponential (long-tail) imbalance on labeled samples only
            elif imbalance == "exp_l_only":
                num_labeled_per_domain = num_labeled // num_domains
                num_labeled_per_cd = exp_imbalance_l(num_labeled_per_domain, len(labels), gamma)
                assert (num_labeled_per_cd[::-1] <= np.sort(min_distribution)).all(), 'No configuration possible with the current number of samples'
                random.shuffle(labels) # randomize the majority class
                num_labeled_per_cd_tmp = [num_labeled_per_cd[label] for label in labels]
                sorted_indexes = np.argsort(num_labeled_per_cd_tmp)
                # rearrange the values to avoid overflow
                indexes_pb = [i for i, e in enumerate(num_labeled_per_cd_tmp) if e > min_distribution[i]]
                for i in indexes_pb:
                    # invert the value with the smallest value that does not cause overflow
                    for j in sorted_indexes:
                        if num_labeled_per_cd_tmp[i] <= min_distribution[j]:
                            old_value = num_labeled_per_cd_tmp[i]
                            num_labeled_per_cd_tmp[i] = num_labeled_per_cd_tmp[j]
                            num_labeled_per_cd_tmp[j] = old_value
                            break
                    sorted_indexes = np.argsort(num_labeled_per_cd_tmp)
                num_labeled_per_cd = [num_labeled_per_cd_tmp for _ in range(num_domains)]
            
            for domain, dname in enumerate(input_domains):
                dname = dname.upper()
                path = osp.join(self.dataset_dir, dname, split)
                folders = listdir_nohidden(path, sort=True)

                for label, folder in enumerate(folders):
                    impaths = glob.glob(osp.join(path, folder, "*.jpg"))
                    assert len(impaths) >= num_labeled_per_cd[domain][label], "Not enough labeled data for class {} in domain {}".format(folder, dname)
                    random.shuffle(impaths)

                    for i, impath in enumerate(impaths):
                        item = Datum(impath=impath, label=label, domain=domain)
                        if (i + 1) <= num_labeled_per_cd[domain][label]:
                            items_x.append(item)
                        else:
                            items_u.append(item)           

            return items_x, items_u
            
        else:
            assert one_source_l in input_domains, "Labelled source domain not in the input domains"

            # Get number of labels
            path = osp.join(self.dataset_dir, input_domains[0].upper(), split)
            folders = listdir_nohidden(path, sort=True)
            labels = np.arange(len(folders))

            # Get the number of samples in each category/domain
            distribution = [
                len(glob.glob(osp.join(self.dataset_dir, one_source_l.upper(), split, folder, "*.jpg")))
                for folder in folders
            ]

            # Original implementation
            if imbalance == "original":
                num_labeled_per_class = np.ones(len(labels)) * num_labeled // len(labels)

            # Randomly assign number of labeled samples per class and domain
            elif imbalance == "random":
                num_labeled_per_class = random_numbers(num_labeled, len(labels))

            # Exponential (long-tail) imbalance on labeled samples only
            elif imbalance == "exp_l_only":
                num_labeled_per_class = exp_imbalance_l(num_labeled, len(labels), gamma)
                assert (num_labeled_per_class[::-1] <= np.sort(distribution)).all(), 'No configuration possible with the current number of samples'
                random.shuffle(labels) # randomize the majority class
                num_labeled_per_class = [num_labeled_per_class[label] for label in labels]
                sorted_indexes = np.argsort(num_labeled_per_class)
                # rearrange the values to avoid overflow
                indexes_pb = [i for i, e in enumerate(num_labeled_per_class) if e > distribution[i]]
                for i in indexes_pb:
                    # invert the value with the smallest value that does not cause overflow
                    for j in sorted_indexes:
                        if num_labeled_per_class[i] <= distribution[j]:
                            old_value = num_labeled_per_class[i]
                            num_labeled_per_class[i] = num_labeled_per_class[j]
                            num_labeled_per_class[j] = old_value
                            break
                    sorted_indexes = np.argsort(num_labeled_per_class)

            else:
                raise ValueError(f"Unknown imbalance type for one_source_l: {imbalance}")
            
            for domain, dname in enumerate(input_domains):
                dname = dname.upper()
                path = osp.join(self.dataset_dir, dname, split)
                folders = listdir_nohidden(path, sort=True)

                for label, folder in enumerate(folders):
                    impaths = glob.glob(osp.join(path, folder, "*.jpg"))
                    if dname == one_source_l:
                        assert len(impaths) >= num_labeled_per_class[label], "Not enough labeled data for class {} in domain {}".format(folder, dname)
                    random.shuffle(impaths)

                    for i, impath in enumerate(impaths):
                        item = Datum(impath=impath, label=label, domain=domain)
                        if dname == one_source_l.upper() and (i + 1) <= num_labeled_per_class[label]:
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

    def _read_data_test(self, input_domains, split):
        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory, sort=True)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []

        for domain, dname in enumerate(input_domains):
            dname = dname.upper()
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                item = Datum(impath=impath, label=label, domain=domain)
                items.append(item)

        return items