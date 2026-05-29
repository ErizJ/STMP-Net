import csv
import json
import os
import warnings

import math
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image
try:
    from scipy import io
except Exception:
    io = None

# distortion_map = { "01": "Gaussian blur", "02": "Lens blur", "03": "Motion blur", "04": "Color diffusion", "05": "Color shift",
# "06": "Color quantization", "07": "Color saturation 1", "08": "Color saturation 2", "09": "JPEG2000 compression",
# "10": "JPEG compression", "11": "White noise", "12": "White noise in color component", "13": "Impulse noise", "14": "Multiplicative noise",
# "15": "Denoise", "16": "Brighten", "17": "Darken", "18": "Mean shift", "19": "Jitter",
# "20": "Non-eccentricity patch", "21": "Pixelate", "22": "Quantization", "23": "Color block", "24": "High sharpen", "25": "Contrast change"
# }

distortion_map = {
    1: "Gaussian blur 1",
    2: "Gaussian blur 2",
    3: "Gaussian blur 3",
    4: "Gaussian blur 4",
    5: "Gaussian blur 5",
    6: "Lens blur 1",
    7: "Lens blur 2",
    8: "Lens blur 3",
    9: "Lens blur 4",
    10: "Lens blur 5",
    11: "Motion blur 1",
    12: "Motion blur 2",
    13: "Motion blur 3",
    14: "Motion blur 4",
    15: "Motion blur 5",
    16: "Color diffusion 1",
    17: "Color diffusion 2",
    18: "Color diffusion 3",
    19: "Color diffusion 4",
    20: "Color diffusion 5",
    21: "Color shift 1",
    22: "Color shift 2",
    23: "Color shift 3",
    24: "Color shift 4",
    25: "Color shift 5",
    26: "Color quantization 1",
    27: "Color quantization 2",
    28: "Color quantization 3",
    29: "Color quantization 4",
    30: "Color quantization 5",
    31: "Color saturation 1-1",
    32: "Color saturation 1-2",
    33: "Color saturation 1-3",
    34: "Color saturation 1-4",
    35: "Color saturation 1-5",
    36: "Color saturation 2-1",
    37: "Color saturation 2-2",
    38: "Color saturation 2-3",
    39: "Color saturation 2-4",
    40: "Color saturation 2-5",
    41: "JPEG2000 compression 1",
    42: "JPEG2000 compression 2",
    43: "JPEG2000 compression 3",
    44: "JPEG2000 compression 4",
    45: "JPEG2000 compression 5",
    46: "JPEG compression 1",
    47: "JPEG compression 2",
    48: "JPEG compression 3",
    49: "JPEG compression 4",
    50: "JPEG compression 5",
    51: "White noise 1",
    52: "White noise 2",
    53: "White noise 3",
    54: "White noise 4",
    55: "White noise 5",
    56: "White noise in color component 1",
    57: "White noise in color component 2",
    58: "White noise in color component 3",
    59: "White noise in color component 4",
    60: "White noise in color component 5",
    61: "Impulse noise 1",
    62: "Impulse noise 2",
    63: "Impulse noise 3",
    64: "Impulse noise 4",
    65: "Impulse noise 5",
    66: "Multiplicative noise 1",
    67: "Multiplicative noise 2",
    68: "Multiplicative noise 3",
    69: "Multiplicative noise 4",
    70: "Multiplicative noise 5",
    71: "Denoise 1",
    72: "Denoise 2",
    73: "Denoise 3",
    74: "Denoise 4",
    75: "Denoise 5",
    76: "Brighten 1",
    77: "Brighten 2",
    78: "Brighten 3",
    79: "Brighten 4",
    80: "Brighten 5",
    81: "Darken 1",
    82: "Darken 2",
    83: "Darken 3",
    84: "Darken 4",
    85: "Darken 5",
    86: "Mean shift 1",
    87: "Mean shift 2",
    88: "Mean shift 3",
    89: "Mean shift 4",
    90: "Mean shift 5",
    91: "Jitter 1",
    92: "Jitter 2",
    93: "Jitter 3",
    94: "Jitter 4",
    95: "Jitter 5",
    96: "Non-eccentricity patch 1",
    97: "Non-eccentricity patch 2",
    98: "Non-eccentricity patch 3",
    99: "Non-eccentricity patch 4",
    100: "Non-eccentricity patch 5",
    101: "Pixelate 1",
    102: "Pixelate 2",
    103: "Pixelate 3",
    104: "Pixelate 4",
    105: "Pixelate 5",
    106: "Quantization 1",
    107: "Quantization 2",
    108: "Quantization 3",
    109: "Quantization 4",
    110: "Quantization 5",
    111: "Color block 1",
    112: "Color block 2",
    113: "Color block 3",
    114: "Color block 4",
    115: "Color block 5",
    116: "High sharpen 1",
    117: "High sharpen 2",
    118: "High sharpen 3",
    119: "High sharpen 4",
    120: "High sharpen 5",
    121: "Contrast change 1",
    122: "Contrast change 2",
    123: "Contrast change 3",
    124: "Contrast change 4",
    125: "Contrast change 5"}

number_map = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
              6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
              12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen',
              16: 'sixteen', 17: 'seventeen', 18: 'eighteen', 19: 'nineteen',
              20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
              60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety'}

dist_map = {'jpeg2000 compression':'jpeg2000 compression', 'jpeg compression':'jpeg compression',
                   'white noise':'noise', 'gaussian blur':'blur', 'fastfading': 'jpeg2000 compression', 'fnoise':'noise',
                   'contrast':'contrast', 'lens':'blur', 'motion':'blur', 'diffusion':'color', 'shifting':'blur',
                   'color quantization':'quantization', 'oversaturation':'color', 'desaturation':'color',
                   'white with color':'noise', 'impulse':'noise', 'multiplicative':'noise',
                   'white noise with denoise':'noise', 'brighten':'overexposure', 'darken':'underexposure', 'shifting the mean':'other',
                   'jitter':'spatial', 'noneccentricity patch':'spatial', 'pixelate':'spatial', 'quantization':'quantization',
                   'color blocking':'spatial', 'sharpness':'contrast', 'realistic blur':'blur', 'realistic noise':'noise',
                   'underexposure':'underexposure', 'overexposure':'overexposure', 'realistic contrast change':'contrast', 'other realistic':'other'}

map2label = {'jpeg2000 compression':0, 'jpeg compression':1, 'noise':2, 'blur':3, 'color':4,
             'contrast':5, 'overexposure':6, 'underexposure':7, 'spatial':8, 'quantization':9, 'other':10}

scene2label = {'animal':0, 'cityscape':1, 'human':2, 'indoor':3, 'landscape':4, 'night':5, 'plant':6, 'still_life':7,
               'others':8}


def get_number(i):
    if i in number_map:
        return number_map[i]
    else:
        tens = (i // 10) * 10
        ones = i % 10
        return number_map[tens]+"-"+number_map[ones]


def transfer(database, label):
    if database == 'live':
        label = 100 - label
    elif database == "csiq":
        label = 100 - label * 100.0
    elif database == "tid2013":
        label = label / 9 * 100.0
    elif database == "kadid":
        label = (label - 1) * 25.0
    elif database == "bid":
        label = label * 20.0
    elif database == "cviu17":
        label = label * 10.0  # 0-10 → 0-100
    elif database == "qads":
        label = label * 100.0  # 根据实际MOS范围调整
    elif database == "waterloo15":
        label = label * 10.0  # 根据实际MOS范围调整
    elif database == "sisar":
        label = label * 100.0  # 根据实际MOS范围调整
    return label


class KONIQDATASET_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        super(KONIQDATASET_clip, self).__init__()
        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "koniq10k_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                # fields[0] = fields[0].split('/')[0] + '/' + fields[0].split('/')[-1]
                image_data[fields[0]] = (fields[2], fields[3])

        self.data_path = root
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "koniq10k_scores_and_distributions.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["image_name"])
                mos = np.array(float(row["MOS_zscore"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for _, item in enumerate(index):
            for _ in range(patch_num):
                # sample.append(
                #     (os.path.join(root, "1024x768", imgname[item]),
                #      mos_all[item], 1, 1)
                # )
                try:
                    image_path = os.path.join(root, "1024x768", imgname[item])
                    find_path = image_path.split(root + '/')[-1]
                    # print(f"findpath:{find_path}")
                    dist, scene = image_data.get(find_path)
                    sample.append(
                        (
                            image_path,
                            mos_all[item],
                            scene2label[scene], 
                            map2label[dist_map[dist]]
                        )
                    )
                except:
                    print("key error")

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, scene, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, distortion

    def __len__(self):
        length = len(self.samples)
        return length


class LIVECDATASET_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "clive_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                image_data[fields[0]] = (fields[2], fields[3])

        imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        imgpath = imgpath["AllImages_release"]
        imgpath = imgpath[7:1169]
        mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        labels = mos["AllMOS_release"].astype(np.float32)
        labels = labels[0][7:1169]

        # data = pd.read_csv("/home/pws/IQA/global_local/IQA/livec.csv", sep='\t', header=None)
        # dist_type = data.iloc[index, 2].astype(str)
        # print(type(dist_type))
        # scene = data.iloc[index, 3].astype(str)
        # scene_content2 = data.iloc[index, 4]
        # scene_content3 = data.iloc[index, 5]
        
        sample = []
        for _, item in enumerate(index):
            for aug in range(patch_num):
                image_path = "Images/" + imgpath[item][0][0]
                dist, scene = image_data.get(image_path)
                # if scene2 == "invalid":
                #     scene_label2 = -1
                # else:
                #     scene_label2 = scene2label[scene2]
                # if scene3 == "invalid":
                #     scene_label3 = -1
                # else:
                #     scene_label3 = scene2label[scene3]
                
                sample.append(
                    (os.path.join(root, "Images", imgpath[item][0][0]), labels[item],
                     scene2label[scene], map2label[dist_map[dist]])
                )
        
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, scene, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, distortion

    def __len__(self):
        length = len(self.samples)
        return length


class BIDDATASET_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "bid_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                image_data[fields[0]] = (fields[1], fields[2], fields[3])

        # imgpath = io.loadmat(os.path.join(root, "Data", "AllImages_release.mat"))
        # imgpath = imgpath["AllImages_release"]
        # imgpath = imgpath[7:1169]
        # mos = io.loadmat(os.path.join(root, "Data", "AllMOS_release.mat"))
        # labels = mos["AllMOS_release"].astype(np.float32)
        # labels = labels[0][7:1169]

        # data = pd.read_csv("/home/pws/IQA/global_local/IQA/livec.csv", sep='\t', header=None)
        # dist_type = data.iloc[index, 2].astype(str)
        # print(type(dist_type))
        # scene = data.iloc[index, 3].astype(str)
        # scene_content2 = data.iloc[index, 4]
        # scene_content3 = data.iloc[index, 5]
        
        sample = []
        for _, item in enumerate(index):
            for aug in range(patch_num):
                num_str = str(item).zfill(4)
                # print(num_str)
                try:
                    image_path = "ImageDatabase/DatabaseImage" + num_str + ".JPG"
                    label, dist, scene = image_data.get(image_path)
                    sample.append((os.path.join(root, image_path.split("ImageDatabase/")[-1]), round(transfer("bid", float(label)), 4),
                                scene2label[scene], map2label[dist_map[dist]]))
                except TypeError:
                    print(f"{image_path} does not exist.")


        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, scene, distortion= self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, distortion

    def __len__(self):
        length = len(self.samples)
        return length


class LIVEDataset_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        
        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "live_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                image_data[fields[0]] = (fields[2], fields[3])
                # print(fields[2], fields[3])
        # print(image_data)        

        refpath = os.path.join(root, "refimgs")
        refname = getFileName(refpath, ".bmp")

        jp2kroot = os.path.join(root, "jp2k")
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, "jpeg")
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, "wn")
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, "gblur")
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, "fastfading")
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        dmos = io.loadmat(os.path.join(root, "dmos_realigned.mat"))
        labels = dmos["dmos_new"].astype(np.float32)

        orgs = dmos["orgs"]
        refnames_all = io.loadmat(os.path.join(root, "refnames_all.mat"))
        refnames_all = refnames_all["refnames_all"]

        refname.sort()
        sample = []

        for i in range(0, len(index)):
            train_sel = refname[index[i]] == refnames_all
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                label = transfer("live", labels[0][item])
                # if num <= 0 or num > 100:
                #     continue
                # if labels[0][item] <= 0 or labels[0][item] > 100:
                #     continue
                for aug in range(patch_num):
                    # sample.append((imgpath[item], transfer("live", labels[0][item])))
                    # split = imgpath[item].split("_")
                    # category = split[0][1:]
                    # distortion = (int(split[1]) - 1) * 5 + int(split[2].split(".")[0])
                    # count = (int(split[0][1:]) - 1)*125 + distortion - 1
                    # print("aaaa", labels[0][item])
                    try:
                        find_path = imgpath[item].split(root + '/')[-1]
                        # print(f"findpath:{find_path}")
                        dist, scene = image_data.get(find_path)
                        sample.append(
                            (
                                os.path.join(imgpath[item]),
                                label,
                                scene2label[scene], 
                                map2label[dist_map[dist]]
                                # scene,
                                # dist
                            )
                        )
                    except:
                        print("key error")
                    
                # print(math.floor(transfer("live", labels[0][item])))
                # print(labels[0][item])
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, scene, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, distortion
    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = "%s%s%s" % ("img", str(index), ".bmp")
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


class TID2013Dataset_clip(data.Dataset):

    def __init__(self, root, index, patch_num, transform=None):
        refpath = os.path.join(root, "reference_images")
        refname = getTIDFileName(refpath, ".bmp.BMP")
        txtpath = os.path.join(root, "mos_with_names.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])
        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        refname.sort()
        sample = []

        for i, item in enumerate(index):

            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    # sample.append(
                    #     (
                    #         os.path.join(root, "distorted_images", imgnames[item]),
                    #         transfer("tid2013", labels[item])
                    #     )
                    # )
                    split = imgnames[item].split("_")
                    category = split[0][1:]
                    distortion = (int(split[1]) - 1) * 5 + int(split[2].split(".")[0])
                    # count = (int(split[0][1:]) - 1)*125 + distortion - 1
                    sample.append(
                        (
                            os.path.join(root, "distorted_images", imgnames[item]),
                            transfer("tid2013", labels[item]),
                            math.floor(transfer("tid2013", labels[item])),
                            # count,
                            category,
                            distortion_map[distortion],
                        )
                    )
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, num, category, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, num, category, distortion

    def __len__(self):
        length = len(self.samples)
        return length


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


class CSIQDataset_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        
        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "csiq_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                fields[0] = fields[0].split('/')[0] + '/' + fields[0].split('/')[-1]
                image_data[fields[0]] = (fields[2], fields[3])

        refpath = os.path.join(root, "src_imgs")
        refname = getFileName(refpath, ".png")
        txtpath = os.path.join(root, "csiq_label.txt")
        fh = open(txtpath, "r")
        imgnames = []
        target = []
        refnames_all = []
        for line in fh:
            line = line.split("\n")
            words = line[0].split()
            imgnames.append((words[0]))
            target.append(words[1])
            ref_temp = words[0].split(".")
            refnames_all.append(ref_temp[0] + "." + "png")

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                label = transfer("csiq", labels[item])
                for aug in range(patch_num):
                    try:
                        image_path = os.path.join(root, "dst_imgs", imgnames[item]+".png")
                        find_path = image_path.split(root + '/')[-1]
                        # print(f"findpath:{find_path}")
                        dist, scene = image_data.get(find_path)
                        sample.append(
                            (
                                os.path.join(root, "dst_imgs_all", imgnames[item]+".png"),
                                label,
                                scene2label[scene], 
                                map2label[dist_map[dist]]
                                # scene,
                                # dist
                            )
                        )
                    except:
                        print("key error")
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, scene, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, distortion

    def __len__(self):
        length = len(self.samples)
        return length


class KADIDDataset_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):

        image_data = {}
        # 使用相对路径
        label_path = os.path.join(os.path.dirname(__file__), "kadid10k_all_clip.txt")
        with open(label_path, "r") as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 6:
                    print(len(fields))
                    continue
                # fields[0] = fields[0].split('/')[0] + '/' + fields[0].split('/')[-1]
                image_data[fields[0]] = (fields[2], fields[3])
                
        refpath = os.path.join(root, "reference_images")
        refname = getTIDFileName(refpath, ".png.PNG")

        imgnames = []
        target = []
        refnames_all = []

        csv_file = os.path.join(root, "dmos.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row["dist_img"])
                refnames_all.append(row["ref_img"][1:3])

                mos = np.array(float(row["dmos"])).astype(np.float32)
                target.append(mos)

        labels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)
        refname.sort()
        sample = []
        print("index", index)
        # count = 0
        for i, item in enumerate(index):
            train_sel = refname[index[i]] == refnames_all
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                label = transfer("kadid", labels[item])
                for aug in range(patch_num):
                    try:
                        image_path = os.path.join(root, "images", imgnames[item])
                        find_path = image_path.split(root + '/')[-1]
                        dist, scene = image_data.get(find_path)
                        sample.append(
                            (
                                image_path,
                                label,
                                scene2label[scene], 
                                map2label[dist_map[dist]]
                                # scene,
                                # dist
                            )
                        )
                    except:
                        print("key error")

                # for _ in range(patch_num):
                    
                #     # split = imgnames[item].split("_")
                #     # category = split[0][1:]
                #     # distortion = (int(split[1]) - 1)*5 + int(split[2].split(".")[0])
                #     # count = (int(split[0][1:]) - 1)*125 + distortion - 1
                #     sample.append(
                #         (
                #             os.path.join(root, "images", imgnames[item]),
                #             transfer("kadid", labels[item]),
                #             math.floor(transfer("kadid", labels[item])),
                #             # count,
                #             category,
                #             distortion_map[distortion],
                #         )
                #     )
                # count += 1
        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, category, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, category, distortion

    def __len__(self):
        length = len(self.samples)
        return length


def get_labels(config):
    root = config.DATA.DATA_PATH
    refpath = os.path.join(root, "images")
    refname = getTIDFileName(refpath, ".png.PNG")

    imgnames = []
    target = []
    refnames_all = []

    csv_file = os.path.join(root, "dmos.csv")
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            imgnames.append(row["dist_img"])
            refnames_all.append(row["ref_img"][1:3])

            mos = np.array(float(row["dmos"])).astype(np.float32)
            target.append(mos)

    labels = np.array(target).astype(np.float32)
    refnames_all = np.array(refnames_all)
    refname.sort()
    sample = []

    for i, _ in enumerate(config.SET.TRAIN_INDEX):
        train_sel = refname[i] == refnames_all
        train_sel = np.where(train_sel == True)
        train_sel = train_sel[0].tolist()
        for j, item in enumerate(train_sel):
            sample.append(transfer("kadid", labels[item]))
    return sample


class SPAQDATASET_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        super(SPAQDATASET_clip, self).__init__()

        self.data_path = root
        anno_folder = os.path.join(self.data_path, "Annotations")
        xlsx_file = os.path.join(anno_folder, "MOS and Image attribute scores.xlsx")
        read = pd.read_excel(xlsx_file)
        imgname = read["Image name"].values.tolist()
        mos_all = read["MOS"].values.tolist()
        for i in range(len(mos_all)):
            mos_all[i] = np.array(mos_all[i]).astype(np.float32)
        sample = []
        for _, item in enumerate(index):
            for _ in range(patch_num):
                # sample.append(
                #     (
                #         os.path.join(
                #             self.data_path,
                #             "SPAQ zip",
                #             "512x384",
                #             imgname[item],
                #         ),
                #         mos_all[item],
                #     )
                # )
                # split = imgname[item].split("_")
                # category = split[0][1:]
                # distortion = (int(split[1]) - 1) * 5 + int(split[2].split(".")[0])
                # count = (int(split[0][1:]) - 1)*125 + distortion - 1
                sample.append(
                    (
                        os.path.join(root, "512x384", imgname[item]),
                        mos_all[item],
                        math.floor(mos_all[item]),
                        # count,
                        1,
                        1
                    )
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, num, category, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, num, category, distortion

    def __len__(self):
        length = len(self.samples)
        return length


class FBLIVEFolder_clip(data.Dataset):
    def __init__(self, root, index, patch_num, transform=None):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, "labels_image.csv")
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row["name"])
                mos = np.array(float(row["mos"])).astype(np.float32)
                mos_all.append(mos)

        sample = []
        for _, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append(
                #     (os.path.join(root, "database", imgname[item]), mos_all[item])
                # )
                # split = imgname[item].split("_")
                # category = split[0][1:]
                # distortion = (int(split[1]) - 1) * 5 + int(split[2].split(".")[0])
                # count = (int(split[0][1:]) - 1)*125 + distortion - 1
                sample.append(
                    (
                        os.path.join(root, "database", imgname[item]),
                        mos_all[item],
                        math.floor(mos_all[item]),
                        # count,
                        1,
                        1
                    )
                )

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, num, category, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, num, category, distortion

    def __len__(self):
        length = len(self.samples)
        return length


# CVIU17 纹理失真类型映射 (对应 models/clip_vit.py 中的 texture_dists)
cviu17_texture_map = {
    'none': 0, 'uncertain': 1, 'noise_amplification': 2, 'ringing_halo': 3, 
    'checkerboard': 4, 'moire': 5, 'false_texture_hallucination': 6, 
    'texture_smoothing': 7, 'over_sharpening': 8, 'compression_blockiness': 9, 
    'other_artifact': 10
}

# CVIU17 结构失真类型映射 (对应 models/clip_vit.py 中的 structure_dists)
cviu17_structure_map = {
    'none': 0, 'uncertain': 1, 'edge_blur': 2, 'detail_loss': 3, 
    'geometric_distortion': 4, 'aliasing_jaggies': 5
}

# SR 数据集 distortion_type 标签 → map2label 索引映射
# 标注 prompt 输出的标签名 → 模型 dists_map 索引
sr_dist2label = {
    'jpeg2000_compression': 0, 'jpeg_compression': 1, 'noise': 2, 'blur': 3,
    'color_distortion': 4, 'contrast_distortion': 5, 'overexposure': 6,
    'underexposure': 7, 'spatial_local_distortion': 8, 'quantization': 9,
    'other_distortion': 10, 'none': 10,
    # 兼容旧格式（带空格的 map2label key）
    'jpeg2000 compression': 0, 'jpeg compression': 1, 'color': 4,
    'contrast': 5, 'spatial': 8, 'other': 10,
}


class CVIU17DATASET_clip(data.Dataset):
    """CVIU17 数据集，支持场景、纹理失真、结构失真和粗粒度失真标签（6元组输出）"""
    def __init__(self, root, index, patch_num, transform=None):
        # 读取标签文件，建立索引到图片信息的映射
        label_path = os.path.join(os.path.dirname(__file__), "cviu17_all_clip_sr.txt")
        
        # 按行读取，保存为列表（按索引访问）
        all_images = []
        with open(label_path, "r") as f:
            for line in f:
                # 跳过注释行和空行
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.strip().split('\t')
                # 需要至少5列: path, mos, scene, texture, structure [, distortion_type]
                if len(fields) < 5:
                    continue
                all_images.append({
                    'path': fields[0],
                    'mos': float(fields[1]),
                    'scene': fields[2],
                    'distortion': fields[3] if len(fields) > 3 else 'other_distortion',
                    'texture': fields[4] if len(fields) > 4 else 'other_artifact',
                    'structure': fields[5] if len(fields) > 5 else 'none',
                })

        sample = []
        for _, item in enumerate(index):
            if item >= len(all_images):
                continue
            for aug in range(patch_num):
                try:
                    img_info = all_images[item]
                    # 场景标签
                    scene_label = scene2label.get(img_info['scene'], scene2label['others'])
                    # 纹理失真标签
                    texture_label = cviu17_texture_map.get(img_info['texture'], cviu17_texture_map['other_artifact'])
                    # 结构失真标签
                    structure_label = cviu17_structure_map.get(img_info['structure'], cviu17_structure_map['none'])
                    # 粗粒度失真标签
                    dist_label = sr_dist2label.get(img_info['distortion'], sr_dist2label['other_distortion'])
                    
                    sample.append((
                        os.path.join(root, img_info['path']),
                        transfer("cviu17", img_info['mos']),  # 转换到 0-100
                        scene_label,
                        texture_label,
                        structure_label,
                        dist_label,
                    ))
                except (TypeError, KeyError) as e:
                    print(f"Error processing index {item}: {e}")

        # 预过滤损坏文件
        bad_paths = set()
        for s in sample:
            path = s[0]
            if path not in bad_paths:
                try:
                    with Image.open(path) as im:
                        im.load()
                except Exception:
                    bad_paths.add(path)
        if bad_paths:
            print(f"[CVIU17] 发现 {len(bad_paths)} 个损坏文件，已从数据集中移除")
            sample = [s for s in sample if s[0] not in bad_paths]

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """返回 6 元组: (image, target, scene, texture, structure, distortion)"""
        path, target, scene, texture, structure, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, texture, structure, distortion

    def __len__(self):
        return len(self.samples)


# ============ SISAR 数据集 ============
class SISARDATASET_clip(data.Dataset):
    """SISAR 数据集，支持场景、纹理失真、结构失真和粗粒度失真标签（6元组输出）"""
    def __init__(self, root, index, patch_num, transform=None):
        # TODO: 修改为你的标签文件名
        label_path = os.path.join(os.path.dirname(__file__), "sisar_all_clip_sr.txt")
        
        all_images = []
        with open(label_path, "r") as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                all_images.append({
                    'path': fields[0],
                    'mos': float(fields[1]),
                    'scene': fields[2],
                    'distortion': fields[3] if len(fields) > 3 else 'other_distortion',
                    'texture': fields[4] if len(fields) > 4 else 'other_artifact',
                    'structure': fields[5] if len(fields) > 5 else 'none',
                })

        sample = []
        for _, item in enumerate(index):
            if item >= len(all_images):
                continue
            for aug in range(patch_num):
                try:
                    img_info = all_images[item]
                    scene_label = scene2label.get(img_info['scene'], scene2label['others'])
                    texture_label = cviu17_texture_map.get(img_info['texture'], cviu17_texture_map['other_artifact'])
                    structure_label = cviu17_structure_map.get(img_info['structure'], cviu17_structure_map['none'])
                    dist_label = sr_dist2label.get(img_info['distortion'], sr_dist2label['other_distortion'])
                    
                    sample.append((
                        os.path.join(root, img_info['path']).replace('\\', '/'),
                        transfer("sisar", img_info['mos']),  # TODO: 确认 MOS 范围，在 transfer 函数中添加对应转换
                        scene_label,
                        texture_label,
                        structure_label,
                        dist_label,
                    ))
                except (TypeError, KeyError) as e:
                    print(f"Error processing index {item}: {e}")

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        path, target, scene, texture, structure, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, texture, structure, distortion

    def __len__(self):
        return len(self.samples)


# ============ QADS 数据集 ============
class QADSDATASET_clip(data.Dataset):
    """QADS 数据集，支持场景、纹理失真、结构失真和粗粒度失真标签（6元组输出）"""
    def __init__(self, root, index, patch_num, transform=None):
        # TODO: 修改为你的标签文件名
        label_path = os.path.join(os.path.dirname(__file__), "qads_all_clip_sr.txt")
        
        all_images = []
        with open(label_path, "r") as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                all_images.append({
                    'path': fields[0],
                    'mos': float(fields[1]),
                    'scene': fields[2],
                    'distortion': fields[3] if len(fields) > 3 else 'other_distortion',
                    'texture': fields[4] if len(fields) > 4 else 'other_artifact',
                    'structure': fields[5] if len(fields) > 5 else 'none',
                })

        sample = []
        for _, item in enumerate(index):
            if item >= len(all_images):
                continue
            for aug in range(patch_num):
                try:
                    img_info = all_images[item]
                    scene_label = scene2label.get(img_info['scene'], scene2label['others'])
                    texture_label = cviu17_texture_map.get(img_info['texture'], cviu17_texture_map['other_artifact'])
                    structure_label = cviu17_structure_map.get(img_info['structure'], cviu17_structure_map['none'])
                    dist_label = sr_dist2label.get(img_info['distortion'], sr_dist2label['other_distortion'])
                    
                    sample.append((
                        os.path.join(root, img_info['path']),
                        transfer("qads", img_info['mos']),
                        scene_label,
                        texture_label,
                        structure_label,
                        dist_label,
                    ))
                except (TypeError, KeyError) as e:
                    print(f"Error processing index {item}: {e}")

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        path, target, scene, texture, structure, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, texture, structure, distortion

    def __len__(self):
        return len(self.samples)


# ============ Waterloo15 数据集 ============
class WATERLOO15DATASET_clip(data.Dataset):
    """Waterloo15 数据集，支持场景、纹理失真、结构失真和粗粒度失真标签（6元组输出）"""
    def __init__(self, root, index, patch_num, transform=None):
        # TODO: 修改为你的标签文件名
        label_path = os.path.join(os.path.dirname(__file__), "waterloo15_all_clip_sr.txt")
        
        all_images = []
        with open(label_path, "r") as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                all_images.append({
                    'path': fields[0],
                    'mos': float(fields[1]),
                    'scene': fields[2],
                    'distortion': fields[3] if len(fields) > 3 else 'other_distortion',
                    'texture': fields[4] if len(fields) > 4 else 'other_artifact',
                    'structure': fields[5] if len(fields) > 5 else 'none',
                })

        sample = []
        for _, item in enumerate(index):
            if item >= len(all_images):
                continue
            for aug in range(patch_num):
                try:
                    img_info = all_images[item]
                    scene_label = scene2label.get(img_info['scene'], scene2label['others'])
                    texture_label = cviu17_texture_map.get(img_info['texture'], cviu17_texture_map['other_artifact'])
                    structure_label = cviu17_structure_map.get(img_info['structure'], cviu17_structure_map['none'])
                    dist_label = sr_dist2label.get(img_info['distortion'], sr_dist2label['other_distortion'])
                    
                    sample.append((
                        os.path.join(root, img_info['path']),
                        transfer("waterloo15", img_info['mos']),
                        scene_label,
                        texture_label,
                        structure_label,
                        dist_label,
                    ))
                except (TypeError, KeyError) as e:
                    print(f"Error processing index {item}: {e}")

        # 预过滤损坏文件，避免用随机噪声污染评估结果
        bad_paths = set()
        for s in sample:
            path = s[0]
            if path not in bad_paths:
                try:
                    with Image.open(path) as im:
                        im.load()  # 实际解码，比 verify() 更可靠
                except Exception:
                    bad_paths.add(path)
        if bad_paths:
            print(f"[WATERLOO15] 发现 {len(bad_paths)} 个损坏文件，已从数据集中移除")
            sample = [s for s in sample if s[0] not in bad_paths]

        self.samples = sample
        self.transform = transform

    def _load_image(self, path):
        try:
            im = Image.open(path).convert("RGB")
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        path, target, scene, texture, structure, distortion = self.samples[index]
        sample = self._load_image(path)
        sample = self.transform(sample)
        return sample, target, scene, texture, structure, distortion

    def __len__(self):
        return len(self.samples)
