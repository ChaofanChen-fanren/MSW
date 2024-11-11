import torch
import argparse
import os
import numpy as np
import random
import logging
from prefetch_generator import BackgroundGenerator
from models import clip, FiLo
import torchvision.transforms as transforms
from datasets import Dataset
from utils import FocalLoss, BinaryDiceLoss
from tqdm import tqdm
import torch.nn.functional as F


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_logger(save_path):
    logger = logging.getLogger("Train Anomaly")
    logger.setLevel(logging.INFO)  # Set the level for the logger itself

    # Avoid duplicate handlers if this function is called multiple times
    if not logger.hasHandlers():
        # # Create a console handler and set the level to info
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)

        # Create a file handler and set the level to info
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_handler = logging.FileHandler(os.path.join(save_path, "log.txt"))
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add both handlers to the logger
        # logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def train(args):
    logger = set_logger(os.path.join(args.save_path, args.dataset))

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = args.device
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epoch
    dataset_name = args.dataset

    # prepare data
    logger.info("=============prepare data==============")
    _, _, preprocess_val = clip.create_model_and_transforms(args.clip_model, img_size=args.image_size,
                                                            pretrained=args.clip_pretrained)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    train_data = Dataset(root=args.dataset_path, transform=preprocess_val, target_transform=transform,
                         dataset_name=args.dataset)
    train_dataloader = DataLoaderX(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    # prepare model
    logger.info("=============prepare model==============")
    learn_prompt_cfg = {
        'dataset': 'visa',
        'feature_dim': 768,
        'n_ctx': args.n_ctx,
        'device': device
    }
    filo_model = FiLo(learn_prompt_cfg, args, device).to(device)

    main_part_param_groups = [
        {'params': filo_model.decoder_cov.parameters(), 'lr': args.decoder_learning_rate},
        {'params': filo_model.decoder_linear.parameters(), 'lr': args.decoder_learning_rate},
        {'params': filo_model.normal_prompt_learner.parameters(), 'lr': args.learning_rate},
        {'params': filo_model.abnormal_prompt_learner.parameters(), 'lr': args.learning_rate}
    ]

    optimizer_main_part = torch.optim.AdamW(
        main_part_param_groups,
        betas=(0.5, 0.999),
    )

    adapter_param_groups = [
        {'params': filo_model.adapter.parameters(), 'lr': args.adapter_learning_rate},
    ]

    optimizer_adapter = torch.optim.AdamW(
        adapter_param_groups,
        betas=(0.5, 0.999),
    )

    logger.info("=============prepare loss===============")
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    logger.info("=============Train start==============")
    for epoch in range(epochs):
        pixel_loss_list = []
        for items in tqdm(train_dataloader):
            # image = items["img"].to(device)
            # cls_name = items["cls_name"][0]
            # image_path = items["img_path"]
            # anomaly_cls = items["anomaly_class"][0]
            # label = items['anomaly'].to(device)
            text_probs, anomaly_maps = filo_model(items, with_adapter=False)

            # losses
            gt = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            pixel_loss = 0
            for num in range(len(anomaly_maps)):
                pixel_loss += loss_focal(anomaly_maps[num], gt)
                pixel_loss += loss_dice(anomaly_maps[num][:, 1, :, :], gt)
                pixel_loss += loss_dice(anomaly_maps[num][:, 0, :, :], 1 - gt)

            optimizer_main_part.zero_grad()
            pixel_loss.backward()
            optimizer_main_part.step()

            pixel_loss_list.append(pixel_loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], pixel_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(pixel_loss_list)))

    for epoch in range(args.adapter_epoch):
        image_loss_list = []
        for items in tqdm(train_dataloader):
            # image = items["img"].to(device)
            # cls_name = items["cls_name"][0]
            # image_path = items["img_path"]
            # anomaly_cls = items["anomaly_class"][0]
            label = items['anomaly'][0].to(device)
            text_probs, anomaly_maps = filo_model(items, only_train_adapter=True, with_adapter=True)

            # losses
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label)

            optimizer_adapter.zero_grad()
            image_loss.backward()
            optimizer_adapter.step()

            image_loss_list.append(image_loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], img_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(image_loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            save_name = os.path.join(args.save_path, dataset_name)
            ckp_path = os.path.join(save_name, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"filo": filo_model.state_dict()}, ckp_path)
    logger.info("=============Train end==============")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint/filo', help='path to save results')
    parser.add_argument("--clip_model", type=str, default="ViT-L-14-336", help="clip model name")
    parser.add_argument("--clip_pretrained", type=str, default="openai", help="pretrained clip model wight path")

    parser.add_argument("--dataset_path", type=str, default='/Users/chenchaofan/python_project/data/VisA', help="train dataset path")
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")

    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--decoder_learning_rate", type=float, default=0.0001, help="learning rate for decoder")
    parser.add_argument("--adapter_learning_rate", type=float, default=0.00001, help="learning rate for adapter")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
