import os
import time
from typing import List, Optional

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchreid import metrics
from torchreid.utils import AverageMeter, visualize_ranked_results


class CustomEvaluationEngine:
    """
    Parameters
    ------
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.

    Examples
    ------
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        ).cuda()

        engine = CustomEvaluationEngine(
            datamanager, model, save_dir='log/resnet50-softmax-market1501'
        )
        engine.run()
    """

    def __init__(
        self,
        datamanager,
        model,
        dist_metric: Optional[str] = "euclidean",
        normalize_feature: Optional[bool] = False,
        visrank: Optional[bool] = False,
        visrank_topk: Optional[int] = 10,
        save_dir: Optional[str] = "",
        use_metric_cuhk03: Optional[bool] = False,
        ranks: Optional[List[int]] = None,
        rerank: Optional[bool] = False,
    ):
        # property for input data
        self.datamanager = datamanager
        # self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader

        # property for model
        self.epoch = 0
        self.model = model.eval()

        # property for evaluation
        self.use_gpu = torch.cuda.is_available()
        self.dist_metric = dist_metric
        self.normalize_feature = normalize_feature
        self.visrank = visrank
        self.visrank_topk = visrank_topk
        self.save_dir = save_dir
        self.use_metric_cuhk03 = use_metric_cuhk03
        self.ranks = ranks or [1, 5, 10, 20]
        self.rerank = rerank
        self.writer = SummaryWriter(log_dir=save_dir)

    def run(self):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.

        Returns
        ------
        rank1 : int
            TOP@1 のスコア
        distmat : np.ndarray (Num of query, Num of gallery)
            各 query 画像と各 gallery 画像との間の score
        """
        targets = list(self.test_loader.keys())

        for dataset_name in targets:
            domain = "source" if dataset_name in self.datamanager.sources else "target"
            print("##### Evaluating {} ({}) #####".format(dataset_name, domain))
            query_loader = self.test_loader[dataset_name]["query"]
            gallery_loader = self.test_loader[dataset_name]["gallery"]
            rank1, mAP, distmat = self._evaluate(
                dataset_name=dataset_name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
            )

            if self.writer is not None:
                self.writer.add_scalar(f"Test/{dataset_name}/rank1", rank1, self.epoch)
                self.writer.add_scalar(f"Test/{dataset_name}/mAP", mAP, self.epoch)

        # NOTE: distmat is added to the Returns
        return rank1, distmat

    def parse_data_for_eval(self, data):
        imgs = data["img"]
        pids = data["pid"]
        camids = data["camid"]
        return imgs, pids, camids

    def extract_features(self, input):
        return self.model(input)

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name="",
        query_loader=None,
        gallery_loader=None,
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            features_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.cpu()
                features_.append(features)
                pids_.extend(pids.tolist())
                camids_.extend(camids.tolist())

            features_ = torch.cat(features_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return features_, pids_, camids_

        print("Extracting features from query set ...")
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print("Done, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        print("Extracting features from gallery set ...")
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print("Done, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        print("Speed: {:.4f} sec/batch".format(batch_time.avg))

        if self.normalize_feature:
            print("Normalzing features with L2 norm ...")
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print("Computing distance matrix with metric={} ...".format(self.dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, self.dist_metric)
        distmat = distmat.numpy()

        # if self.rerank:
        #     print('Applying person re-ranking ...')
        #     distmat_qq = metrics.compute_distance_matrix(qf, qf, self.dist_metric)
        #     distmat_gg = metrics.compute_distance_matrix(gf, gf, self.dist_metric)
        #     distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print("Computing CMC and mAP ...")
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=self.use_metric_cuhk03,
        )

        print("** Results **")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in self.ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))

        if self.visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=os.path.join(self.save_dir, "visrank_" + dataset_name),
                topk=self.visrank_topk,
            )

        # NOTE: distmat is added to the Returns
        return cmc[0], mAP, distmat
