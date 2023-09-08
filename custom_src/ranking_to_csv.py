from datetime import date
from typing import Dict, List, Tuple, TypedDict, Union

import numpy as np
import pandas as pd


class PidCamidPair(TypedDict):
    pid: int
    camid: str


class CsvCreator:
    def __init__(self) -> None:
        pass

    def _prepare_mapping(self, dataset) -> Dict[int, str]:
        """
        gallery 画像の index から
        画像ファイル名 への mapping データを作成

        Parameters
        ------
        dataset : List[Tuple[str, int, int, int]]
            query か gallery のデータセット
            Tuple 内の各要素は impath, pid, camid, <不明> の4つ

        Returns
        ------
        df_gallery_to_impath : pd.DataFrame
            Columns
            ------
            g_idx : int
                gallery 画像の index
            imname : str
                画像ファイル名
        """
        mapping: Dict[int, str] = {}
        for i, data_tuple in enumerate(dataset):
            impath, pid, camid, _ = data_tuple
            mapping[i] = impath.split("/")[-1]

        return mapping

    def _validate_data_shape(self, distmat, query, gallery):
        num_q, num_g = distmat.shape
        assert num_q == len(query), "query データセットのデータ数と、スコア出力結果の query 数が一致していません"
        assert num_g == len(gallery), "gallery データセットのデータ数と、スコア出力結果の gallery 数が一致していません"

    def _create_df_distance(
        self, distmat: np.ndarray, rank_max: int = 10
    ) -> pd.DataFrame:
        """
        各画像間の距離を、昇順に並べたものを DataFrame にして返す

        Returns
        ------
        pd.DataFrame - shape: (Num of query, rank_max)
        """
        distances: np.ndarray = np.sort(distmat, axis=1)[:, :rank_max]
        df_distances: pd.DataFrame = pd.DataFrame(distances)
        return df_distances

    def _create_df_imnames(
        self,
        distmat: np.ndarray,
        gallery,
        rank_max: int = 10,
    ) -> pd.DataFrame:
        """
        各画像間の距離の昇順に画像ファイル名を並べたものを DataFrame にして返す

        Returns
        ------
        pd.DataFrame - shape: (Num of query, rank_max)
        """

        def _g_index_to_imname(gallery_index: int, mapping: Dict[int, str]) -> str:
            return mapping[gallery_index]

        rankings: np.ndarray = np.argsort(distmat, axis=1)[:, :rank_max]
        _df_imnames: pd.DataFrame = pd.DataFrame(rankings)

        mapping_gindex_to_imname: Dict[int, str] = self._prepare_mapping(gallery)
        df_imnames: pd.DataFrame = _df_imnames.applymap(
            _g_index_to_imname, mapping=mapping_gindex_to_imname
        )
        return df_imnames

    def _merge_into_ranking(
        self,
        df_distances: pd.DataFrame,
        df_imnames: pd.DataFrame,
        q_image_names: pd.Series,
    ) -> pd.DataFrame:
        """
        CSVに出力しやすく、かつ出力後も見やすいように各データをまとめ、整形する
        """
        df_ranking = df_imnames.merge(
            df_distances,
            left_index=True,
            right_index=True,
            how="outer",
            suffixes=("_image", "_distance"),
        ).sort_index(axis="columns")

        df_ranking.insert(0, "query_image", q_image_names)
        df_ranking = df_ranking.sort_values(["query_image"]).reset_index(drop=True)
        return df_ranking

    def _extract_pid_camid(self, query) -> Tuple[List[str], List[PidCamidPair]]:
        """
        query 画像のファイル名一覧を抽出
        """
        impaths: List[str] = []
        p_cam_pairs: List[PidCamidPair] = []
        for impath, pid, cid, _ in query:
            impaths.append(impath.split("/")[-1])
            p_cam_pairs.append({"pid": pid, "camid": cid})

        return impaths, p_cam_pairs

    def _create_valid_distances(
        self, distmat, q_p_cam_pairs: List[PidCamidPair], gallery
    ) -> np.ndarray:
        """
        query, gallery 間で pid, camid が共に一致する画像の distmat を nan で上書き

        Parameters
        ------
        distmat : np.ndarray - shape: (Num of Query, Num of Gallery)

        Returns
        ------
        np.ndarray - shape: (Num of Query, Num of Gallery)
        """
        valid_indices = []
        _, g_p_cam_pairs = self._extract_pid_camid(gallery)
        for q_p_cam_pair in q_p_cam_pairs:
            g_pidcamids: pd.Series = pd.Series(
                [
                    f"{g_p_cam_pair['pid']}_{g_p_cam_pair['camid']}"
                    for g_p_cam_pair in g_p_cam_pairs
                ]
            )
            remove_index: pd.Series = (
                g_pidcamids == f"{q_p_cam_pair['pid']}_{q_p_cam_pair['camid']}"
            )
            valid_indices.append(~remove_index)

        valid_indices: np.ndarray = np.array(valid_indices)
        # query, gallery 間で pid, camid 共に一致する画像の distmat を nan で上書き
        only_valid_dists: np.ndarray = np.where(valid_indices, distmat, np.nan)

        assert distmat.shape == only_valid_dists.shape

        return only_valid_dists

    def run(self, distmat: np.ndarray, datamanager, dataset_name: str) -> None:
        """
        distmat を整形して CSV に出力する

        Parameters
        ------
        distmat : np.ndarray - shape: (Num of query, Num of gallery)
            各 query 画像と各 gallery 画像との間の score
        datamanager : torchreid.data.DataManager
            データセットの情報を保持するオブジェクト
        dataset_name : str
            "market1501" など
        """
        # NOTE: データセットの情報を取得
        dataset = datamanager.fetch_test_loaders(dataset_name)
        query, gallery = dataset
        self._validate_data_shape(distmat, query, gallery)

        # NOTE: 中間データを作成
        q_image_names: List[str]
        q_p_cam_pairs: List[PidCamidPair]
        q_image_names, q_p_cam_pairs = self._extract_pid_camid(query)
        valid_distmat: np.ndarray = self._create_valid_distances(
            distmat, q_p_cam_pairs, gallery
        )

        # NOTE: 10 位までの 画像ファイル名と distance のみに絞る
        df_distances: pd.DataFrame = self._create_df_distance(valid_distmat)
        df_imnames: pd.DataFrame = self._create_df_imnames(valid_distmat, gallery)
        df_ranking: pd.DataFrame = self._merge_into_ranking(
            df_distances, df_imnames, q_image_names
        )

        # NOTE: CSV 出力
        csv_name: str = f"{date.today().strftime('%Y%m%d')}_similarity_ranking.csv"
        df_ranking.to_csv(csv_name)
        print(f"{csv_name} is created!")
