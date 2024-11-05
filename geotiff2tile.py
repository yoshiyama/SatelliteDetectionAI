from osgeo import gdal
import os
import logging
from datetime import datetime
from typing import List, Dict
import argparse
import sys


class GeoImagePyramid:
    """地理空間画像のピラミッド処理を行うクラス"""

    def __init__(self, input_path: str):
        """
        初期化メソッド

        Args:
            input_path (str): 入力画像のパス
        """
        self.input_path = os.path.abspath(input_path)
        self.input_dir = os.path.dirname(self.input_path)
        self.input_filename = os.path.basename(self.input_path)

        # 出力ファイル名の生成
        name, ext = os.path.splitext(self.input_filename)
        self.output_path = os.path.join(self.input_dir, f"{name}_tile{ext}")

        self.dataset = None
        self.image_info = {}

        # ロギングの設定
        self._setup_logging()

    def _setup_logging(self):
        """ロギングの設定"""
        log_file = os.path.join(
            self.input_dir,
            f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_image(self) -> bool:
        """
        画像を読み込み、基本情報を取得

        Returns:
            bool: 読み込み成功の場合True
        """
        try:
            self.dataset = gdal.Open(self.input_path)
            if self.dataset is None:
                self.logger.error(f"画像を開けません: {self.input_path}")
                return False

            self.image_info = {
                'width': self.dataset.RasterXSize,
                'height': self.dataset.RasterYSize,
                'bands': self.dataset.RasterCount,
                'projection': self.dataset.GetProjection(),
                'geotransform': self.dataset.GetGeoTransform(),
                'datatype': self.dataset.GetRasterBand(1).DataType
            }

            self.logger.info(f"画像情報: {self.image_info}")
            self.logger.info(f"出力先: {self.output_path}")
            return True

        except Exception as e:
            self.logger.error(f"画像読み込みエラー: {str(e)}")
            return False

    def calculate_pyramid_levels(self, min_size: int = 256) -> List[int]:
        """
        ピラミッドレベルを計算

        Args:
            min_size (int): 最小サイズ（ピクセル）

        Returns:
            List[int]: ピラミッドレベルのリスト
        """
        levels = []
        max_dimension = max(self.image_info['width'], self.image_info['height'])
        current_level = 2

        while max_dimension // current_level >= min_size:
            levels.append(current_level)
            current_level *= 2

        return levels

    def create_optimized_image(self, compression: str = 'LZW') -> bool:
        """
        最適化された画像を作成

        Args:
            compression (str): 圧縮方式

        Returns:
            bool: 成功した場合True
        """
        try:
            translate_options = gdal.TranslateOptions(
                format='GTiff',
                creationOptions=[
                    f'COMPRESS={compression}',
                    'TILED=YES',
                    'BLOCKXSIZE=256',
                    'BLOCKYSIZE=256',
                    'BIGTIFF=YES'
                ]
            )

            gdal.Translate(self.output_path, self.dataset, options=translate_options)
            self.logger.info(f"最適化画像を作成: {self.output_path}")
            return True

        except Exception as e:
            self.logger.error(f"最適化画像作成エラー: {str(e)}")
            return False

    def build_pyramids(self, resampling_method: str = 'AVERAGE') -> bool:
        """
        ピラミッドレイヤーを構築

        Args:
            resampling_method (str): リサンプリング方法

        Returns:
            bool: 成功した場合True
        """
        try:
            dataset = gdal.Open(self.output_path, 1)  # 1 = 書き込みモード
            levels = self.calculate_pyramid_levels()

            dataset.BuildOverviews(resampling_method, levels)
            self.logger.info(f"ピラミッドを作成: レベル {levels}")

            dataset = None  # クローズ
            return True

        except Exception as e:
            self.logger.error(f"ピラミッド作成エラー: {str(e)}")
            return False

    def get_statistics(self) -> Dict:
        """
        画像の統計情報を取得

        Returns:
            Dict: 統計情報
        """
        stats = {}
        try:
            for band in range(1, self.image_info['bands'] + 1):
                band_data = self.dataset.GetRasterBand(band)
                min_val, max_val, mean, std_dev = band_data.GetStatistics(True, True)
                stats[f'Band_{band}'] = {
                    'min': min_val,
                    'max': max_val,
                    'mean': mean,
                    'std_dev': std_dev
                }
            return stats
        except Exception as e:
            self.logger.error(f"統計情報取得エラー: {str(e)}")
            return {}

    def process(self) -> bool:
        """
        画像処理の全工程を実行

        Returns:
            bool: 全工程が成功した場合True
        """
        self.logger.info(f"処理開始: {self.input_path}")

        if not self.load_image():
            return False

        if not self.create_optimized_image():
            return False

        if not self.build_pyramids():
            return False

        stats = self.get_statistics()
        if stats:
            self.logger.info(f"統計情報: {stats}")

        self.logger.info("処理完了")
        return True

    def __del__(self):
        """デストラクタ - リソースの解放"""
        if self.dataset is not None:
            self.dataset = None
            self.logger.info("リソースを解放しました")


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(
        description='地理空間画像のピラミッド処理を行います'
    )
    parser.add_argument(
        'input_file',
        help='入力画像ファイルのパス'
    )
    parser.add_argument(
        '--compression',
        choices=['LZW', 'DEFLATE', 'NONE'],
        default='LZW',
        help='圧縮方式の指定 (デフォルト: LZW)'
    )
    parser.add_argument(
        '--resampling',
        choices=['AVERAGE', 'NEAREST', 'BILINEAR'],
        default='AVERAGE',
        help='リサンプリング方式の指定 (デフォルト: AVERAGE)'
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    if not os.path.exists(args.input_file):
        print(f"エラー: 入力ファイルが存在しません: {args.input_file}")
        sys.exit(1)

    # 処理の実行
    processor = GeoImagePyramid(args.input_file)
    if processor.process():
        print("処理が正常に完了しました")
        sys.exit(0)
    else:
        print("処理中にエラーが発生しました")
        sys.exit(1)


if __name__ == "__main__":
    main()