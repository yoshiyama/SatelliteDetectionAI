import json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import shapely.geometry
from shapely.geometry import shape, box
from pathlib import Path
import os


def create_tiled_geotiff_masks(geojson_path, output_dir,
                               tile_size=512,
                               overlap=64,
                               resolution=0.0001):
    """
    GeoJSONから重複を持つタイル状の二値マスクGeoTIFFを作成
    """
    try:
        # 出力ディレクトリの作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # GeoJSONの読み込み
        with open(geojson_path, 'r', encoding='utf-8-sig') as f:
            geojson_data = json.load(f)

        # 全ての地物をShapelyジオメトリに変換
        geometries = []
        for feature in geojson_data['features']:
            geom = shape(feature['geometry'])
            geometries.append(geom)

        all_geoms = shapely.geometry.MultiPolygon(geometries)
        bounds = all_geoms.bounds  # (minx, miny, maxx, maxy)

        # 全体の範囲とサイズを計算
        width = int((bounds[2] - bounds[0]) / resolution)
        height = int((bounds[3] - bounds[1]) / resolution)

        # タイルの分割数を計算
        effective_tile_size = tile_size - 2 * overlap
        n_tiles_x = int(np.ceil(width / effective_tile_size))
        n_tiles_y = int(np.ceil(height / effective_tile_size))

        print(f"全体サイズ: {width}x{height} ピクセル")
        print(f"タイル分割数: {n_tiles_x}x{n_tiles_y}")

        # タイルごとに処理
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # タイルの範囲を計算（ピクセル座標）
                x_start = j * effective_tile_size - overlap
                y_start = i * effective_tile_size - overlap
                x_end = x_start + tile_size
                y_end = y_start + tile_size

                # 境界チェックと調整
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(width, x_end)
                y_end = min(height, y_end)

                # 現在のタイルサイズ
                current_width = x_end - x_start
                current_height = y_end - y_start

                # タイルが有効なサイズを持つか確認
                if current_width <= 0 or current_height <= 0:
                    continue

                # 地理座標に変換
                geo_x_start = bounds[0] + x_start * resolution
                geo_y_start = bounds[1] + y_start * resolution
                geo_x_end = bounds[0] + x_end * resolution
                geo_y_end = bounds[1] + y_end * resolution

                # タイルの変換行列
                tile_transform = from_bounds(
                    geo_x_start, geo_y_start,
                    geo_x_end, geo_y_end,
                    current_width, current_height
                )

                # タイル領域のジオメトリ
                tile_bounds = box(geo_x_start, geo_y_start, geo_x_end, geo_y_end)

                # タイル内の地すべり領域を抽出
                tile_geometries = [
                    geometry for geometry in geometries
                    if geometry.intersects(tile_bounds)
                ]

                # 空のマスクを作成
                mask = np.zeros((current_height, current_width), dtype=np.uint8)

                # 地すべり領域が存在する場合のみラスタ化
                if tile_geometries:
                    mask = rasterize(
                        [(geometry, 1) for geometry in tile_geometries],
                        out_shape=(current_height, current_width),
                        transform=tile_transform,
                        fill=0,
                        dtype=np.uint8
                    )

                # タイルを保存
                output_path = output_dir / f'tile_{i:03d}_{j:03d}.tif'

                with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=current_height,
                        width=current_width,
                        count=1,
                        dtype=np.uint8,
                        crs='EPSG:4326',
                        transform=tile_transform,
                        compress='lzw'
                ) as dst:
                    dst.write(mask, 1)

                    # メタデータの追加
                    dst.update_tags(
                        TILE_ROW=i,
                        TILE_COL=j,
                        OVERLAP=overlap,
                        RESOLUTION=resolution
                    )

                print(f'タイル作成: {output_path}')

        print(f'処理が完了しました。出力ディレクトリ: {output_dir}')

    except Exception as e:
        print(f'エラーが発生しました: {str(e)}')
        raise  # エラーの詳細を表示


# 実行
geojson_path = '/Users/survey/Desktop/Noto/noto_anamizu_20240118/hokai_all_20240118_anamizu.geojson'
output_dir = '/Users/survey/Desktop/Noto/landslide_tiles'

create_tiled_geotiff_masks(
    geojson_path=geojson_path,
    output_dir=output_dir,
    tile_size=512,
    overlap=64,
    resolution=0.0001
)