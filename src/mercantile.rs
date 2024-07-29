use super::errors::{InvalidZoomError, MercantileError};
use geojson::{Feature, GeoJson, Geometry, Value};
use std::collections::{HashMap, HashSet};
use std::f64::consts::{E, PI};

// 地球赤道半径
const EQUATORIAL_RADIUS: f64 = 6378137.0;
// 地球赤道周长
const EQUATORIAL_CIRCUMFERENCE: f64 = 2.0 * PI * EQUATORIAL_RADIUS;

const EPSILON: f64 = 1e-14;

const LL_EPSILON: f64 = 1e-11;

const NEG_INFINITY: f64 = f64::NEG_INFINITY;

const INFINITY: f64 = f64::INFINITY;

///
/// tile x y z
///
#[derive(Debug, Default, PartialEq, Eq, Hash, Copy)]
pub struct Tile {
    x: u32,
    y: u32,
    z: Zoom,
}

type Zoom = u8;

impl Tile {
    ///
    /// `new` 构造函数
    ///
    pub fn new(x: u32, y: u32, z: Zoom) -> Self {
        Self { x, y, z }
    }

    pub fn x(&self) -> u32 {
        self.x
    }
    pub fn y(&self) -> u32 {
        self.y
    }
    pub fn z(&self) -> Zoom {
        self.z
    }

    ///
    /// `ul` return upper left longitude and latitude
    ///
    pub fn ul(&self) -> LngLat {
        let z2: u32 = 1 << self.z;
        let lng = (self.x as f64 / z2 as f64) * 360.0 - 180.0;
        let lat = rad2deg(((PI * (1.0 - 2.0 * (self.y as f64 / z2 as f64))).sinh()).atan());
        LngLat::new(lng, lat)
    }

    ///
    /// `bounds` return bound box in longitude and latitude
    ///
    pub fn bounds(&self) -> LngLatBbox {
        let z2: u32 = 1 << self.z;
        let west = self.x as f64 / z2 as f64 * 360.0 - 180.0;
        let south = rad2deg(((PI * (1.0 - 2.0 * ((self.y + 1) as f64 / z2 as f64))).sinh()).atan());
        let east = (self.x + 1) as f64 / z2 as f64 * 360.0 - 180.0;
        let north = rad2deg(((PI * (1.0 - 2.0 * (self.y as f64 / z2 as f64))).sinh()).atan());
        LngLatBbox::new(west, south, east, north)
    }

    ///
    /// `neighbors` 返回瓦片相连的瓦片，同一层级的瓦片
    ///
    pub fn neighbors(&self) -> Vec<Self> {
        let tmp: [i32; 3] = [-1, 0, 1];
        let (x, y, z) = (self.x as i32, self.y as i32, self.z);

        tmp.iter()
            .flat_map(|&i| {
                tmp.iter().map(move |&j| (i, j))
            })
            .filter(|&(i, j)| {
                !(i == 0 && j == 0)
                    && x + i >= 0
                    && y + j >= 0
                    && x + i < (1 << z)
                    && y + j < (1 << z)
            })
            .map(move |(i, j)| Tile::new((x + i) as u32, (y + j) as u32, z))
            .filter(Self::valid)
            .collect()
    }



    ///
    /// `xy_bounds` Get the web mercator bounding box of a tile
    ///
    pub fn xy_bounds(&self) -> Bbox {
        let tile_size = EQUATORIAL_CIRCUMFERENCE / (1 << self.z) as f64;
        let left = self.x as f64 * tile_size - EQUATORIAL_CIRCUMFERENCE / 2.0;
        let right = left + tile_size;
        let top = EQUATORIAL_CIRCUMFERENCE / 2.0 - self.y as f64 * tile_size;
        let bottom = top - tile_size;
        Bbox::new(left, bottom, right, top)
    }

    ///
    /// `tile` 按照经纬度和zoom构建瓦片
    ///
    pub fn tile(lng: f64, lat: f64, zoom: Zoom, truncate: bool) -> Self {
        let (x, y) = _xy(lng, lat, truncate);
        let z2 = 1 << zoom;

        let xtile = match x {
            x if x <= 0.0 => 0,
            x if x >= 1.0 => z2 - 1,
            _ => ((x + EPSILON) * (z2 as f64)).floor() as u32,
        };

        let ytile = match y {
            y if y <= 0.0 => 0,
            y if y >= 1.0 => z2 - 1,
            _ => ((y + EPSILON) * (z2 as f64)).floor() as u32,
        };

        Self::new(xtile, ytile, zoom)
    }

    ///
    /// `quadkey`
    ///
    pub fn quadkey(&self) -> String {
        let mut qk = String::with_capacity(self.z as usize);
        (1..=self.z).rev().for_each(|z| {
            let mask = 1 << (z - 1);
            let digit = match (self.x & mask != 0, self.y & mask != 0) {
                (false, false) => '0',
                (true, false) => '1',
                (false, true) => '2',
                (true, true) => '3',
            };
            qk.push(digit);
        });
        qk
    }

    pub fn quadkey_to_tile(qk: &str) -> Result<Self, MercantileError> {
        if qk.is_empty() {
            return Ok(Self::new(0, 0, 0));
        }

        let (xtile, ytile, _) = 
            qk.chars()
            .rev()
            .enumerate()
            .try_fold((0, 0, 0), |(xtile, ytile, zoom), (i, digit)| {
                let mask = 1 << i;
                match digit {
                    '0' => Ok((xtile, ytile, zoom)),
                    '1' => Ok((xtile | mask, ytile, zoom)),
                    '2' => Ok((xtile, ytile | mask, zoom)),
                    '3' => Ok((xtile | mask, ytile | mask, zoom)),
                    _ => Err(MercantileError::QuadKeyError),
                }
        })?;

        let zoom = qk.len() as Zoom;

        Ok(Self::new(xtile, ytile, zoom))
    }


    ///
    /// `tiles` 按照四至及层级生成Tile
    ///
    pub fn tiles<'a>(
        mut west: f64,
        mut south: f64,
        mut east: f64,
        mut north: f64,
        zooms: &'a Vec<Zoom>,
        truncate: bool,
    ) -> impl Iterator<Item = Self> + 'a {
        if truncate {
            (west, south) = truncate_lnglat(west, south);
            (east, north) = truncate_lnglat(east, north);
        }
        let mut bboxes = vec![];
        if west > east {
            let bbox_west = (-180.0, south, east, north);
            let bbox_east = (west, south, 180.0, north);
            bboxes.push(bbox_west);
            bboxes.push(bbox_east);
        } else {
            bboxes.push((west, south, east, north));
        }
        bboxes
            .into_iter()
            .flat_map(move |(mut w, mut s, mut e, mut n)| {
                w = w.max(-180.0);
                s = s.max(-85.051129);
                e = e.min(180.0);
                n = n.min(85.051129);

                zooms.iter().flat_map(move |&z| {
                    let ul_tile = Self::tile(w, n, z, false);
                    let lr_tile = Self::tile(e - LL_EPSILON, s + LL_EPSILON, z, false);

                    (ul_tile.x..=lr_tile.x).flat_map(move |i| {
                        (ul_tile.y..=lr_tile.y).map(move |j| Self::new(i, j, z))
                    })
                })
            })
    }


    ///
    /// `parent` 获取当前瓦片的父瓦片
    ///
    pub fn parent(&self) -> Result<Self, MercantileError> {
        if self.z == 0 {
            return Err(MercantileError::ParentTileError);
        }
        Ok(Self::new(self.x >> 1, self.y >> 1, self.z - 1))
    }

    ///
    /// `parent_by_zoom` 获取特定层级的父瓦片
    ///
    pub fn parent_by_zoom(&self, zoom: Zoom) -> Result<Self, MercantileError> {
        if self.z == 0 {
            return Err(MercantileError::ParentTileError);
        }
        if zoom >= self.z {
            return Err(MercantileError::InvalidZoomError(
                InvalidZoomError::ZoomIsTooLarge(zoom),
            ));
        }

        (zoom..self.z)
            .rev()
            .try_fold(Self::new(self.x, self.y, self.z), |t, _| t.parent())
    }


    ///
    /// `children` 获取四个子瓦片
    ///
    pub fn children(&self) -> Vec<Self> {
        vec![
            Self::new(self.x * 2, self.y * 2, self.z + 1),
            Self::new(self.x * 2 + 1, self.y * 2, self.z + 1),
            Self::new(self.x * 2 + 1, self.y * 2 + 1, self.z + 1),
            Self::new(self.x * 2, self.y * 2 + 1, self.z + 1),
        ]
    }

    ///
    /// `children_by_zoom` 获取特定层级子瓦片
    //
    ///
    pub fn children_by_zoom(&self, zoom: Zoom) -> Result<Vec<Self>, MercantileError> {
        if self.z >= zoom {
            return Err(MercantileError::InvalidZoomError(
                InvalidZoomError::ZoomIsTooSmall(zoom),
            ));
        }

        let mut tiles = vec![Self::new(self.x, self.y, self.z)];

        while tiles.first().map_or(false, |t| t.z < zoom) {
            let mut new_tiles = Vec::with_capacity(tiles.len() * 4);
            for t in tiles.drain(..) {
                new_tiles.extend_from_slice(&[
                    Self::new(t.x * 2, t.y * 2, t.z + 1),
                    Self::new(t.x * 2 + 1, t.y * 2, t.z + 1),
                    Self::new(t.x * 2 + 1, t.y * 2 + 1, t.z + 1),
                    Self::new(t.x * 2, t.y * 2 + 1, t.z + 1),
                ]);
            }
            tiles = new_tiles;
        }

        Ok(tiles)
    }


    ///
    /// `simplify` Reduces the size of the tileset as much as possible by merging leaves into parents.
    ///
    // pub fn simplify(tiles: &mut Vec<Self>) -> Result<HashSet<Self>, MercantileError> {
    //     let mut root_set = HashSet::new();
    //     for t in Self::sorted_tiles(tiles) {
    //         let mut is_new_tile = true;
    //         let vec: Vec<Zoom> = (0..t.z).collect();
    //         let parent_tiles: Vec<Result<Self, MercantileError>> =
    //             vec.iter().map(|&x| t.parent_by_zoom(x)).collect();
    //         for super_tile in parent_tiles {
    //             if root_set.contains(&super_tile?) {
    //                 is_new_tile = false;
    //                 continue;
    //             }
    //         }
    //         if is_new_tile {
    //             root_set.insert(t.clone());
    //         }
    //     }
    //     let mut is_merging = true;
    //     while is_merging {
    //         (root_set, is_merging) = Self::merge(root_set)?
    //     }
    //     Ok(root_set)
    // }

    pub fn simplify(tiles: &mut Vec<Self>) -> Result<HashSet<Self>, MercantileError> {
        let mut root_set = HashSet::new();

        // 获取排序后的瓦片
        let sorted_tiles = Self::sorted_tiles(tiles);

        // 处理每个瓦片，决定是否加入到 root_set 中
        for t in sorted_tiles {
            let parent_tiles: Result<Vec<_>, _> = (0..t.z)
                .map(|z| t.parent_by_zoom(z))
                .collect();

            let parent_tiles = parent_tiles?;

            if parent_tiles.iter().all(|parent| !root_set.contains(parent)) {
                root_set.insert(t.clone());
            }
        }

        // 合并瓦片直到没有更多合并
        let mut is_merging = true;
        while is_merging {
            let (merged_set, merging) = Self::merge(root_set)?;
            root_set = merged_set;
            is_merging = merging;
        }

        Ok(root_set)
    }


    ///
    /// `merge` Checks to see if there are 4 tiles in merge_set which can be merged.
    //         If there are, this merges them.
    //         This returns a list of tiles, as well as a boolean indicating if any were merged.
    //         By repeatedly applying merge, a tileset can be simplified.
    ///
    // pub fn merge(merge_set: HashSet<Self>) -> Result<(HashSet<Self>, bool), MercantileError> {
    //     let mut upwards_merge: HashMap<Self, HashSet<Tile>> = HashMap::new();
    //     for tile in merge_set {
    //         let mut tmp = HashSet::new();
    //         let tile_parent = tile.parent()?;
    //         if !upwards_merge.contains_key(&tile_parent) {
    //             upwards_merge.insert(tile_parent, HashSet::new());
    //         }
    //         tmp.insert(tile);
    //         let merged = upwards_merge[&tile_parent].union(&tmp).copied().collect();
    //         upwards_merge.insert(tile_parent, merged);
    //     }
    //     let mut current_tileset = HashSet::new();
    //     let mut changed = false;
    //     for (supertile, children) in upwards_merge {
    //         if children.len() == 4 {
    //             current_tileset.insert(supertile);
    //             changed = true;
    //         } else {
    //             for c in children {
    //                 current_tileset.insert(c);
    //             }
    //         }
    //     }
    //     Ok((current_tileset, changed))
    // }
    pub fn merge(merge_set: HashSet<Self>) -> Result<(HashSet<Self>, bool), MercantileError> {
        let mut upwards_merge: HashMap<Self, HashSet<Self>> = HashMap::new();

        // 将每个瓦片及其父瓦片收集到上下合并映射中
        for tile in merge_set {
            let tile_parent = tile.parent()?;
            upwards_merge
                .entry(tile_parent)
                .or_insert_with(HashSet::new)
                .insert(tile);
        }

        // 根据父瓦片的子瓦片数量构建新的瓦片集合
        let (current_tileset, changed) = upwards_merge.into_iter().fold(
            (HashSet::new(), false),
            |(mut current_tileset, mut changed), (supertile, children)| {
                if children.len() == 4 {
                    current_tileset.insert(supertile);
                    changed = true;
                } else {
                    current_tileset.extend(children);
                }
                (current_tileset, changed)
            },
        );

        Ok((current_tileset, changed))
    }


    ///
    /// `bounding_tile` Get the smallest tile to cover a LngLatBbox 经纬度范围
    ///
    pub fn bounding_tile(bbox: LngLatBbox) -> Self {
        let min = Tile::point2tile(bbox.west, bbox.south, 32);
        let max = Tile::point2tile(bbox.east, bbox.north, 32);
        let bbox = Bbox::new(min.x as f64, min.y as f64, max.x as f64, max.y as f64);
        let z = bbox.get_bbox_zoom();
        if z == 0 {
            return Tile::new(0, 0, 0);
        }
        Tile::new(
            (bbox.left as u32) >> (32 - z),
            (bbox.bottom as u32) >> (32 - z),
            z,
        )
    }

    ///
    /// `tile2lnglatbbox` tile转为lnglatbbox 经纬度范围
    ///
    pub fn tile2lnglatbbox(&self) -> LngLatBbox {
        let e = Self::tile2lon(self.x + 1, self.z);
        let w = Self::tile2lon(self.x, self.z);
        let s = Self::tile2lat(self.y + 1, self.z);
        let n = Self::tile2lat(self.y, self.z);
        LngLatBbox::new(e, w, s, n)
    }

    ///
    ///  `tile2geojson` tile转geojson
    ///
    pub fn tile2geojson(&self) -> String {
        let bbox = Self::tile2lnglatbbox(&self);
        let geometry = Geometry::new(Value::Polygon(vec![
            vec![vec![bbox.west, bbox.north]],
            vec![vec![bbox.west, bbox.south]],
            vec![vec![bbox.east, bbox.south]],
            vec![vec![bbox.east, bbox.north]],
            vec![vec![bbox.west, bbox.north]],
        ]));
        let bbox = self.tile2lnglatbbox().to_vec();
        let poly = GeoJson::Feature(Feature {
            bbox: Some(bbox),
            geometry: Some(geometry),
            id: None,
            // See the next section about Feature properties
            properties: None,
            foreign_members: None,
        });
        poly.to_string()
    }

    ///
    /// `tiles_equal` 判断两个tile是否相等
    ///
    pub fn tiles_equal(&self, tile: &Tile) -> bool {
        self.x == tile.x && self.y == tile.y && self.z == tile.z
    }

    ///
    /// `tile2lon` 按照tile的x和z转为经度
    ///
    pub fn tile2lon(x: u32, z: Zoom) -> f64 {
        x as f64 / 2_i32.pow(z as u32) as f64 * 360.0 - 180.0
    }

    ///
    /// `tile2lat` 按照tile的y和z转为纬度
    ///
    pub fn tile2lat(y: u32, z: Zoom) -> f64 {
        let n = PI - (2.0 * PI * (y as f64)) / (2_i32.pow(z as u32) as f64);
        180.0 / PI * (0.5 * (n.exp() - (-n).exp())).atan()
    }

    ///
    /// `point2tile` Get the tile location for a point at a zoom level
    ///
    pub fn point2tile(lon: f64, lat: f64, z: Zoom) -> Tile {
        let sin = (lat * PI / 180.0).sin();
        let z2 = 2_i64.pow(z as u32) as f64;
        let mut x = z2 * (lon / 360.0 + 0.5);
        let y = z2 * (0.5 - 0.25 * ((1.0 + sin) / (1.0 - sin)).log(E) / PI);

        x = x % z2;
        if x < 0.0 {
            x = x + z2;
        }
        // 向下取整
        Tile::new(x as u32, y as u32, z)
    }

    ///
    /// `valid` 瓦片是否合理
    ///
    pub fn valid(t: &Self) -> bool {
        if t.x <= (1 << t.z) - 1 && t.y <= (1 << t.z) - 1 {
            return true;
        }
        false
    }

    ///
    /// `sorted_tiles` 对tiles进行排序,返回一个新的结果
    ///
    // pub fn sorted_tiles(tiles: &Vec<Self>) -> Vec<Self> {
    //     let mut tiles = tiles.clone();
    //     let len = tiles.len();
    //     for i in 0..len {
    //         for j in 0..len - i - 1 {
    //             if tiles[j].z > tiles[j + 1].z {
    //                 // 可以直接使用swap
    //                 tiles.swap(j, j + 1);
    //             }
    //         }
    //     }
    //     tiles
    // }
    pub fn sorted_tiles(tiles: &mut Vec<Self>) -> &Vec<Self> {
        // let mut tiles = tiles.clone();
        tiles.sort_by(|a, b| a.z.cmp(&b.z));
        tiles
    }

}

impl From<&Vec<u32>> for Tile {
    fn from(value: &Vec<u32>) -> Self {
        assert_eq!(value.len(), 3, "Invalid vector length");
        Self::new(value[0], value[1], value[2] as u8)
    }
}

impl From<&[u32;3]> for Tile {
    fn from(value:&[u32;3])->Self{
         Self::new(value[0], value[1], value[2] as u8)
    }
}

impl Clone for Tile {
    fn clone(&self) -> Self {
        Tile::new(self.x, self.y, self.z)
    }
}

///
/// lng lat 经纬度位置
///
#[derive(Debug, Default)]
pub struct LngLat {
    lng: f64,
    lat: f64,
}

impl LngLat {
    pub fn new(lng: f64, lat: f64) -> LngLat {
        LngLat { lng, lat }
    }

    pub fn lng(&self) -> f64 {
        self.lng
    }
    pub fn lat(&self) -> f64 {
        self.lat
    }
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.lng, self.lat]
    }
    pub fn to_array(&self) -> [f64; 2] {
        [self.lng, self.lat]
    }
}

impl From<&Vec<f64>> for LngLat{
    fn from(value: &Vec<f64>) -> Self {
        assert_eq!(value.len(), 2, "Invalid vector length {}",value.len());
        Self::new(value[0],value[1])
    }
}

impl From<&[f64;2]> for LngLat {
    fn from(value: &[f64; 2]) -> Self {
        Self::new(value[0],value[1])
    }
}

impl PartialEq for LngLat {
    fn eq(&self, other: &Self) -> bool {
        if self.lng == other.lng && self.lat == other.lat {
            return true;
        }
        false
    }
}
///
///  lnglat bounding box unit degree
///
#[derive(Debug, Default)]
pub struct LngLatBbox {
    west: f64,
    south: f64,
    east: f64,
    north: f64,
}

impl LngLatBbox {
    pub fn new(west: f64, south: f64, east: f64, north: f64) -> LngLatBbox {
        LngLatBbox {
            west,
            south,
            east,
            north,
        }
    }

    ///
    /// `to_vec` 将lnglatbbox的转为[west,south,east,north]的vec
    ///
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.west, self.south, self.east, self.north]
    }

    ///
    /// `to_array` 将lnglatbbox的转为[west,south,east,north]的array
    ///
    pub fn to_array(&self) -> [f64; 4] {
        [self.west, self.south, self.east, self.north]
    }

    ///
    /// `from_vec` 从[west,south,east,north]的vec生成lnglatbbox
    ///
    pub fn from_vec(v: &Vec<f64>) -> Result<LngLatBbox, MercantileError> {
        let len = v.len() as u32;
        if len != 4 {
            return Err(MercantileError::InvalidVecLengthError {
                real_length: len,
                valid_length: 4,
            });
        }
        Ok(LngLatBbox::new(v[0], v[1], v[2], v[3]))
    }

    ///
    /// `from_array` 数组[west,south,east,north]直接生成LngLatBbox
    ///
    pub fn from_array(a: [f64; 4]) -> LngLatBbox {
        LngLatBbox::new(a[0], a[1], a[2], a[3])
    }
}

///
/// 判断两个LngLatBbox是否相等
///
impl PartialEq for LngLatBbox {
    fn eq(&self, other: &Self) -> bool {
        if self.west == other.west
            && self.east == other.east
            && self.south == other.south
            && self.north == self.north
        {
            return true;
        }
        false
    }
}

///
/// mercator bounding box unit meter
///
pub struct Bbox {
    left: f64,
    bottom: f64,
    right: f64,
    top: f64,
}

impl Bbox {
    pub fn new(left: f64, bottom: f64, right: f64, top: f64) -> Bbox {
        Bbox {
            left,
            bottom,
            right,
            top,
        }
    }

    pub fn from_vec(v: &Vec<f64>) -> Result<Bbox, MercantileError> {
        let len = v.len() as u32;
        if len != 4 {
            return Err(MercantileError::InvalidVecLengthError {
                valid_length: 4,
                real_length: len,
            });
        }
        Ok(Bbox::new(v[0], v[1], v[2], v[3]))
    }
    pub fn from_array(a: [f64; 4]) -> Bbox {
        Bbox::new(a[0], a[1], a[2], a[3])
    }
    ///
    /// `get_bbox_zoom` 按照bbox范围获取最小层级
    ///
    pub fn get_bbox_zoom(&self) -> Zoom {
        let max_zoom = 28;
        for z in 0..max_zoom {
            let mask = 1 << (32 - (z + 1)); // 2^a
            if (self.left as u32 & mask) != (self.right as u32 & mask)
                || (self.bottom as u32 & mask) != (self.top as u32 & mask)
            {
                return z;
            }
        }
        max_zoom
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
/// impl
///////////////////////////////////////////////////////////////////////////////////////////

///
/// `rad2deg`弧度转度
///
/// # Arguments
///
/// * `rad` - 弧度值
///
pub fn rad2deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

///
/// `deg2rad` 度转弧度
///
pub fn deg2rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

///
/// `truncate_lnglat`防止边界溢出
///
pub fn truncate_lnglat(mut lng: f64, mut lat: f64) -> (f64, f64) {
    if lng > 180.0 {
        lng = 180.0;
    } else if lng < -180.0 {
        lng = -180.0;
    }
    if lat > 90.0 {
        lat = 90.0;
    } else if lat < -90.0 {
        lat = -90.0;
    }
    (lng, lat)
}

///
/// `xy` 经纬度转为mercator x y
///
pub fn xy(mut lng: f64, mut lat: f64, truncate: bool) -> (f64, f64) {
    if truncate {
        (lng, lat) = truncate_lnglat(lng, lat)
    }
    let x = EQUATORIAL_RADIUS * deg2rad(lng);
    let y;

    if lat <= -90.0 {
        y = NEG_INFINITY;
    } else if lat >= 90.0 {
        y = INFINITY;
    } else {
        y = EQUATORIAL_RADIUS * ((PI * 0.25) + (0.5 * deg2rad(lat))).tan().log(E);
    }
    (x, y)
}

///
/// `lnglat` mercantor x y 转为经纬度
///
pub fn lnglat(x: f64, y: f64, truncate: bool) -> (f64, f64) {
    let lng = x * (180.0 / PI) / EQUATORIAL_RADIUS;
    let lat = ((PI * 0.5) - 2.0 * (-y / EQUATORIAL_RADIUS).exp().atan()) * (180.0 / PI);
    if truncate {
        return truncate_lnglat(lng, lat);
    }
    (lng, lat)
}

fn _xy(mut lng: f64, mut lat: f64, truncate: bool) -> (f64, f64) {
    if truncate {
        (lng, lat) = truncate_lnglat(lng, lat);
    }
    let x: f64 = lng / 360.0 + 0.5;
    let sinlat = deg2rad(lat).sin();
    let y = 0.5 - 0.25 * ((1.0 + sinlat) / (1.0 - sinlat)).log(E) / PI;
    (x, y)
}

///
/// `rshift` 将val 除以 2^n，丢弃任何小数结果
///
pub fn rshift(val: usize, n: usize) -> usize {
    (val % 0x100000000) >> n
}
