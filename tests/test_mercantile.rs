
#[cfg(test)]
mod tests {
    use std::string::String;
    use mercantile::mercantile;

    #[test]
    fn test_ul() {
        let tile = mercantile::Tile::new(486,332,10);
        let lnglat = tile.ul();
        let expected = mercantile::LngLat::new(-9.140625,53.33087298301705);
        assert_eq!(lnglat,expected);

    }
    #[test]
    fn test_bounds() {
        let expected = mercantile::LngLatBbox::from(vec![-9.140625, 53.120405283106564, -8.7890625, 53.33087298301705]);
        let tile = mercantile::Tile::new(486,332,10);
        let lnglat = tile.bounds();
        assert_eq!(lnglat,expected);
    }
    #[test]
    fn test_truncate_lnglat() {
        let (lng,lat) = mercantile::truncate_lnglat(-181.0,0.0);
        assert_eq!(lng,-180.0);
        assert_eq!(lat,0.0);

    }

    #[test]
    fn test_lnglatbbox_from_vec(){
        let expected = mercantile::LngLatBbox::new(-9.140625, 53.120405283106564, -8.7890625, 53.33087298301705);
        let b = mercantile::LngLatBbox::from(vec![-9.140625, 53.120405283106564, -8.7890625, 53.33087298301705]);
        assert_eq!(b,expected);
    }

    #[test]
    fn test_neighbors() {
        let tile = mercantile::Tile::new(243,166,9);
        let tiles = tile.neighbors();
        let collection = vec![-1,0,1];
        assert_eq!(tiles.len(),8);
        for t in tiles{
            assert_eq!(t.z,tile.z);
            assert_eq!(collection.contains(&(t.x as i32 -tile.x as i32)),true);
            assert_eq!(collection.contains(&(t.y as i32 -tile.y as i32)),true);
        }
    }

    #[test]
    fn test_xy() {
        let (x,y) = mercantile::xy(-9.140625,53.33087298301705,false);
        assert_eq!(x,-1017529.7205322663);
        assert_eq!(y,7044436.52676184);
    }

    #[test]
    fn test_quadkey() {
        let tile = mercantile::Tile::new(486,332,10);
        let expected = String::from("0313102310");
        let s = tile.into_quadkey();
        assert_eq!(s,expected);
    }

    #[test]
    fn test_quadkey_to_tile(){
        let s = "0313102310";
        let tile = mercantile::Tile::from_quadkey(s).unwrap();
        let expected = mercantile::Tile::new(486,332,10);
        assert_eq!(tile,expected);
    }

    #[test]
    fn test_tiles(){
        let zoom = vec![14];
        let tiles = mercantile::Tile::from_bounds(-105.0, 39.99, -104.99, 40.0, &zoom, false);
        let expected = vec![
            mercantile::Tile::new(3413,6202,14),
            mercantile::Tile::new(3413,6203,14),
        ];
        let mut len = 0;
        for t in tiles{
            assert_eq!(expected.contains(&t),true);
            len += 1;
        }
        assert_eq!(len,2);
    }

    #[test]
    fn test_parent(){
        let tile = mercantile::Tile::new(486,332,10);
        let p = tile.parent().unwrap();
        let expected = mercantile::Tile::new(243,166,9);
        assert_eq!(p,expected);

    }

    #[test]
    fn parent_by_zoom(){
        let tile = mercantile::Tile::new(486,332,10);
        let p = tile.parent_by_zoom(8).unwrap();
        let expected = mercantile::Tile::new(121,83,8);
        assert_eq!(p,expected);
    }
    #[test]
    fn test_children(){
        let tile = mercantile::Tile::new(243,166,9);
        let p = tile.children();
        assert_eq!(p.len(),4);
        assert_eq!(p[0],mercantile::Tile::new(2*243,2*166,9+1));
        assert_eq!(p[1],mercantile::Tile::new(2*243+1,2*166,9+1));
        assert_eq!(p[2],mercantile::Tile::new(2*243+1,2*166+1,9+1));
        assert_eq!(p[3],mercantile::Tile::new(2*243,2*166+1,9+1));
    }

    #[test]
    fn test_children_by_zoom(){
        let tile = mercantile::Tile::new(243, 166, 9);
        let children = tile.children_by_zoom(11).unwrap();
        let targets = vec![
            mercantile::Tile::new(972, 664, 11),
            mercantile::Tile::new(973, 664, 11),
            mercantile::Tile::new(973, 665, 11),
            mercantile::Tile::new(972, 665, 11),
            mercantile::Tile::new(974, 664, 11),
            mercantile::Tile::new(975, 664, 11),
            mercantile::Tile::new(975, 665, 11),
            mercantile::Tile::new(974, 665, 11),
            mercantile::Tile::new(974, 666, 11),
            mercantile::Tile::new(975, 666, 11),
            mercantile::Tile::new(975, 667, 11),
            mercantile::Tile::new(974, 667, 11),
            mercantile::Tile::new(972, 666, 11),
            mercantile::Tile::new(973, 666, 11),
            mercantile::Tile::new(973, 667, 11),
            mercantile::Tile::new(972, 667, 11),
        ];
        assert_eq!(children.len(),16);
        for target in targets{
            assert_eq!(children.contains(&target),true);
        }
    }

    #[test]
    fn sorted_tiles(){
        let mut tiles = vec![
            mercantile::Tile::new(0,1,1),
            mercantile::Tile::new(0,2,5),
            mercantile::Tile::new(1,1,1)];
        let t = mercantile::Tile::sort_tiles(&mut tiles);
        let expected = vec![
            mercantile::Tile::new(0,1,1),
            mercantile::Tile::new(1,1,1),
            mercantile::Tile::new(0,2,5)];
        assert_eq!(*t,expected);
    }
    #[test]
    fn test_bounding_tile(){
        let bbox = mercantile::LngLatBbox::new(-84.72656249999999, 11.178401873711785, -5.625, 61.60639637138628);
        let tile = mercantile::Tile::bounding_tile(bbox);
        assert_eq!(tile.x,1);
        assert_eq!(tile.y,1);
        assert_eq!(tile.z,2);
    }

    // 'Verify that tiles are being removed by simplify()
    #[test]
    fn test_simplify_removal(){
        let mut tiles = vec![
            mercantile::Tile::from([1298, 3129, 13]),
            mercantile::Tile::from([649, 1564, 12]),
            mercantile::Tile::from([650, 1564, 12])
        ];
        let simplified = mercantile::Tile::simplify(&mut tiles).unwrap();
        assert_eq!(simplified.contains(&mercantile::Tile::from([1298, 3129, 13])),false);
        assert_eq!(simplified.contains(&mercantile::Tile::from([649, 1564, 12])),true);
        assert_eq!(simplified.contains(&mercantile::Tile::from([650, 1564, 12])),true);
    }
    #[test]
    fn test_simplify() {
        let tile = mercantile::Tile::new(243, 166, 9);
        let mut children = tile.children_by_zoom(12).unwrap();
        let len = children.len();
        assert_eq!(len,64);
        children.remove(len-1);
        children.remove(len-2);
        children.remove(len-3);
        children.push(children[0]);
    
        let targets = vec![
            mercantile::Tile::from([487, 332, 10]),
            mercantile::Tile::from([486, 332, 10]),
            mercantile::Tile::from([487, 333, 10]),
            mercantile::Tile::from([973, 667, 11]),
            mercantile::Tile::from([973, 666, 11]),
            mercantile::Tile::from([972, 666, 11]),
            mercantile::Tile::from([1944, 1334, 12]),
        ];
        let tiles = mercantile::Tile::simplify(&mut children).unwrap();
    
        for t in targets {
            assert_eq!(tiles.contains(&t), true);
        }
    }
}
