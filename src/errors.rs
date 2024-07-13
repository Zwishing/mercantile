use thiserror::Error;
use crate::errors;

///////////////////////////////////////////////////////////////////////////////////////////
/// MercantileError
///////////////////////////////////////////////////////////////////////////////////////////
#[derive(Error,Debug)]
pub enum MercantileError{
    #[error("quadkey number is not bigger than 3")]
    QuadKeyError,
    #[error(transparent)]
    InvalidZoomError(#[from] errors::InvalidZoomError),

    #[error("Vec length must be {valid_length} and now is {real_length}")]
    InvalidVecLengthError{
        real_length:u32,
        valid_length:u32,
    },
    #[error("The current Tile has no parent Tile")]
    ParentTileError
}

#[derive(Error,Debug)]
pub enum InvalidZoomError{
    #[error("The input zoom {0} is smaller than Tile zoom")]
    ZoomIsTooSmall(u8),
    #[error("The input zoom {0} is bigger than Tile zoom")]
    ZoomIsTooLarge(u8)
}