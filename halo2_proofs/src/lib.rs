//! # halo2_proofs
#![allow(clippy::suspicious_arithmetic_impl)]
#![allow(warnings)] 
pub mod arithmetic;
pub use pairing;
pub mod circuit;
pub use halo2curves;
mod multicore;
pub mod plonk;
/// Gpu fft
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub mod gpu;
/// Gpu worker thread
pub mod worker {
    pub use super::multicore::*;
}
pub mod poly;
pub mod transcript;

pub mod dev;
mod helpers;
pub use helpers::SerdeFormat;

pub use blstrs;

#[cfg(feature = "gpu")]
#[macro_use]
extern crate lazy_static;