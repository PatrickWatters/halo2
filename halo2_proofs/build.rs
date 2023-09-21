fn main() {
    cfg_if_nightly();
    gpu_kernel();
}

#[rustversion::nightly]
fn cfg_if_nightly() {
    println!("cargo:rustc-cfg=nightly");
}

#[rustversion::not(nightly)]
fn cfg_if_nightly() {}

/// The build script is used to generate the CUDA kernel and OpenCL source at compile-time, if the
/// `cuda` and/or `opencl` feature is enabled.
#[cfg(any(feature = "cuda", feature = "opencl"))]
fn gpu_kernel() {
    use ec_gpu_gen::SourceBuilder;

    #[cfg(feature = "bls")]
    use blstrs::{Fp, Fp2, G1Affine, G2Affine, Scalar};
    #[cfg(feature = "bls")]
    let source_builder = SourceBuilder::new()
        .add_fft::<Scalar>()
        .add_multiexp::<G1Affine, Fp>()
        .add_multiexp::<G2Affine, Fp2>();

    #[cfg(feature = "bn254")]
    use halo2curves::bn256::Fr;
    use halo2curves::bn256::curve::G1Affine;
    use halo2curves::bn256::curve::G2Affine;
    use halo2curves::bn256::Fq;
    use halo2curves::bn256::Fq2;
    //use halo2curves::bn256::{Fr, Fq, G1Affine, G2Affine};
    #[cfg(feature = "bn254")]
    let source_builder = SourceBuilder::new()
        .add_fft::<Fr>()
        .add_multiexp::<G1Affine, Fq>()
        .add_multiexp::<G2Affine, Fq2>();
    ec_gpu_gen::generate(&source_builder);
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn gpu_kernel() {}
