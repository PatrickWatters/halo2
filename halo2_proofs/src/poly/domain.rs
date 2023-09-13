//! Contains utilities for performing polynomial arithmetic over an evaluation
//! domain that is of a suitable size for the application.

#[cfg(feature = "gpu")]
use crate::arithmetic::best_fft_gpu;

#[cfg(feature = "cpu")]
use crate::arithmetic::best_fft_cpu;

use crate::{
    arithmetic::{parallelize, FftGroup},
    plonk::Assigned,
};

use super::{Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation};
use ff::WithSmallOrderMulGroup;
use group::{
    ff::{BatchInvert, Field, PrimeField},
    Group,
};

use group::Group as CryptGroup;
use test_log;

use std::marker::PhantomData;
#[cfg(feature = "gpu")]
use crate::gpu;
use log::{info, warn};
use std::time::Duration;


#[derive(serde::Serialize)]
struct FFTLoggingInfo {
    logn: String,
    fft_duration: String,
    fft_type: String,
    gpu_transfer:String, 
}




/// This structure contains precomputed constants and other details needed for
/// performing operations on an evaluation domain of size $2^k$ and an extended
/// domain of size $2^{k} * j$ with $j \neq 0$.
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: Field> {
    n: u64,
    k: u32,
    extended_k: u32,
    omega: F,
    omega_inv: F,
    extended_omega: F,
    extended_omega_inv: F,
    g_coset: F,
    g_coset_inv: F,
    quotient_poly_degree: u64,
    ifft_divisor: F,
    extended_ifft_divisor: F,
    t_evaluations: Vec<F>,
    barycentric_weight: F,
}

impl<F: WithSmallOrderMulGroup<3>> EvaluationDomain<F> {
    /// This constructs a new evaluation domain object based on the provided
    /// values $j, k$.
    pub fn new(j: u32, k: u32) -> Self {
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let quotient_poly_degree = (j - 1) as u64;

        // n = 2^k
        let n = 1u64 << k;

        // We need to work within an extended domain, not params.k but params.k + i
        // for some integer i such that 2^(params.k + i) is sufficiently large to
        // describe the quotient polynomial.
        let mut extended_k = k;
        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
        }

        let mut extended_omega = F::ROOT_OF_UNITY;

        // Get extended_omega, the 2^{extended_k}'th root of unity
        // The loop computes extended_omega = omega^{2 ^ (S - extended_k)}
        // Notice that extended_omega ^ {2 ^ extended_k} = omega ^ {2^S} = 1.
        for _ in extended_k..F::S {
            extended_omega = extended_omega.square();
        }
        let extended_omega = extended_omega;
        let mut extended_omega_inv = extended_omega; // Inversion computed later

        // Get omega, the 2^{k}'th root of unity (i.e. n'th root of unity)
        // The loop computes omega = extended_omega ^ {2 ^ (extended_k - k)}
        //           = (omega^{2 ^ (S - extended_k)})  ^ {2 ^ (extended_k - k)}
        //           = omega ^ {2 ^ (S - k)}.
        // Notice that omega ^ {2^k} = omega ^ {2^S} = 1.
        let mut omega = extended_omega;
        for _ in k..extended_k {
            omega = omega.square();
        }
        let omega = omega;
        let mut omega_inv = omega; // Inversion computed later

        // We use zeta here because we know it generates a coset, and it's available
        // already.
        // The coset evaluation domain is:
        // zeta {1, extended_omega, extended_omega^2, ..., extended_omega^{(2^extended_k) - 1}}
        let g_coset = F::ZETA;
        let g_coset_inv = g_coset.square();

        let mut t_evaluations = Vec::with_capacity(1 << (extended_k - k));
        {
            // Compute the evaluations of t(X) = X^n - 1 in the coset evaluation domain.
            // We don't have to compute all of them, because it will repeat.
            let orig = F::ZETA.pow_vartime(&[n as u64, 0, 0, 0]);
            let step = extended_omega.pow_vartime(&[n as u64, 0, 0, 0]);
            let mut cur = orig;
            loop {
                t_evaluations.push(cur);
                cur *= &step;
                if cur == orig {
                    break;
                }
            }
            assert_eq!(t_evaluations.len(), 1 << (extended_k - k));

            // Subtract 1 from each to give us t_evaluations[i] = t(zeta * extended_omega^i)
            for coeff in &mut t_evaluations {
                *coeff -= &F::ONE;
            }

            // Invert, because we're dividing by this polynomial.
            // We invert in a batch, below.
        }

        let mut ifft_divisor = F::from(1 << k); // Inversion computed later
        let mut extended_ifft_divisor = F::from(1 << extended_k); // Inversion computed later

        // The barycentric weight of 1 over the evaluation domain
        // 1 / \prod_{i != 0} (1 - omega^i)
        let mut barycentric_weight = F::from(n); // Inversion computed later

        // Compute batch inversion
        t_evaluations
            .iter_mut()
            .chain(Some(&mut ifft_divisor))
            .chain(Some(&mut extended_ifft_divisor))
            .chain(Some(&mut barycentric_weight))
            .chain(Some(&mut extended_omega_inv))
            .chain(Some(&mut omega_inv))
            .batch_invert();

        EvaluationDomain {
            n,
            k,
            extended_k,
            omega,
            omega_inv,
            extended_omega,
            extended_omega_inv,
            g_coset,
            g_coset_inv,
            quotient_poly_degree,
            ifft_divisor,
            extended_ifft_divisor,
            t_evaluations,
            barycentric_weight,
        }
    }

    /// Obtains a polynomial in Lagrange form when given a vector of Lagrange
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn lagrange_from_vec(&self, values: Vec<F>) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Obtains a polynomial in coefficient form when given a vector of
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn coeff_from_vec(&self, values: Vec<F>) -> Polynomial<F, Coeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the coefficient basis
    pub fn empty_coeff(&self) -> Polynomial<F, Coeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis
    pub fn empty_lagrange(&self) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis, with
    /// deferred inversions.
    pub(crate) fn empty_lagrange_assigned(&self) -> Polynomial<Assigned<F>, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO.into(); self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the Lagrange coefficient basis
    pub fn constant_lagrange(&self, scalar: F) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.n as usize],
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the extended Lagrange coefficient
    /// basis
    pub fn empty_extended(&self) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.extended_len()],
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the extended Lagrange coefficient
    /// basis
    pub fn constant_extended(&self, scalar: F) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.extended_len()],
            _marker: PhantomData,
        }
    }

    /// This takes us from an n-length vector into the coefficient form.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    pub fn lagrange_to_coeff(&self, mut a: Polynomial<F, LagrangeCoeff>) -> Polynomial<F, Coeff> {
        assert_eq!(a.values.len(), 1 << self.k);

        // Perform inverse FFT to obtain the polynomial in coefficient form
        Self::ifft(&mut a.values, self.omega_inv, self.k, self.ifft_divisor);

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// This takes us from an n-length coefficient vector into a coset of the extended
    /// evaluation domain, rotating by `rotation` if desired.
    pub fn coeff_to_extended(
        &self,
        mut a: Polynomial<F, Coeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(a.values.len(), 1 << self.k);

        self.distribute_powers_zeta(&mut a.values, true);
        a.values.resize(self.extended_len(), F::ZERO);
        #[cfg(feature = "gpu")]
        best_fft_gpu(&mut [&mut a.values], self.extended_omega, self.extended_k).unwrap();

        #[cfg(feature = "cpu")]
        best_fft_cpu(&mut a.values, self.extended_omega, self.extended_k);

    
        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// Rotate the extended domain polynomial over the original domain.
    pub fn rotate_extended(
        &self,
        poly: &Polynomial<F, ExtendedLagrangeCoeff>,
        rotation: Rotation,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let new_rotation = ((1 << (self.extended_k - self.k)) * rotation.0.abs()) as usize;

        let mut poly = poly.clone();

        if rotation.0 >= 0 {
            poly.values.rotate_left(new_rotation);
        } else {
            poly.values.rotate_right(new_rotation);
        }

        poly
    }

    /// This takes us from the extended evaluation domain and gets us the
    /// quotient polynomial coefficients.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    // TODO/FIXME: caller should be responsible for truncating
    pub fn extended_to_coeff(&self, mut a: Polynomial<F, ExtendedLagrangeCoeff>) -> Vec<F> {
        assert_eq!(a.values.len(), self.extended_len());

        // Inverse FFT
        Self::ifft(
            &mut a.values,
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        );

        // Distribute powers to move from coset; opposite from the
        // transformation we performed earlier.
        self.distribute_powers_zeta(&mut a.values, false);

        // Truncate it to match the size of the quotient polynomial; the
        // evaluation domain might be slightly larger than necessary because
        // it always lies on a power-of-two boundary.
        a.values
            .truncate((&self.n * self.quotient_poly_degree) as usize);

        a.values
    }

    /// This divides the polynomial (in the extended domain) by the vanishing
    /// polynomial of the $2^k$ size domain.
    pub fn divide_by_vanishing_poly(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(a.values.len(), self.extended_len());

        // Divide to obtain the quotient polynomial in the coset evaluation
        // domain.
        parallelize(&mut a.values, |h, mut index| {
            for h in h {
                *h *= &self.t_evaluations[index % self.t_evaluations.len()];
                index += 1;
            }
        });

        Polynomial {
            values: a.values,
            _marker: PhantomData,
        }
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [zeta]a_1, [zeta^2]a_2, a_3, [zeta]a_4, [zeta^2]a_5, a_6, ...]`,
    /// where zeta is a cube root of unity in the multiplicative subgroup with
    /// order (p - 1), i.e. zeta^3 = 1.
    ///
    /// `into_coset` should be set to `true` when moving into the coset,
    /// and `false` when moving out. This toggles the choice of `zeta`.
    fn distribute_powers_zeta(&self, a: &mut [F], into_coset: bool) {
        let coset_powers = if into_coset {
            [self.g_coset, self.g_coset_inv]
        } else {
            [self.g_coset_inv, self.g_coset]
        };
        parallelize(a, |a, mut index| {
            for a in a {
                // Distribute powers to move into/from coset
                let i = index % (coset_powers.len() + 1);
                if i != 0 {
                    *a *= &coset_powers[i - 1];
                }
                index += 1;
            }
        });
    }

    fn ifft(a: &mut [F], omega_inv: F, log_n: u32, divisor: F) {

        #[cfg(feature = "gpu")]
        best_fft_gpu(&mut [a], omega_inv, log_n).unwrap();

        #[cfg(feature = "cpu")]
        best_fft_cpu(a, omega_inv, log_n);

        parallelize(a, |a, _| {
            for a in a {
                // Finish iFFT
                *a *= &divisor;
            }
        });
    }

    /// Get the size of the domain
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Get the size of the extended domain
    pub fn extended_k(&self) -> u32 {
        self.extended_k
    }

    /// Get the size of the extended domain
    pub fn extended_len(&self) -> usize {
        1 << self.extended_k
    }

    /// Get $\omega$, the generator of the $2^k$ order multiplicative subgroup.
    pub fn get_omega(&self) -> F {
        self.omega
    }

    /// Get $\omega^{-1}$, the inverse of the generator of the $2^k$ order
    /// multiplicative subgroup.
    pub fn get_omega_inv(&self) -> F {
        self.omega_inv
    }

    /// Get the generator of the extended domain's multiplicative subgroup.
    pub fn get_extended_omega(&self) -> F {
        self.extended_omega
    }

    /// Multiplies a value by some power of $\omega$, essentially rotating over
    /// the domain.
    pub fn rotate_omega(&self, value: F, rotation: Rotation) -> F {
        let mut point = value;
        if rotation.0 >= 0 {
            point *= &self.get_omega().pow_vartime(&[rotation.0 as u64]);
        } else {
            point *= &self
                .get_omega_inv()
                .pow_vartime(&[(rotation.0 as i64).unsigned_abs()]);
        }
        point
    }

    /// Computes evaluations (at the point `x`, where `xn = x^n`) of Lagrange
    /// basis polynomials `l_i(X)` defined such that `l_i(omega^i) = 1` and
    /// `l_i(omega^j) = 0` for all `j != i` at each provided rotation `i`.
    ///
    /// # Implementation
    ///
    /// The polynomial
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)$$
    /// has a root at all points in the domain except $\omega^i$, where it evaluates to
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// and so we divide that polynomial by this value to obtain $l_i(X)$. Since
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)
    ///       = \frac{X^n - 1}{X - \omega^i}$$
    /// then $l_i(x)$ for some $x$ is evaluated as
    ///     $$\left(\frac{x^n - 1}{x - \omega^i}\right)
    ///       \cdot \left(\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}\right).$$
    /// We refer to
    ///     $$1 \over \prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// as the barycentric weight of $\omega^i$.
    ///
    /// We know that for $i = 0$
    ///     $$\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)} = \frac{1}{n}.$$
    ///
    /// If we multiply $(1 / n)$ by $\omega^i$ then we obtain
    ///     $$\frac{1}{\prod_{j=0,j \neq 0}^{n - 1} (\omega^i - \omega^j)}
    ///       = \frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}$$
    /// which is the barycentric weight of $\omega^i$.
    pub fn l_i_range<I: IntoIterator<Item = i32> + Clone>(
        &self,
        x: F,
        xn: F,
        rotations: I,
    ) -> Vec<F> {
        let mut results;
        {
            let rotations = rotations.clone().into_iter();
            results = Vec::with_capacity(rotations.size_hint().1.unwrap_or(0));
            for rotation in rotations {
                let rotation = Rotation(rotation);
                let result = x - self.rotate_omega(F::ONE, rotation);
                results.push(result);
            }
            results.iter_mut().batch_invert();
        }

        let common = (xn - F::ONE) * self.barycentric_weight;
        for (rotation, result) in rotations.into_iter().zip(results.iter_mut()) {
            let rotation = Rotation(rotation);
            *result = self.rotate_omega(*result * common, rotation);
        }

        results
    }

    /// Gets the quotient polynomial's degree (as a multiple of n)
    pub fn get_quotient_poly_degree(&self) -> usize {
        self.quotient_poly_degree as usize
    }

    /// Obtain a pinned version of this evaluation domain; a structure with the
    /// minimal parameters needed to determine the rest of the evaluation
    /// domain.
    pub fn pinned(&self) -> PinnedEvaluationDomain<'_, F> {
        PinnedEvaluationDomain {
            k: &self.k,
            extended_k: &self.extended_k,
            omega: &self.omega,
        }
    }
}

/// Represents the minimal parameters that determine an `EvaluationDomain`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedEvaluationDomain<'a, F: Field> {
    k: &'a u32,
    extended_k: &'a u32,
    omega: &'a F,
}

/// Config gpu fft kernel
#[cfg(feature = "gpu")]
pub fn create_fft_kernel<Scalar,G>(_log_d: usize, priority: bool) -> Option<gpu::MultiFFTKernel<Scalar,G>>
where
    G: FftGroup<Scalar>,
    Scalar: Field,
{
    match gpu::MultiFFTKernel::create(priority) {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some(k)
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}
/* 
/// Wrap `gpu_fft_multiple`
#[cfg(feature = "gpu")]
pub fn best_fft_multiple_gpu<Scalar: Field, G: FftGroup<Scalar>>(
    polys: &mut [&mut [G]],
    omega: &Scalar,
    log_n: u32,
) -> gpu::GPUResult<()> {
    use crate::gpu::LockedMultiFFTKernel;

    let mut kern: Option<LockedMultiFFTKernel<_,_>> = Some(LockedMultiFFTKernel::<_,_>::new(log_n as usize, false));

    if let Some(ref mut kern) = kern {
        if kern
            .with(|k: &mut gpu::MultiFFTKernel<Scalar,G>| gpu_fft_multiple(k, polys, omega, log_n))
            .is_ok()
        {
            println!("using multiple GPUs");
            return Ok(());
        }
    }
    Ok(())
}

/// Use multiple gpu fft
#[cfg(feature = "gpu")]
pub fn gpu_fft_multiple<Scalar: Field, G: FftGroup<Scalar>>(
    kern: &mut gpu::MultiFFTKernel<Scalar,G>,
    polys: &mut [&mut [G]],
    omega: &Scalar,
    log_n: u32,
) -> gpu::GPUResult<()> {
    kern.fft_multiple(polys, omega, log_n)?;

    Ok(())
}
*/
fn omega<F: PrimeField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::ROOT_OF_UNITY;
    for _ in exp..F::S {
        omega = omega.square();
    }
    omega
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
use ec_gpu_gen::threadpool::Worker;
use std::sync::Arc;
use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::EcError;
use ec_gpu_gen::multiexp_cpu::QueryDensity;
use group::prime::PrimeCurveAffine;
use ec_gpu::GpuName;
use ec_gpu_gen::multiexp_cpu::SourceBuilder;
fn multiexp_gpu<Q, D, G, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Curve, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine + GpuName,
    S: SourceBuilder<G>,
{
    let exps = density_map.as_ref().generate_exps::<G::Scalar>(exponents);
    let (bss, skip) = bases.get();
    kern.multiexp(pool, bss, exps, skip).map_err(Into::into)
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test_log::test] // Automatically wraps test to initialize logging
fn test_ecgpu_msm_bls()
{   use ec_gpu_gen::threadpool::Worker;
    use ec_gpu_gen::rust_gpu_tools::Device;
    use blstrs::Bls12;
    use ec_gpu_gen::multiexp::MultiexpKernel;
    use ec_gpu_gen::multiexp_cpu::multiexp_cpu;
    use std::time::Instant;
    use rand_core::OsRng;
    use std::sync::Arc;
    use ec_gpu_gen::program;
    use pairing::Engine;
    use group::Curve;
    use ec_gpu_gen::multiexp_cpu::FullDensity;
    //use crate::halo2curves::pairing::Engine;
    //fil_logger::maybe_init();
    const MAX_LOG_D: usize = 16;
    const START_LOG_D: usize = 10;
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = MultiexpKernel::<<Bls12 as Engine>::G1Affine>::create(programs, &devices)
        .expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = OsRng;

    let mut bases = (0..(1 << START_LOG_D))
        .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
        .collect::<Vec<_>>();

    for log_d in START_LOG_D..=MAX_LOG_D {
        let g = Arc::new(bases.clone());

        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v = Arc::new(
            (0..samples)
                .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
                .collect::<Vec<_>>(),
        );

        let mut now = Instant::now();
        let gpu = multiexp_gpu(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu = multiexp_cpu(&pool, (g.clone(), 0), FullDensity, v.clone())
            .wait()
            .unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu, gpu);

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();

    }
}



#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test_log::test] // Automatically wraps test to initialize logging
fn test_ecgpu_fft_bls()
{   use ec_gpu_gen::threadpool::Worker;
    use ec_gpu_gen::rust_gpu_tools::Device;
    use ec_gpu_gen::fft::FftKernel;
    use blstrs::Scalar as Fr;
    use std::time::Instant;
    use rand_core::OsRng;
    use ec_gpu_gen::fft_cpu::serial_fft;
    use ec_gpu_gen::fft_cpu::parallel_fft;

    //fil_logger::maybe_init();
    let mut rng = OsRng;

    let worker = Worker::new();
    let log_threads = worker.log_num_threads();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for log_d in 1..=20 {
        let d = 1 << log_d;

        let mut v1_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let v1_omega = omega::<Fr>(v1_coeffs.len());
        let mut v2_coeffs = v1_coeffs.clone();
        let v2_omega = v1_omega;

        println!("Testing FFT for {} elements...", d);

        let mut now = Instant::now();
        kern.radix_fft(&mut v1_coeffs, &v1_omega, log_d).expect("GPU FFT failed!");

       // kern.radix_fft_many(&mut [&mut v1_coeffs], &[v1_omega], &[log_d])
       //     .expect("GPU FFT failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        if log_d <= log_threads {
            serial_fft::<Fr>(&mut v2_coeffs, &v2_omega, log_d);
        } else {
            parallel_fft::<Fr>(&mut v2_coeffs, &worker, &v2_omega, log_d, log_threads);
        }
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert!(v1_coeffs == v2_coeffs);
        println!("============================");

    }
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test_log::test] // Automatically wraps test to initialize logging
fn test_ecgpu_fft_bn256()
{   use ec_gpu_gen::threadpool::Worker;
    use ec_gpu_gen::rust_gpu_tools::Device;
    use ec_gpu_gen::fft::FftKernel;
    use crate::halo2curves::bn256::Fr;
    use std::time::Instant;
    use rand_core::OsRng;
    use ec_gpu_gen::fft_cpu::serial_fft;
    use ec_gpu_gen::fft_cpu::parallel_fft;

    //fil_logger::maybe_init();
    let mut rng = OsRng;

    let worker = Worker::new();
    let log_threads = worker.log_num_threads();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for log_d in 1..=20 {
        let d = 1 << log_d;

        let mut v1_coeffs = (0..d).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
        let v1_omega = omega::<Fr>(v1_coeffs.len());
        let mut v2_coeffs = v1_coeffs.clone();
        let v2_omega = v1_omega;

        println!("Testing FFT for {} elements...", d);

        let mut now = Instant::now();

        kern.radix_fft(&mut v1_coeffs, &v1_omega, log_d).expect("GPU FFT failed!");

        //kern.radix_fft_many(&mut [&mut v1_coeffs], &[v1_omega], &[log_d])
        //    .expect("GPU FFT failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        if log_d <= log_threads {
            serial_fft::<Fr>(&mut v2_coeffs, &v2_omega, log_d);
        } else {
            parallel_fft::<Fr>(&mut v2_coeffs, &worker, &v2_omega, log_d, log_threads);
        }
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU ({} cores) took {}ms.", 1 << log_threads, cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert!(v1_coeffs == v2_coeffs);
        println!("============================");

    }
}


#[cfg(any(feature = "cuda", feature = "opencl"))]
#[test_log::test] // Automatically wraps test to initialize logging
fn test_best_fft_multiple_gpu() {
    use crate::halo2curves::bn256::Fr;
    use ff::Field;
    use crate::poly::EvaluationDomain;
    use rand_core::OsRng;
    use crate::arithmetic::best_fft_cpu;
    use crate::arithmetic::best_fft;
    let worker = ec_gpu_gen::threadpool::Worker::new();
    use crate::gpu::LockedFftKernel;

    for k in 14..=15 {
        let rng = OsRng;
        // polynomial degree n = 2^k
        let n = 1u64 << k;
        // polynomial coeffs
        let coeffs: Vec<_> = (0..n).map(|_| Fr::random(rng)).collect();
        // evaluation domain
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(1, k);

        println!("Testing FFT for {} elements, degree {}...", n, k);

        let mut prev_fft_coeffs = coeffs.clone();

        let mut now = std::time::Instant::now();
        
        best_fft_cpu(&mut prev_fft_coeffs, domain.get_omega(), k);
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        //end_timer!(start);
        let mut optimized_fft_coeffs = coeffs.clone();

        now = std::time::Instant::now();
        let mut fft_kern = Some(LockedFftKernel::new(false));

        best_fft(&mut fft_kern, &worker, &mut [&mut optimized_fft_coeffs], 
            &[domain.get_omega()], &[k]).unwrap();

        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        //end_timer!(start);
        println!("GPU took {}ms.", gpu_dur);
        assert_eq!(coeffs, optimized_fft_coeffs);
    }
}
#[cfg(feature = "gpu")]
#[test_log::test] // Automatically wraps test to initialize logging
fn test_best_fft_multiple_gpu() {

    use crate::gpu::LockedMultiFFTKernel;
    use crate::halo2curves::bn256::Fr;
    use::halo2curves::bn256::G1;

    use ff::Field;
    use crate::poly::EvaluationDomain;
    use ark_std::{end_timer, start_timer};
    use rand_core::OsRng;
    use crate::arithmetic::best_fft_cpu;
    for k in 18..=20 {
        let rng = OsRng;
        // polynomial degree n = 2^k
        let n = 1u64 << k;
        // polynomial coeffs
        let coeffs: Vec<_> = (0..n).map(|_| Fr::random(rng)).collect();
        // evaluation domain
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(1, k);

        println!("Testing FFT for {} elements, degree {}...", n, k);

        let mut prev_fft_coeffs = coeffs.clone();

        let mut now = std::time::Instant::now();
        
        best_fft_cpu(&mut prev_fft_coeffs, domain.get_omega(), k);
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        //end_timer!(start);
        let mut optimized_fft_coeffs = coeffs.clone();
        now = std::time::Instant::now();

        best_fft_gpu(
            &mut [&mut optimized_fft_coeffs],
            domain.get_omega(),
           k as u32,
        ).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        //end_timer!(start);
        println!("GPU took {}ms.", gpu_dur);
        assert_eq!(prev_fft_coeffs, optimized_fft_coeffs);
    }
}



#[test]
fn test_fft() {
    use crate::poly::EvaluationDomain;
    use ark_std::{end_timer, start_timer};

    use halo2curves::bn256::Fr;
    use rand_core::OsRng;

    for k in 18..=26 {
        let rng = OsRng;
        // polynomial degree n = 2^k
        let n = 1u64 << k;
        // polynomial coeffs
        let coeffs: Vec<_> = (0..n).map(|_| Fr::random(rng)).collect();
        // evaluation domain
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(1, k);

        let message = format!("prev_fft degree {}", k);
        let start = start_timer!(|| message);

        let mut prev_fft_coeffs = coeffs.clone();

        #[cfg(feature = "gpu")]
        best_fft_gpu(&mut [&mut prev_fft_coeffs], domain.get_omega(), k).unwrap();

        #[cfg(feature = "cpu")]
        best_fft_cpu(&mut prev_fft_coeffs, domain.get_omega(), k);

      
        end_timer!(start);
    }
}



#[test]
fn test_rotate() {
    use rand_core::OsRng;

    use crate::arithmetic::eval_polynomial;
    use halo2curves::pasta::pallas::Scalar;

    let domain = EvaluationDomain::<Scalar>::new(1, 3);
    let rng = OsRng;

    let mut poly = domain.empty_lagrange();
    assert_eq!(poly.len(), 8);
    for value in poly.iter_mut() {
        *value = Scalar::random(rng);
    }

    let poly_rotated_cur = poly.rotate(Rotation::cur());
    let poly_rotated_next = poly.rotate(Rotation::next());
    let poly_rotated_prev = poly.rotate(Rotation::prev());

    let poly = domain.lagrange_to_coeff(poly);
    let poly_rotated_cur = domain.lagrange_to_coeff(poly_rotated_cur);
    let poly_rotated_next = domain.lagrange_to_coeff(poly_rotated_next);
    let poly_rotated_prev = domain.lagrange_to_coeff(poly_rotated_prev);

    let x = Scalar::random(rng);

    assert_eq!(
        eval_polynomial(&poly[..], x),
        eval_polynomial(&poly_rotated_cur[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega),
        eval_polynomial(&poly_rotated_next[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega_inv),
        eval_polynomial(&poly_rotated_prev[..], x)
    );
}

#[test]
fn test_l_i() {
    use rand_core::OsRng;

    use crate::arithmetic::{eval_polynomial, lagrange_interpolate};
    use halo2curves::pasta::pallas::Scalar;
    let domain = EvaluationDomain::<Scalar>::new(1, 3);

    let mut l = vec![];
    let mut points = vec![];
    for i in 0..8 {
        points.push(domain.omega.pow(&[i, 0, 0, 0]));
    }
    for i in 0..8 {
        let mut l_i = vec![Scalar::zero(); 8];
        l_i[i] = Scalar::ONE;
        let l_i = lagrange_interpolate(&points[..], &l_i[..]);
        l.push(l_i);
    }

    let x = Scalar::random(OsRng);
    let xn = x.pow(&[8, 0, 0, 0]);

    let evaluations = domain.l_i_range(x, xn, -7..=7);
    for i in 0..8 {
        assert_eq!(eval_polynomial(&l[i][..], x), evaluations[7 + i]);
        assert_eq!(eval_polynomial(&l[(8 - i) % 8][..], x), evaluations[7 - i]);
    }
}
