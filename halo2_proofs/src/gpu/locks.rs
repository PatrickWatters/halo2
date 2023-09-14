use std::fs::File;
use std::path::PathBuf;

use ec_gpu_gen::fft::FftKernel;
use ec_gpu_gen::rust_gpu_tools::{Device, UniqueId};
use ec_gpu_gen::EcError;
use ff::Field;
use fs2::FileExt;
use group::prime::PrimeCurveAffine;
use log::{debug, info, warn};

use crate::arithmetic::FftGroup;
use crate::gpu::error::{GpuError, GpuResult};
use crate::gpu::{CpuGpuMultiexpKernel, GpuName};

const GPU_LOCK_NAME: &str = "bellman.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "bellman.priority.lock";

fn tmp_path(filename: &str, id: Option<UniqueId>) -> PathBuf {
    let temp_file = match id {
        Some(id) => format!("{}.{}", filename, id),
        None => filename.to_string(),
    };
    std::env::temp_dir().join(temp_file)
}

/// Information about a lock.
///
/// If the `device` is `None`, it means that the lock spans all available devices.
#[derive(Debug)]
struct LockInfo<'a> {
    file: File,
    path: PathBuf,
    device: Option<&'a Device>,
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GPULock<'a>(Vec<LockInfo<'a>>);

impl<'a> GPULock<'a> {
    pub fn lock() -> Self {
        if let Ok(val) = std::env::var("BELLPERSON_GPUS_PER_LOCK") {
            match val.parse::<usize>() {
                Ok(val) if val > 0 => {
                    let devices = Device::all();
                    info!(
                        "BELLPERSON_GPUS_PER_LOCK == {}, try lock {}/{} gpus",
                        val,
                        val,
                        devices.len(),
                    );

                    let mut locks = Vec::new();
                    for (index, device) in devices.iter().enumerate() {
                        let uid = device.unique_id();
                        let path = tmp_path(GPU_LOCK_NAME, Some(uid));
                        debug!("Acquiring GPU lock {}/{} at {:?} ...", index, val, &path);
                        let file = File::create(&path).unwrap_or_else(|_| {
                            panic!("Cannot create GPU {:?} lock file at {:?}", uid, &path)
                        });
                        if file.try_lock_exclusive().is_err() {
                            continue;
                        }
                        debug!("GPU lock acquired at {:?}", path);
                        locks.push(LockInfo {
                            file,
                            path,
                            device: Some(device),
                        });
                        if locks.len() >= val {
                            break;
                        }
                    }

                    return GPULock(locks);
                }
                Ok(val) if val == 0 => {
                    info!("BELLPERSON_GPUS_PER_LOCK == 0, no lock acquired");
                    return GPULock(Vec::new());
                }
                Ok(val) => warn!(
                    "BELLPERSON_GPUS_PER_LOCK has invalid value {}, using all gpus",
                    val,
                ),
                Err(_) => warn!("BELLPERSON_GPUS_PER_LOCK parsing failed, using all gpus"),
            };
        }

        info!("BELLPERSON_GPUS_PER_LOCK fallback to single lock mode");

        // Fallback to create single lock
        let path = tmp_path(GPU_LOCK_NAME, None);
        debug!("Acquiring GPU lock at {:?} ...", &path);
        let file = File::create(&path).unwrap_or_else(|_| {
            panic!("Cannot create GPU lock file at {:?}", &path);
        });
        file.lock_exclusive().unwrap();
        debug!("GPU lock acquired at {:?}", path);
        GPULock(vec![LockInfo {
            file,
            path,
            device: None,
        }])
    }

    /// Retuns the devices this lock holds.
    ///
    /// It returns all devices if there is no lock at all.
    fn devices(&self) -> Vec<&'a Device> {
        // No locks signal that there should no lock be at all. If there is only one lock, which
        // doesn't specify a devices, it signals that it's a single lock, that spans all devices.
        if self.0.is_empty() || (self.0.len() == 1 && self.0[0].device.is_none()) {
            Device::all()
        } else {
            self.0
                .iter()
                .filter_map(|&LockInfo { device, .. }| device)
                .collect()
        }
    }
}

impl Drop for GPULock<'_> {
    fn drop(&mut self) {
        for lock_info in &self.0 {
            lock_info.file.unlock().unwrap();
            debug!("GPU lock released at {:?}", lock_info.path);
        }
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority process
/// needs to acquire the GPU really soon. Acquiring the `PriorityLock` is like
/// signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub(crate) struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME, None);
        debug!("Acquiring priority lock at {:?} ...", &priority_lock_file);
        let f = File::create(&priority_lock_file).unwrap_or_else(|_| {
            panic!(
                "Cannot create priority lock file at {:?}",
                &priority_lock_file
            )
        });
        f.lock_exclusive().unwrap();
        debug!("Priority lock acquired!");
        PriorityLock(f)
    }

    fn wait(priority: bool) {
        if !priority {
            if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME, None))
                .unwrap()
                .lock_exclusive()
            {
                warn!("failed to create priority log: {:?}", err);
            }
        }
    }

    /// Returns true if the priority lock is currently taken.
    ///
    /// This is used by low priority proofs to determine whether to run on the GPU or not.
    ///
    /// It also returns `false` in case the state of the lock cannot be determined.
    fn is_taken() -> bool {
        if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME, None))
            .unwrap()
            .try_lock_shared()
        {
            // Check that the error is actually a locking one
            if err.raw_os_error() == fs2::lock_contended_error().raw_os_error() {
                return true;
            }
            warn!("failed to check lock: {:?}", err);
        }
        false
    }
}

impl Drop for PriorityLock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("Priority lock released!");
    }
}

fn create_fft_kernel<'a,Scalar,G>(priority: bool) -> Option<(FftKernel<'a,Scalar,G>, GPULock<'a>)>
where
    G: FftGroup<Scalar>+GpuName,
    Scalar: Field,

{
    let lock = GPULock::lock();
    let programs = lock
        .devices()
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .ok()?;

    let kernel = if priority {
        FftKernel::create(programs)
    } else {
        // Low priority kernels may be aborted if a high priority kernel wants to run/is running.
        FftKernel::create_with_abort(programs, &PriorityLock::is_taken)
    };
    match kernel {
        Ok(k) => {
            info!("GPU FFT kernel instantiated!");
            Some((k, lock))
        }
        Err(e) => {
            warn!("Cannot instantiate GPU FFT kernel! Error: {}", e);
            None
        }
    }
}

macro_rules! locked_kernel {
    (
        $kernel:ty,
        $func:ident,
        $name:expr,
        pub struct $class:ident<$a:lifetime, $generic:ident, $generic1:ident>
        where
        $(
            $bound:ty: $boundvalue:tt<$boundvalue2:tt> $(+ $morebounds:tt)*,
        )+
    ) => {
        #[allow(missing_debug_implementations)]
        pub struct $class<$a, $generic, $generic1> // Removed lifetime parameter here
        where
            $(
                $bound: $boundvalue<$boundvalue2> $(+ $morebounds)*,
            )+
        {
            priority: bool,
            kernel_and_lock: Option<($kernel, GPULock<'a>)>,
        }

        impl<'a, Scalar, G> $class<'a, Scalar, G>
        where
            G: FftGroup<Scalar>,
            Scalar: Field,
        {
            pub fn new(priority: bool) -> Self {
                Self {
                    priority,
                    kernel_and_lock: None,
                }
            }

            fn init(&mut self) {
                if self.kernel_and_lock.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    if let Some((kernel, lock)) = $func(self.priority) {
                        self.kernel_and_lock = Some((kernel, lock));
                    }
                }
            }

            fn free(&mut self) {
                if let Some(_) = self.kernel_and_lock.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<Fun, R>(&mut self, mut f: Fun) -> GpuResult<R>
            where
                Fun: FnMut(&mut $kernel) -> GpuResult<R>,
            {
                if let Ok(bellman_no_gpu) = std::env::var("BELLMAN_NO_GPU") {
                    if bellman_no_gpu != "0" {
                        return Err(GpuError::GpuDisabled);
                    }
                }

                loop {
                    // `init()` is a possibly blocking call that waits until the GPU is available.
                    self.init();
                    if let Some((ref mut k, ref _gpu_lock)) = self.kernel_and_lock {
                        match f(k) {
                            // Re-trying to run on the GPU is the core of this loop, all other
                            // cases abort the loop.
                            Err(GpuError::EcGpu(EcError::Aborted)) => {
                                self.free();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => return Ok(v),
                        }
                    } else {
                        return Err(GpuError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

// Define the struct outside of the macro
pub struct LockedFftKernel<'a, Scalar, G>
where
    G: FftGroup<Scalar>+GpuName,
    Scalar: Field,
{
    priority: bool,
    kernel_and_lock: Option<(FftKernel<'a, Scalar, G>, GPULock<'a>)>,
}

impl<'a, Scalar, G> LockedFftKernel<'a, Scalar, G>
where
    G: FftGroup<Scalar>+GpuName,
    Scalar: Field,
{
    pub fn new(priority: bool) -> Self {
        Self {
            priority,
            kernel_and_lock: None,
        }
    }

    fn init(&mut self) {
        if self.kernel_and_lock.is_none() {
            PriorityLock::wait(self.priority);
            info!("GPU is available for FFT!");
            if let Some((kernel, lock)) = create_fft_kernel(self.priority) {
                self.kernel_and_lock = Some((kernel, lock));
            }
        }
    }

    fn free(&mut self) {
        if let Some(_) = self.kernel_and_lock.take() {
            warn!("GPU acquired by a high priority process! Freeing up FFT kernels...");
        }
    }

    pub fn with<Fun, R>(&mut self, mut f: Fun) -> GpuResult<R>
    where
        Fun: FnMut(&mut FftKernel<'a, Scalar, G>) -> GpuResult<R>,
    {
        if let Ok(bellman_no_gpu) = std::env::var("BELLMAN_NO_GPU") {
            if bellman_no_gpu != "0" {
                return Err(GpuError::GpuDisabled);
            }
        }

        loop {
            // `init()` is a possibly blocking call that waits until the GPU is available.
            self.init();
            if let Some((ref mut k, ref _gpu_lock)) = self.kernel_and_lock {
                match f(k) {
                    // Re-trying to run on the GPU is the core of this loop, all other
                    // cases abort the loop.
                    Err(GpuError::EcGpu(EcError::Aborted)) => {
                        self.free();
                    }
                    Err(e) => {
                        warn!("GPU FFT failed! Falling back to CPU... Error: {}", e);
                        return Err(e);
                    }
                    Ok(v) => return Ok(v),
                }
            } else {
                return Err(GpuError::KernelUninitialized);
            }
        }
    }
}

// Define the macro for other parts of your code
macro_rules! locked_kernel {
    (
        $kernel:ty,
        $func:ident,
        $name:expr,
        pub struct $class:ident<$lifetime:lifetime, $generic:ident, $generic1:ident>
        where
        $(
            $bound:ty: $boundvalue:tt<$boundvalue2:tt> $(+ $morebounds:tt)*,
        )+
    ) => {
        // Any other code related to the macro goes here
    };
}



/* 
locked_kernel!(
    FftKernel<'a,Scalar,G>,
    create_fft_kernel,
    "FFT",
    pub struct LockedFftKernel<'a,Scalar,G> 
    where
    G: FftGroup<Scalar>,
    Scalar: Field,
);
*/
/* 
locked_kernel!(
    CpuGpuMultiexpKernel<'a, Scalar,G>,
    create_multiexp_kernel,
    "Multiexp",
    pub struct LockedMultiexpKernel<'a, Scalar,G>
    where
        G: PrimeCurveAffine + GpuName,
);
*/