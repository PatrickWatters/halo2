use crate::arithmetic::FftGroup;
//use crate::halo2curves::group::Group;
use std::marker::PhantomData;

use crate::gpu::{
    error::{GPUError, GPUResult},
    get_lock_name_and_gpu_range, locks, sources,
};
use crate::worker::THREAD_POOL;
use ff::Field;
use log::{error, info};
use rayon::join;
use rust_gpu_tools::*;
use std::cmp::min;
use std::ops::MulAssign;
use std::{cmp, env};
use ark_std::time::Instant;
use std::error::Error;
use std::time::Duration;

#[derive(serde::Serialize)]
struct FFTLoggingInfo {
    toaltime: String,
    size: String,
    logn: String,
    write_from_buffer: String,
    fft_rounds: String,
    read_into_buffer: String,
    precalculate: String,
}

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 9; // Radix512
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 8; // 256

/// SingleFFTKernel
#[allow(missing_debug_implementations)]
pub struct SingleFFTKernel<Scalar,G>

where
    G: FftGroup<Scalar>,
    Scalar: Field,
{
    program: opencl::Program,
    pq_buffer: opencl::Buffer<Scalar>,
    omegas_buffer: opencl::Buffer<Scalar>,
    #[allow(dead_code)]
    priority: bool,
    phantom: PhantomData<G>,
}

impl<Scalar,G> SingleFFTKernel<Scalar,G>
where
    G: FftGroup<Scalar>,
    Scalar: Field,
{
    /// New gpu fft kernel device
    pub fn create(device: opencl::Device, priority: bool) -> GPUResult<SingleFFTKernel<Scalar,G>> {
        let src = sources::kernel::<Scalar,G>(device.brand() == opencl::Brand::Nvidia);

        let program = opencl::Program::from_opencl(device, &src)?;
        let pq_buffer: opencl::Buffer<Scalar> = program.create_buffer::<Scalar>(1 << MAX_LOG2_RADIX >> 1)?;
        let omegas_buffer = program.create_buffer::<Scalar>(LOG2_MAX_ELEMENTS)?;

        info!("FFT: Device: {}", program.device().name());

        Ok(SingleFFTKernel {
            program,
            pq_buffer,
            omegas_buffer,
            priority,
            phantom: PhantomData,

        })
    }

    /// Peforms a FFT round
    /// * `log_n` - Specifies log2 of number of elements
    /// * `log_p` - Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    /// * `deg` - 1=>radix2, 2=>radix4, 3=>radix8, ...
    /// * `max_deg` - The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    fn radix_fft_round(
        &mut self,
        src_buffer: &opencl::Buffer<G>,
        dst_buffer: &opencl::Buffer<G>,
        log_n: u32,
        log_p: u32,
        deg: u32,
        max_deg: u32,
    ) -> GPUResult<()> {
        // if locks::PriorityLock::should_break(self.priority) {
        //     return Err(GPUError::GPUTaken);
        // }

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = (n >> deg) * local_work_size;
        let kernel = self.program.create_kernel(
            "radix_fft",
            global_work_size as usize,
            Some(local_work_size as usize),
        );
        kernel
            .arg(src_buffer)
            .arg(dst_buffer)
            .arg(&self.pq_buffer)
            .arg(&self.omegas_buffer)
            .arg(opencl::LocalBuffer::<G>::new(1 << deg))
            .arg(n)
            .arg(log_p)
            .arg(deg)
            .arg(max_deg)
            .run()?;
        Ok(())
    }

    /// Share some precalculated values between threads to boost the performance
    fn setup_pq_omegas(&mut self, omega: &Scalar, n: usize, max_deg: u32) -> GPUResult<()> {
        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![Scalar::ZERO; 1 << max_deg >> 1];
        let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
        pq[0] = Scalar::ONE;
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }
        self.pq_buffer.write_from(0, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![Scalar::ZERO; 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow_vartime([2u64]);
        }
        self.omegas_buffer.write_from(0, &omegas)?;

        Ok(())
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(
        &mut self,
        a: &mut [G],
        omega: &Scalar,
        log_n: u32,
    ) -> GPUResult<()> {
        let mut dur = Instant::now();

        let mut stat_collector = FFTLoggingInfo{
            toaltime:String::from(""),
            size:String::from(""),
            logn:String::from(""),
            fft_rounds:String::from(""), 
            write_from_buffer:String::from(""), 
            read_into_buffer:String::from(""), 
            precalculate:String::from(""), 

        };

        stat_collector.size = format!("{}",a.len() as u32);
        stat_collector.logn = format!("{}",log_n as u32);
        
        //println!("{}", log_n.to_string());
        let n = 1 << log_n;
        let mut src_buffer: opencl::Buffer<G> = self.program.create_buffer::<G>(n)?;
        let mut dst_buffer: opencl::Buffer<G> = self.program.create_buffer::<G>(n)?;

        let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

        let mut now: Instant = Instant::now();
        self.setup_pq_omegas(omega, n, max_deg)?;
        let precalculate_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("precalculate took {}ms.", precalculate_dur);
        stat_collector.precalculate = format!("{:?}",precalculate_dur);

        now = Instant::now();
        src_buffer.write_from(0, &*a)?;
        let write_from_buffer_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("write_from_buffer took {}ms.", write_from_buffer_dur);
        stat_collector.write_from_buffer = format!("{:?}ms",write_from_buffer_dur);

        now = Instant::now();
        let mut log_p = 0u32;
        while log_p < log_n {
            let deg = cmp::min(max_deg, log_n - log_p);
            self.radix_fft_round(&src_buffer, &dst_buffer, log_n, log_p, deg, max_deg)?;
            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }
        let fft_rounds_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("fft_rounds took {}ms.", fft_rounds_dur);
        stat_collector.fft_rounds = format!("{:?}ms",fft_rounds_dur);

        now = Instant::now();
        src_buffer.read_into(0, a)?;
        let read_into_buffer_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("read_into_buffer took {}ms.", read_into_buffer_dur);
        stat_collector.read_into_buffer = format!("{:?}ms",read_into_buffer_dur);
      
        let gpu_dur = dur.elapsed().as_secs() * 1000 + dur.elapsed().subsec_millis() as u64;
        println!("radix_fft took {}ms.", gpu_dur);
        stat_collector.toaltime = format!("{:?}ms",gpu_dur);
      
        let _ = log_stats(stat_collector);
        Ok(())
    }
}


fn log_stats(stat_collector:FFTLoggingInfo)-> Result<(), Box<dyn Error>>
{   
    use std::path::Path;
    //let filename = "/home/project2reu/patrick/gpuhalo2/halo2/stats/fft_times.csv";
    let filename = "../halo2/stats/fft_times.csv";
    //let filename = "gpu_fft_breakdownt.csv";

    let already_exists= Path::new(filename).exists();

    let file = std::fs::OpenOptions::new()
    .write(true)
    .create(true)
    .append(true)
    .open(filename)
    .unwrap();

    let mut wtr = csv::Writer::from_writer(file);
    
    if already_exists == false
    {
        wtr.write_record(&["size","log_n", "total_time","precalculate", "read_into_buffer", "fft_rounds", "write_from_buffer"])?;    
    }

    wtr.write_record(&[stat_collector.size, stat_collector.logn, stat_collector.toaltime, stat_collector.precalculate, 
        stat_collector.read_into_buffer, stat_collector.fft_rounds, stat_collector.write_from_buffer])?;
    wtr.flush()?;
    Ok(())    
}


/// Gpu fft kernel vec
#[allow(missing_debug_implementations)]
pub struct MultiFFTKernel<Scalar,G>
where
    G: FftGroup<Scalar>,
    Scalar: Field,
{
    //dummy:   G,
    kernels: Vec<SingleFFTKernel<Scalar,G>>,
    _lock: locks::GPULock,
}

impl<Scalar,G> MultiFFTKernel<Scalar,G>
where
    G: FftGroup<Scalar>,
    Scalar: Field,
{
    /// New gpu kernel device
    pub fn create(priority: bool) -> GPUResult<MultiFFTKernel<Scalar,G>> {
        let mut all_devices = opencl::Device::all();
        //let num_devices = all_devices.len();
        let num_devices=1;
        let (lock_index, gpu_range) = get_lock_name_and_gpu_range(num_devices);

        let lock = locks::GPULock::lock(lock_index);

        let devices: Vec<&opencl::Device> = all_devices.drain(gpu_range).collect();

        // use all of the  GPUs
        let kernels: Vec<_> = devices
            .into_iter()
            .map(|d| (d, SingleFFTKernel::<Scalar, G>::create(d.clone(), priority)))
            .filter_map(|(device, res)| {
                if let Err(ref e) = res {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device.name(),
                        e
                    );
                }
                res.ok()
            })
            .collect();

        Ok(MultiFFTKernel {
            kernels,
            _lock: lock
        })
    }


  
    /// fft_multiple call for kernel radix_fft
    pub fn fft_multiple(
        &mut self,
        polys: &mut [&mut [G]],
        omega: &Scalar,
        log_n: u32,
    ) -> GPUResult<()> {

        use rayon::prelude::*;
        let mut now: Instant = Instant::now();
        for poly in polys.chunks_mut(self.kernels.len()) {
            crate::worker::THREAD_POOL.install(|| {
                poly.par_iter_mut()
                    .zip(self.kernels.par_iter_mut())
                    .for_each(|(p, kern)| kern.radix_fft(p, omega, log_n).unwrap())
            });
        }
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("fft_multiple took {}ms.", gpu_dur);

        Ok(())
    }

}
