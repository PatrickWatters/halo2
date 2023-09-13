use std::cmp;
use std::sync::{Arc, RwLock};

use ec_gpu::GpuName;
use ff::Field;
use log::{error, info};
use rust_gpu_tools::{program_closures, LocalBuffer, Program};
use std::time::Instant;
use crate::error::{EcError, EcResult};
use crate::threadpool::THREAD_POOL;
use std::error::Error;
const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

struct FFTLoggingInfo {
    toaltime: String,
    size: String,
    logn: String,
    write_from_buffer: String,
    fft_rounds: String,
    read_into_buffer: String,
    precalculate: String,
}

/// FFT kernel for a single GPU.
pub struct SingleFftKernel<'a, F>
where
    F: Field + GpuName,
{
    program: Program,
    /// An optional function which will be called at places where it is possible to abort the FFT
    /// calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, F: Field + GpuName> SingleFftKernel<'a, F> {
    /// Create a new FFT instance for the given device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        program: Program,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        Ok(SingleFftKernel {
            program,
            maybe_abort,
            _phantom: Default::default(),
        })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        let closures = program_closures!(|program, input: &mut [F]| -> EcResult<()> {
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
            let mut now: Instant = Instant::now();

            let n = 1 << log_n;

            stat_collector.size = format!("{}",n);
            stat_collector.logn = format!("{}",log_n as u32);

            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![F::ZERO; 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = F::ONE;
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![F::ZERO; 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            let precalculate_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            //println!("precalculate took {}ms.", precalculate_dur);
            stat_collector.precalculate = format!("{:?}ms",precalculate_dur);

            now = Instant::now();
            program.write_from_buffer(&mut src_buffer, &*input)?;
            let write_from_buffer_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            //println!("write_from_buffer took {}ms.", write_from_buffer_dur);
            stat_collector.write_from_buffer = format!("{:?}ms",write_from_buffer_dur);

            now = Instant::now();

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                if let Some(maybe_abort) = &self.maybe_abort {
                    if maybe_abort() {
                        return Err(EcError::Aborted);
                    }
                }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel_name = format!("{}_radix_fft", F::name());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<F>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            let fft_rounds_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            //println!("fft_rounds took {}ms.", fft_rounds_dur);
            stat_collector.fft_rounds = format!("{:?}ms",fft_rounds_dur);


            now = Instant::now();

            program.read_into_buffer(&src_buffer, input)?;

            let read_into_buffer_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            //println!("read_into_buffer took {}ms.", read_into_buffer_dur);
            stat_collector.read_into_buffer = format!("{:?}ms",read_into_buffer_dur);
            
            let gpu_dur = dur.elapsed().as_secs() * 1000 + dur.elapsed().subsec_millis() as u64;
            //println!("radix_fft took {}ms.", gpu_dur);
            stat_collector.toaltime = format!("{:?}ms",gpu_dur);
            
            let _ = log_stats(stat_collector);


            Ok(())
        });

        self.program.run(closures, input)
    }
}

/// One FFT kernel for each GPU available.
pub struct FftKernel<'a, F>
where
    F: Field + GpuName,
{
    kernels: Vec<SingleFftKernel<'a, F>>,
}

impl<'a, F> FftKernel<'a, F>
where
    F: Field + GpuName,
{
    /// Create new kernels, one for each given device.
    pub fn create(programs: Vec<Program>) -> EcResult<Self> {
        Self::create_optional_abort(programs, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        programs: Vec<Program>,
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(programs, Some(maybe_abort))
    }

    fn create_optional_abort(
        programs: Vec<Program>,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let kernels: Vec<_> = programs
            .into_iter()
            .filter_map(|program| {
                let device_name = program.device_name().to_string();
                let kernel = SingleFftKernel::<F>::create(program, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device_name, e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        info!("FFT: {} working device(s) selected. ", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!("FFT: Device {}: {}", i, k.program.device_name(),);
        }

        Ok(Self { kernels })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses the first available GPU.
    pub fn radix_fft(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        self.kernels[0].radix_fft(input, omega, log_n)
    }

    /// Performs FFT on `inputs`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses all available GPUs to distribute the work.
    /// 
    /// 

    pub fn radix_fft_many(
        &mut self,
        inputs: &mut [&mut [F]],
        omegas: &[F],
        log_ns: &[u32],
    ) -> EcResult<()> {

        let n: usize = inputs.len();
        //let num_devices = self.kernels.len();
        let num_devices = 1;
        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        let result: Arc<RwLock<Result<(), EcError>>> = Arc::new(RwLock::new(Ok(())));


        THREAD_POOL.scoped(|s| {
            for (((inputs, omegas), log_ns), kern) in inputs
                .chunks_mut(chunk_size)
                .zip(omegas.chunks(chunk_size))
                .zip(log_ns.chunks(chunk_size))
                .zip(self.kernels.iter_mut())
            {
                let result = result.clone();
                s.execute(move || {
                    for ((input, omega), log_n) in
                        inputs.iter_mut().zip(omegas.iter()).zip(log_ns.iter())
                    {
                        if result.read().unwrap().is_err() {
                            break;
                        }

                        if let Err(err) = kern.radix_fft(input, omega, *log_n) {
                            *result.write().unwrap() = Err(err);
                            break;
                        }
                    }
                });
            }
        });



        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }
}

fn log_stats(stat_collector:FFTLoggingInfo)-> Result<(), Box<dyn Error>>
{   
    use std::path::Path;
    //let filename = "/home/project2reu/patrick/gpuhalo2/halo2/stats/gpu_fft_breakdown.csv";
    let filename = "ec_gpu_fft_stats.csv";

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
