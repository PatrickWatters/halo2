#[macro_use]
extern crate criterion;

use crate::arithmetic::{best_fft_gpu,best_fft_cpu};
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;

use criterion::{BenchmarkId, Criterion};
use rand_core::OsRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    for k in 3..19 {
        group.bench_function(BenchmarkId::new("k", k), |b| {
            let mut a = (0..(1 << k)).map(|_| Fp::random(OsRng)).collect::<Vec<_>>();
            let omega = Fp::random(OsRng); // would be weird if this mattered
            b.iter(|| {
                if cfg!(feature = "gpu")
                {
                    best_fft_gpu(&mut [&mut a], omega, k).unwrap();
                }else {
                    best_fft_cpu(&mut a, omega, k as u32);
                }

                //best_fft(&mut a, omega, k as u32);
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);