use ark_ec::{CurveGroup, Group, VariableBaseMSM};
use ark_ff::{Field, PrimeField};
use ark_poly::{
    domain::{radix2::Elements, DomainCoeff},
    EvaluationDomain, Radix2EvaluationDomain,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;

#[derive(Copy, Clone, Hash, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize, Debug)]
pub struct GpuDomain(Radix2EvaluationDomain<ark_bn254::Fr>, [ark_bn254::Fr; 64]);

trait GpuDomainCoeff: Sized + DomainCoeff<ark_bn254::Fr> {
    fn fft_in_place(domain: &GpuDomain, input: &mut Vec<Self>);

    fn ifft_in_place(domain: &GpuDomain, input: &mut Vec<Self>);
}

impl<T: DomainCoeff<ark_bn254::Fr>> GpuDomainCoeff for T {
    default fn fft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
        let coefficients_time = start_timer!(|| format!("CPU FFT {}", input.len()));
        domain.0.fft_in_place(input);
        end_timer!(coefficients_time);
    }

    default fn ifft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
        let coefficients_time = start_timer!(|| format!("CPU iFFT {}", input.len()));
        domain.0.ifft_in_place(input);
        end_timer!(coefficients_time);
    }
}

impl EvaluationDomain<ark_bn254::Fr> for GpuDomain {
    fn fft_in_place<T: DomainCoeff<ark_bn254::Fr>>(&self, coeffs: &mut Vec<T>) {
        GpuDomainCoeff::fft_in_place(&self, coeffs)
    }

    fn ifft_in_place<T: DomainCoeff<ark_bn254::Fr>>(&self, evals: &mut Vec<T>) {
        GpuDomainCoeff::ifft_in_place(&self, evals)
    }

    type Elements = Elements<ark_bn254::Fr>;

    fn new(num_coeffs: usize) -> Option<Self> {
        let domain = Radix2EvaluationDomain::new(num_coeffs)?;
        Some(Self::from_domain(domain))
    }

    fn get_coset(&self, offset: ark_bn254::Fr) -> Option<Self> {
        let coset_domain = self.0.get_coset(offset)?;
        Some(Self::from_domain(coset_domain))
    }

    fn compute_size_of_domain(num_coeffs: usize) -> Option<usize> {
        Radix2EvaluationDomain::<ark_bn254::Fr>::compute_size_of_domain(num_coeffs)
    }

    fn size_inv(&self) -> ark_bn254::Fr {
        // Delegate other logic to evaluation domain
        self.0.size_inv()
    }

    fn size(&self) -> usize {
        self.0.size()
    }

    fn log_size_of_group(&self) -> u64 {
        self.0.log_size_of_group()
    }

    fn group_gen(&self) -> ark_bn254::Fr {
        self.0.group_gen()
    }

    fn group_gen_inv(&self) -> ark_bn254::Fr {
        self.0.group_gen_inv()
    }

    fn coset_offset(&self) -> ark_bn254::Fr {
        self.0.coset_offset()
    }

    fn coset_offset_inv(&self) -> ark_bn254::Fr {
        self.0.coset_offset_inv()
    }

    fn coset_offset_pow_size(&self) -> ark_bn254::Fr {
        self.0.coset_offset_pow_size()
    }

    fn elements(&self) -> Self::Elements {
        self.0.elements()
    }
}

pub trait GpuVariableBaseMSM: VariableBaseMSM {
    fn msm_bigint_gpu(
        bases: &[Self::MulBase],
        bigints: &[<<Self as Group>::ScalarField as PrimeField>::BigInt],
    ) -> Self;
}

impl<T: CurveGroup> GpuVariableBaseMSM for T {
    default fn msm_bigint_gpu(
        bases: &[Self::MulBase],
        bigints: &[<<Self as Group>::ScalarField as PrimeField>::BigInt],
    ) -> Self {
        let coefficients_time = start_timer!(|| format!("CPU MSM {}", bases.len()));
        let answer = <Self as VariableBaseMSM>::msm_bigint(bases, bigints);
        end_timer!(coefficients_time);
        answer
    }
}

#[cfg(feature = "cuda")]
mod gpu_impl {
    use super::{GpuDomain, GpuDomainCoeff, GpuVariableBaseMSM};
    use ark_bn254::{self, Fr, G1Projective};
    use ark_ec::short_weierstrass::Projective;
    use rayon::prelude::*;

    impl GpuDomainCoeff for G1Projective {
        fn fft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU FFT {}", input.len()));
            ag_cuda_ec::fft::radix_fft_mt(input, &domain.1[0..32], Some(domain.0.offset)).unwrap();
            end_timer!(coefficients_time);
        }

        fn ifft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU iFFT {}", input.len()));
            ag_cuda_ec::fft::radix_ifft_mt(
                input,
                &domain.1[32..64],
                Some(domain.0.offset_inv),
                domain.0.size_inv,
            )
            .unwrap();
            end_timer!(coefficients_time);
        }
    }

    impl GpuDomainCoeff for Fr {
        fn fft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU FFT {}", input.len()));
            ag_cuda_ec::fft::radix_fft_mt(input, &domain.1[0..32], Some(domain.0.offset)).unwrap();
            end_timer!(coefficients_time);
        }

        fn ifft_in_place(domain: &GpuDomain, input: &mut Vec<Self>) {
            let coefficients_time = start_timer!(|| format!("GPU iFFT {}", input.len()));
            ag_cuda_ec::fft::radix_ifft_mt(
                input,
                &domain.1[32..64],
                Some(domain.0.offset_inv),
                domain.0.size_inv,
            )
            .unwrap();
            end_timer!(coefficients_time);
        }
    }

    impl GpuVariableBaseMSM for Projective<ark_bn254::g1::Config> {
        fn msm_bigint_gpu(
            bases: &[Self::MulBase],
            bigints: &[<<Self as ark_ec::Group>::ScalarField as ark_ff::prelude::PrimeField>::BigInt],
        ) -> Self {
            let coefficients_time = start_timer!(|| format!("GPU MSM {}", bases.len()));
            let answer = ag_cuda_ec::multiexp::multiexp_mt(bases, bigints, 256, 7, false)
                .unwrap()
                .par_iter()
                .cloned()
                .sum();
            end_timer!(coefficients_time);
            answer
        }
    }

    impl GpuVariableBaseMSM for Projective<ark_bn254::g2::Config> {
        fn msm_bigint_gpu(
            bases: &[Self::MulBase],
            bigints: &[<<Self as ark_ec::Group>::ScalarField as ark_ff::prelude::PrimeField>::BigInt],
        ) -> Self {
            let coefficients_time = start_timer!(|| format!("GPU MSM {}", bases.len()));
            let answer = ag_cuda_ec::multiexp::multiexp_mt(bases, bigints, 256, 7, false)
                .unwrap()
                .par_iter()
                .cloned()
                .sum();
            end_timer!(coefficients_time);
            answer
        }
    }
}

impl GpuDomain {
    fn from_domain(domain: Radix2EvaluationDomain<ark_bn254::Fr>) -> Self {
        let mut omega_cache = [ark_bn254::Fr::zero(); 64];
        let omegas = &mut omega_cache[0..32];
        omegas[0] = domain.group_gen;
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }

        let inv_omegas = &mut omega_cache[32..64];
        inv_omegas[0] = domain.group_gen_inv;
        for i in 1..32 {
            inv_omegas[i] = inv_omegas[i - 1].square();
        }
        Self(domain, omega_cache)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use ag_cuda_ec::pairing_suite::Scalar;
    use ag_cuda_ec::test_tools::random_input;
    use ark_ff::{FftField, Field};
    use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
    use ark_std::{rand::thread_rng, Zero};
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_gpu_domain() {
        let mut rng = thread_rng();
        for degree in 4..8 {
            let n = 1 << degree;

            let mut omegas = vec![Scalar::zero(); 32];
            omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
            for i in 1..32 {
                omegas[i] = omegas[i - 1].square();
            }
            let original_coeffs: Vec<ark_bn254::Fr> = random_input(n, &mut rng);
            let mut fft_coeffs = original_coeffs.clone();

            // Perform FFT
            let domain: GpuDomain = EvaluationDomain::new(fft_coeffs.len()).unwrap();
            domain.fft_in_place(&mut fft_coeffs);
            if original_coeffs == fft_coeffs {
                panic!("FFT results do not change");
            }

            // Perform IFFT
            domain.ifft_in_place(&mut fft_coeffs);

            // Compare the IFFT result with the original coefficients
            if original_coeffs != fft_coeffs {
                panic!("FFT and IFFT results do not match the original coefficients");
            }
        }
    }

    #[test]
    fn test_gpu_and_cpu() {
        let mut rng = thread_rng();
        let degree = 15;
        let n = 1 << degree;

        let mut omegas = vec![Scalar::zero(); 32];
        omegas[0] = Scalar::get_root_of_unity(n as u64).unwrap();
        for i in 1..32 {
            omegas[i] = omegas[i - 1].square();
        }
        // 这里类型填ark_bn254::Fr会更快，但是GPU和CPU就没有任何区别了
        let mut gpu_coeffs: Vec<ark_bn254::G1Projective> = random_input(n, &mut rng);
        let mut cpu_coeffs = gpu_coeffs.clone();

        let gpu_domain: GpuDomain = EvaluationDomain::new(gpu_coeffs.len()).unwrap();

        // GPU FFT
        let now = Instant::now();
        gpu_domain.fft_in_place(&mut gpu_coeffs);
        let gpu_dur = now.elapsed().as_millis();
        println!("GPU domain took {}ms.", gpu_dur);

        let cpu_domain: Radix2EvaluationDomain<ark_bn254::Fr> =
            EvaluationDomain::new(cpu_coeffs.len()).unwrap();
        // CPU FFT
        let now = Instant::now();
        cpu_domain.fft_in_place(&mut cpu_coeffs);
        let gpu_dur = now.elapsed().as_millis();
        println!("CPU domain took {}ms.", gpu_dur);
    }
}
