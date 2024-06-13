use crate::r1cs_to_qap::*;
use ark_bn254::G1Projective;
use ark_ff::{FftField, Field};
use ark_poly::{
    domain::{self, radix2::Elements, DomainCoeff},
    EvaluationDomain, Radix2EvaluationDomain,
};
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSystemRef, Result as R1CSResult, SynthesisError,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Copy, Clone, Hash, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize, Debug)]
pub struct GpuDomain(Radix2EvaluationDomain<ark_bn254::Fr>, [ark_bn254::Fr; 64]);

pub trait GpuDomainCoeff: Sized + DomainCoeff<ark_bn254::Fr> {
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

impl GpuDomainCoeff for ark_bn254::Fr {
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

impl EvaluationDomain<ark_bn254::Fr> for GpuDomain {
    fn size_inv(&self) -> ark_bn254::Fr {
        // Delegate other logic to evaluation domain
        self.0.size_inv()
    }

    fn fft_in_place<T: DomainCoeff<ark_bn254::Fr>>(&self, coeffs: &mut Vec<T>) {
        GpuDomainCoeff::fft_in_place(&self, coeffs)
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
        todo!()
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

    fn ifft_in_place<T: DomainCoeff<ark_bn254::Fr>>(&self, evals: &mut Vec<T>) {
        GpuDomainCoeff::ifft_in_place(&self, evals)
    }

    fn elements(&self) -> Self::Elements {
        self.0.elements()
    }
}

pub struct GpuLibsnarkReduction;
impl R1CSToQAP<ark_bn254::Fr, GpuDomain> for GpuLibsnarkReduction {
    fn instance_map_with_evaluation(
        cs: ConstraintSystemRef<ark_bn254::Fr>,
        t: &ark_bn254::Fr,
    ) -> Result<
        (
            Vec<ark_bn254::Fr>,
            Vec<ark_bn254::Fr>,
            Vec<ark_bn254::Fr>,
            ark_bn254::Fr,
            usize,
            usize,
        ),
        SynthesisError,
    > {
        let matrices = cs.to_matrices().unwrap();
        let domain_size = cs.num_constraints() + cs.num_instance_variables();
        let domain = GpuDomain::new(domain_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();

        let zt = domain.evaluate_vanishing_polynomial(*t);

        // Evaluate all Lagrange polynomials
        let coefficients_time = start_timer!(|| "Evaluate Lagrange coefficients");
        let u = domain.evaluate_all_lagrange_coefficients(*t);
        end_timer!(coefficients_time);

        let qap_num_variables = (cs.num_instance_variables() - 1) + cs.num_witness_variables();

        let mut a = vec![ark_bn254::Fr::zero(); qap_num_variables + 1];
        let mut b = vec![ark_bn254::Fr::zero(); qap_num_variables + 1];
        let mut c = vec![ark_bn254::Fr::zero(); qap_num_variables + 1];

        {
            let start = 0;
            let end = cs.num_instance_variables();
            let num_constraints = cs.num_constraints();
            a[start..end].copy_from_slice(&u[(start + num_constraints)..(end + num_constraints)]);
        }

        for (i, u_i) in u.iter().enumerate().take(cs.num_constraints()) {
            for &(ref coeff, index) in &matrices.a[i] {
                a[index] += &(*u_i * coeff);
            }
            for &(ref coeff, index) in &matrices.b[i] {
                b[index] += &(*u_i * coeff);
            }
            for &(ref coeff, index) in &matrices.c[i] {
                c[index] += &(*u_i * coeff);
            }
        }

        Ok((a, b, c, zt, qap_num_variables, domain_size))
    }

    fn witness_map_from_matrices(
        matrices: &ConstraintMatrices<ark_bn254::Fr>,
        num_inputs: usize,
        num_constraints: usize,
        full_assignment: &[ark_bn254::Fr],
    ) -> R1CSResult<Vec<ark_bn254::Fr>> {
        let domain = GpuDomain::new(num_constraints + num_inputs)
            .ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let zero = ark_bn254::Fr::zero();

        let mut a = vec![zero; domain_size];
        let mut b = vec![zero; domain_size];

        cfg_iter_mut!(a[..num_constraints])
            .zip(cfg_iter_mut!(b[..num_constraints]))
            .zip(cfg_iter!(&matrices.a))
            .zip(cfg_iter!(&matrices.b))
            .for_each(|(((a, b), at_i), bt_i)| {
                *a = evaluate_constraint(&at_i, &full_assignment);
                *b = evaluate_constraint(&bt_i, &full_assignment);
            });

        {
            let start = num_constraints;
            let end = start + num_inputs;
            a[start..end].clone_from_slice(&full_assignment[..num_inputs]);
        }

        domain.ifft_in_place(&mut a);
        domain.ifft_in_place(&mut b);

        let coset_domain = domain.get_coset(ark_bn254::Fr::GENERATOR).unwrap();

        coset_domain.fft_in_place(&mut a);
        coset_domain.fft_in_place(&mut b);

        let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
        drop(a);
        drop(b);

        let mut c = vec![zero; domain_size];
        cfg_iter_mut!(c[..num_constraints])
            .enumerate()
            .for_each(|(i, c)| {
                *c = evaluate_constraint(&matrices.c[i], &full_assignment);
            });

        domain.ifft_in_place(&mut c);
        coset_domain.fft_in_place(&mut c);

        let vanishing_polynomial_over_coset = domain
            .evaluate_vanishing_polynomial(ark_bn254::Fr::GENERATOR)
            .inverse()
            .unwrap();
        cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
            *ab_i -= &c_i;
            *ab_i *= &vanishing_polynomial_over_coset;
        });

        coset_domain.ifft_in_place(&mut ab);

        Ok(ab)
    }

    fn h_query_scalars(
        max_power: usize,
        t: ark_bn254::Fr,
        zt: ark_bn254::Fr,
        delta_inverse: ark_bn254::Fr,
    ) -> Result<Vec<ark_bn254::Fr>, SynthesisError> {
        let scalars = cfg_into_iter!(0..max_power)
            .map(|i| zt * &delta_inverse * &t.pow([i as u64]))
            .collect::<Vec<_>>();
        Ok(scalars)
    }
}

#[cfg(test)]
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
        let mut gpu_coeffs: Vec<G1Projective> = random_input(n, &mut rng);
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
