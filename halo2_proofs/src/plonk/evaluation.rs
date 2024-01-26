use crate::multicore;
use crate::plonk::lookup::prover::Committed;
use crate::plonk::permutation::Argument;

use crate::plonk::{lookup, permutation, AdviceQuery, Any, FixedQuery, InstanceQuery, ProvingKey};
use crate::poly::Basis;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        commitment::Params, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff,
        Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use chrono::Date;
use group::prime::PrimeCurve;
use group::{
    ff::{BatchInvert, Field, PrimeField, WithSmallOrderMulGroup},
    Curve,
};
use std::any::TypeId;
use std::convert::TryInto;
use std::fs::{self, File, OpenOptions};
use std::marker::PhantomData;
use std::num::ParseIntError;
use std::slice;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Index, Mul, MulAssign},
};
use std::io::{self, Write};
//use std::collections::HashMap;
use indexmap::IndexMap;
use super::{shuffle, ConstraintSystem, Expression};

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}




/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum  Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple as: ProverExecutionTracesignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
        caclualtions_dict: &mut IndexMap<String, String>,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };


        match self {
            Calculation::Add(a, b) => {
                caclualtions_dict.insert("Add".to_string(), format!("{:?} + {:?} = {:?}", get_value(a), get_value(b), get_value(a) + get_value(b)));
                //writeln!(file, "\t\t{:?} + {:?} = {:?}", get_value(a), get_value(b), get_value(a) + get_value(b)).expect("Failed to write to file");
            },
            Calculation::Sub(a, b) => {
                caclualtions_dict.insert("Sub".to_string(), format!("{:?} - {:?} = {:?}", get_value(a), get_value(b), get_value(a) - get_value(b)));
                //writeln!(file, "\t\t{:?} - {:?} = {:?}", get_value(a), get_value(b), get_value(a) - get_value(b)).expect("Failed to write to file");
            },
            Calculation::Mul(a, b) => {
                caclualtions_dict.insert("Mul".to_string(), format!("{:?} * {:?} = {:?}", get_value(a), get_value(b), get_value(a) * get_value(b)));
                //writeln!(file, "\t\t{:?} * {:?} = {:?}", get_value(a), get_value(b), get_value(a) * get_value(b)).expect("Failed to write to file");
            },
            Calculation::Square(v) => {
                caclualtions_dict.insert("Square".to_string(), format!("{:?} = {:?}", get_value(v), get_value(v).square()));
                //writeln!(file, "\t\tsquare {:?} = {:?}", get_value(v), get_value(v).square()).expect("Failed to write to file");
            },
            Calculation::Double(v) => {
                caclualtions_dict.insert("Double".to_string(), format!("{:?} = {:?}", get_value(v), get_value(v).double()));
                //writeln!(file, "\t\tdouble {:?} = {:?}", get_value(v), get_value(v).double()).expect("Failed to write to file");
            },
            Calculation::Negate(v) => {
                caclualtions_dict.insert("Negate".to_string(), format!("{:?} = {:?}", get_value(v), -get_value(v)));
                //writeln!(file, "\t\tNegate {:?} = {:?}", get_value(v), -get_value(v)).expect("Failed to write to file");
            },
            Calculation::Horner(start_value, parts, factor) => {
                let factor_value = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    let part_value = get_value(part);
                    caclualtions_dict.insert("Horner".to_string(), format!("{:?} = {:?} * {:?} + {:?} = {:?}", value, value, factor_value, part_value, value * factor_value + part_value));
                    //writeln!(file, "\t\t{:?} = {:?} * {:?} + {:?} = {:?}", value, value, factor_value, part_value, value * factor_value + part_value).expect("Failed to write to file");
                    value = value * factor_value + part_value;
                    //println!("{:?} * {:?} + {:?} = {:?}", vc, factor, tv, value);

                }
            },
        
            Calculation::Store(v) => {
                caclualtions_dict.insert("Store".to_string(), format!("{:?} = {:?}", get_value(v), get_value(v)));
                //writeln!(file, "\t\tStore {:?} = {:?}", get_value(v), get_value(v)).expect("Failed to write to file");
            },
        }
    

        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    //let tv: F = get_value(part);
                    //let vc = value.clone();
                    value = value * factor + get_value(part);
                    //println!("{:?} * {:?} + {:?} = {:?}", vc, factor, tv, value);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    ///  Custom gates evalution
    pub custom_gates: GraphEvaluator<C>,
    ///  Lookups evalution
    pub lookups: Vec<GraphEvaluator<C>>,
    ///  Shuffle evalution
    pub shuffles: Vec<GraphEvaluator<C>>,

}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut parts: Vec<ValueSource> = Vec::new();
        for gate in cs.gates.iter() {
            parts.extend(
                gate.polynomials()
                    .iter()
                    .map(|poly| ev.custom_gates.add_expression(poly)),
            );
        }
        ev.custom_gates.add_calculation(Calculation::Horner(
            ValueSource::PreviousValue(),
            parts,
            ValueSource::Y(),
        ));

        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Shuffles
        for shuffle in cs.shuffles.iter() {
            let evaluate_lc = |expressions: &Vec<Expression<_>>, graph: &mut GraphEvaluator<C>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            let mut graph_input = GraphEvaluator::default();
            let compressed_input_coset = evaluate_lc(&shuffle.input_expressions, &mut graph_input);
            let _ = graph_input.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Gamma(),
            ));

            let mut graph_shuffle = GraphEvaluator::default();
            let compressed_shuffle_coset =
                evaluate_lc(&shuffle.shuffle_expressions, &mut graph_shuffle);
            let _ = graph_shuffle.add_calculation(Calculation::Add(
                compressed_shuffle_coset,
                ValueSource::Gamma(),
            ));

            ev.shuffles.push(graph_input);
            ev.shuffles.push(graph_shuffle);
        }

        ev
    }
    
    pub fn write_to_trace_file(&self, message: &str){

        let mut file = match OpenOptions::new().write(true).create(true).append(true).
        open("C:\\Users\\pw\\projects\\dist-zkml\\trace\\eval_h_trace.txt") {
            Ok(file) => file,
                Err(e) =>todo!()}; 
             writeln!(file, "{}", &message).expect("Failed to write to file");
    }


    /// Evaluate h poly
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        challenges: &[C::ScalarExt],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        shuffles: &[Vec<shuffle::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],

    ) -> Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let fixed = &pk.fixed_cosets[..];
        let extended_omega = domain.get_extended_omega();
        let isize = size as i32;
        let one = C::ScalarExt::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;
        


        // Calculate the advice and instance cosets
        let advice: Vec<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>> = advice_polys
            .iter()
            .map(|advice_polys| {
                advice_polys
                    .iter()
                    .map(|poly| domain.coeff_to_extended(poly.clone()))
                    .collect()
            })
            .collect();
        
        #[cfg(feature = "trace")]
        {
            let eval_h_path = "C:\\Users\\pw\\projects\\dist-zkml\\trace\\eval_h_trace.txt";

            if fs::metadata(&eval_h_path).is_ok() {
                let _ = fs::remove_file(&eval_h_path);
            }
        }

        let instance: Vec<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>> = instance_polys
            .iter()
            .map(|instance_polys| {
                instance_polys
                    .iter()
                    .map(|poly| domain.coeff_to_extended(poly.clone()))
                    .collect()
            })
            .collect();
        
      
        let mut values = domain.empty_extended();
       
        #[cfg(feature = "trace")]
        {  
            //self.write_to_trace_file(&format!("{:?}",("advice_coset".to_string() : format!("{:?}",advice))) 
            self.write_to_trace_file(&format!("{:?}", (
                "advice_coset",
                format!(":{:?}", advice)
            )));
            self.write_to_trace_file(&format!("{:?}", (
                "instance_coset",
                format!(":{:?}", instance)
            )));
            self.write_to_trace_file(&format!("{:?}", (
                "initial_values",
                format!(":{:?}", values)
            )));
        }
        // Core expression evaluations
        //let num_threads = multicore::current_num_threads();
        let num_threads = 1;
        for ((((advice, instance), lookups), shuffles), permutation) in advice
            .iter()
            .zip(instance.iter())
            .zip(lookups.iter())
            .zip(shuffles.iter())
            .zip(permutations.iter())
        {
            // Custom gates
            multicore::scope(|scope| {

                let chunk_size = (size + num_threads - 1) / num_threads;
                for (thread_idx, values) in values.chunks_mut(chunk_size).enumerate() {
                    let start = thread_idx * chunk_size;
                    scope.spawn(move |_| {
                        
                        let mut eval_data = self.custom_gates.instance();
                        for (i, value) in values.iter_mut().enumerate() {
                            let idx = start + i;
                            // Assuming `self.custom_gates.evaluate` returns `EvaluationResult<C>`
                            let evaluation_result = self.custom_gates.evaluate(
                            &mut eval_data,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &value.clone(),
                            idx,
                            rot_scale,
                            isize,
                        );
                                      // Set *value to the result_scalar field of the EvaluationResult
                        *value = evaluation_result.result_scalar;
                        #[cfg(feature = "trace")]
                        {  
                            self.write_to_trace_file(&format!("{:?}", (
                                format!("cg_eval_{:?}", idx),
                                format!("{:?}", evaluation_result.evaluations_dict)
                            )));
                        }
                        }
                    });
                }
            });

            #[cfg(feature = "trace")]
            {  
                self.write_to_trace_file(&format!("{:?}", (
                    "values_after_cg",
                    format!(":{:?}", values)
                )));
            }


    
            // Permutations
            let sets = &permutation.sets;
            if !sets.is_empty() {
                let blinding_factors = pk.vk.cs.blinding_factors();
                let last_rotation = Rotation(-((blinding_factors + 1) as i32));
                let chunk_len = pk.vk.cs.degree() - 2;
                let delta_start = beta * &C::Scalar::ZETA;

                let first_set = sets.first().unwrap();
                let last_set = sets.last().unwrap();

                // Permutation constraints
                parallelize(&mut values, |values, start| {
                    let mut beta_term = extended_omega.pow_vartime(&[start as u64, 0, 0, 0]);
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;
                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);

                        // Enforce only for the first set.
                        // l_0(X) * (1 - z_0(X)) = 0
                        *value = *value * y
                            + ((one - first_set.permutation_product_coset[idx]) * l0[idx]);
                        // Enforce only for the last set.
                        // l_last(X) * (z_l(X)^2 - z_l(X)) = 0
                        *value = *value * y
                            + ((last_set.permutation_product_coset[idx]
                                * last_set.permutation_product_coset[idx]
                                - last_set.permutation_product_coset[idx])
                                * l_last[idx]);
                        // Except for the first set, enforce.
                        // l_0(X) * (z_i(X) - z_{i-1}(\omega^(last) X)) = 0
                        for (set_idx, set) in sets.iter().enumerate() {
                            if set_idx != 0 {
                                *value = *value * y
                                    + ((set.permutation_product_coset[idx]
                                        - permutation.sets[set_idx - 1].permutation_product_coset
                                            [r_last])
                                        * l0[idx]);
                            }
                        }
                        // And for all the sets we enforce:
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z_i(\omega X) \prod_j (p(X) + \beta s_j(X) + \gamma)
                        // - z_i(X) \prod_j (p(X) + \delta^j \beta X + \gamma)
                        // )
                        let mut current_delta = delta_start * beta_term;
                        for ((set, columns), cosets) in sets
                            .iter()
                            .zip(p.columns.chunks(chunk_len))
                            .zip(pk.permutation.cosets.chunks(chunk_len))
                        {
                            let mut left = set.permutation_product_coset[r_next];
                            for (values, permutation) in columns
                                .iter()
                                .map(|&column| match column.column_type() {
                                    Any::Advice(_) => &advice[column.index()],
                                    Any::Fixed => &fixed[column.index()],
                                    Any::Instance => &instance[column.index()],
                                })
                                .zip(cosets.iter())
                            {
                                left *= values[idx] + beta * permutation[idx] + gamma;
                            }

                            let mut right = set.permutation_product_coset[idx];
                            for values in columns.iter().map(|&column| match column.column_type() {
                                Any::Advice(_) => &advice[column.index()],
                                Any::Fixed => &fixed[column.index()],
                                Any::Instance => &instance[column.index()],
                            }) {
                                right *= values[idx] + current_delta + gamma;
                                current_delta *= &C::Scalar::DELTA;
                            }

                            *value = *value * y + ((left - right) * l_active_row[idx]);
                        }
                        beta_term *= &extended_omega;
                    }
                });
            }

            #[cfg(feature = "trace")]
            {  
                self.write_to_trace_file(&format!("{:?}", (
                    "values_after_permuttaions",
                    format!(":{:?}", values)
                )));
            }

            // Lookups
            for (n, lookup) in lookups.iter().enumerate() {
                // Polynomials required for this lookup.
                // Calculated here so these only have to be kept in memory for the short time
                // they are actually needed.
                let product_coset = pk.vk.domain.coeff_to_extended(lookup.product_poly.clone());
                let permuted_input_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_input_poly.clone());
                let permuted_table_coset = pk
                    .vk
                    .domain
                    .coeff_to_extended(lookup.permuted_table_poly.clone());

                // Lookup constraints
                parallelize(&mut values, |values, start| {

                    let lookup_evaluator = &self.lookups[n];
                    let mut eval_data = lookup_evaluator.instance();
                    for (i, value) in values.iter_mut().enumerate() {
                        let idx = start + i;

                        let table_value_result = lookup_evaluator.evaluate(
                            &mut eval_data,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );
                        // Set table_value to the result_scalar field of the EvaluationResult
                        let table_value = table_value_result.result_scalar;

                        #[cfg(feature = "trace")]
                        {  
                            self.write_to_trace_file(&format!("{:?}", (
                                format!("lookup_eval_{:?}", idx),
                                format!("{:?}", table_value_result.evaluations_dict)
                            )));
                        }

                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                        let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);

                        let a_minus_s = permuted_input_coset[idx] - permuted_table_coset[idx];
                        // l_0(X) * (1 - z(X)) = 0
                        *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                        // l_last(X) * (z(X)^2 - z(X)) = 0
                        *value = *value * y
                            + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                * l_last[idx]);
                        // (1 - (l_last(X) + l_blind(X))) * (
                        //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                        //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta)
                        //          (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                        // ) = 0
                        *value = *value * y
                            + ((product_coset[r_next]
                                * (permuted_input_coset[idx] + beta)
                                * (permuted_table_coset[idx] + gamma)
                                - product_coset[idx] * table_value)
                                * l_active_row[idx]);
                        // Check that the first values in the permuted input expression and permuted
                        // fixed expression are the same.
                        // l_0(X) * (a'(X) - s'(X)) = 0
                        *value = *value * y + (a_minus_s * l0[idx]);
                        // Check that each value in the permuted lookup input expression is either
                        // equal to the value above it, or the value at the same index in the
                        // permuted table expression.
                        // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
                        *value = *value * y
                            + (a_minus_s
                                * (permuted_input_coset[idx] - permuted_input_coset[r_prev])
                                * l_active_row[idx]);
                    }
                });
            }

            #[cfg(feature = "trace")]
            {  
                self.write_to_trace_file(&format!("{:?}", (
                    "values_after_lookup",
                    format!(":{:?}", values)
                )));
            }

            // Shuffle constraints
            for (n, shuffle) in shuffles.iter().enumerate() {
                let product_coset = pk.vk.domain.coeff_to_extended(shuffle.product_poly.clone());

                // Shuffle constraints
                parallelize(&mut values, |values, start| {
                    let input_evaluator = &self.shuffles[2 * n];
                    let shuffle_evaluator = &self.shuffles[2 * n + 1];
                    let mut eval_data_input = shuffle_evaluator.instance();
                    let mut eval_data_shuffle = shuffle_evaluator.instance();
                  
                    for (i, value) in values.iter_mut().enumerate() {

                        let idx = start + i;

                        let input_value_result = input_evaluator.evaluate(
                            &mut eval_data_input,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );
                        
                        // Set input_value to the result_scalar field of the EvaluationResult
                        let input_value = input_value_result.result_scalar;
                        // For shuffle_value
                        let shuffle_value_result = shuffle_evaluator.evaluate(
                            &mut eval_data_shuffle,
                            fixed,
                            advice,
                            instance,
                            challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &C::ScalarExt::ZERO,
                            idx,
                            rot_scale,
                            isize,
                        );

                        // Set shuffle_value to the result_scalar field of the EvaluationResult
                        let shuffle_value = shuffle_value_result.result_scalar;

                        let r_next = get_rotation_idx(idx, 1, rot_scale, isize);

                        // l_0(X) * (1 - z(X)) = 0
                        *value = *value * y + ((one - product_coset[idx]) * l0[idx]);
                        // l_last(X) * (z(X)^2 - z(X)) = 0
                        *value = *value * y
                            + ((product_coset[idx] * product_coset[idx] - product_coset[idx])
                                * l_last[idx]);
                        // (1 - (l_last(X) + l_blind(X))) * (z(\omega X) (s(X) + \gamma) - z(X) (a(X) + \gamma)) = 0
                        *value = *value * y
                            + l_active_row[idx]
                                * (product_coset[r_next] * shuffle_value
                                    - product_coset[idx] * input_value)
                    }
                });
            }

            #[cfg(feature = "trace")]
            {  
                self.write_to_trace_file(&format!("{:?}", (
                    "values_final",
                    format!(":{:?}", values)
                )));
            }
        }
        values
    }
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<C::ScalarExt>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }


    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
        ) ->  EvaluationResult<C>{


        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize);
        }

        let mut eval_dict: IndexMap<String, String> = IndexMap::new();
        
        // All calculations, with cached intermediate results
        for (iteration_count, calc) in self.calculations.iter().enumerate() {
            
            let mut caclualtions_dict: IndexMap<String, String> = IndexMap::new();

            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
                &mut caclualtions_dict, // Pass mutable reference
            );


            eval_dict.insert(format!("{:?}", calc.clone()), format!("{:?}",caclualtions_dict));
        }
          // Return the result of the last calculation (if any)

          
        if let Some(calc) = self.calculations.last() {
            eval_dict.insert("Result".to_string(), format!("{:?}",data.intermediates[calc.target]));
             EvaluationResult {result_scalar:data.intermediates[calc.target],evaluations_dict: eval_dict }}
            else 
        {
            eval_dict.insert("Result".to_string(), format!("{:?}",C::ScalarExt::ZERO));
            EvaluationResult {result_scalar:C::ScalarExt::ZERO,evaluations_dict: eval_dict }}

        }
    }
    


/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[Polynomial<F, B>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}


pub struct EvaluationResult<C: CurveAffine> {
    pub result_scalar: C::ScalarExt,
    pub evaluations_dict: IndexMap<String, String>, // Add other fields as needed
}
