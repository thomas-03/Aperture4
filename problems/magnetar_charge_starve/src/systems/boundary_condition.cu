/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "boundary_condition.h"
#include "core/math.hpp"
#include "framework/config.h"
#include "systems/grid.h"
#include "utils/kernel_helper.hpp"
#include "utils/util_functions.h"

namespace Aperture {
/*
struct wpert_cart_t {
  float tp_start, tp_end, nT, dw0, y_start, y_end, q_e;

  //deleted the y_start and y_end parameters
  HD_INLINE wpert_cart_t(float tp_s, float tp_e, float nT_, float dw0_, float qe)
      : tp_start(tp_s), tp_end(tp_e), nT(nT_), dw0(dw0_), q_e(qe) {}

  HD_INLINE Scalar operator()(Scalar t, Scalar x, Scalar y) {
    //took out the condition checking if y is in the right region and replaced it with a condition checking the phase at the stellar surface
    //I kept the t_start and t_end parameters to allow for a time window if we want to use it
    value_t phase = 2.0 * M_PI * m_freq * time;
    if (t >= tp_start && t <= tp_end && phase< 2.0 * M_PI * m_num_lambda) {
        //Scalar omega = dw0*math::sin(phase);
        Scalar omega =
          dw0 *
          math::sin((t - tp_start) * 2.0f * M_PI * nT / (tp_end - tp_start)) *
          math::sin(M_PI * (y - y_start) / (y_end - y_start));
          // math::sin((t - tp_start) * 2.0f * M_PI * nT / (tp_end - tp_start));
      return omega;
    } else {
      return 0.0;
    }
  }

  HD_INLINE Scalar j_x(Scalar t, Scalar x, Scalar y, Scalar theta) {
    return 0.0;
  }

  HD_INLINE Scalar j_y(Scalar t, Scalar x, Scalar y, Scalar theta) {
    return 0.0;
  }
};
*/

//for our spherical coordinates we need to use a different perturbation function
struct wpert_sph_t {
  float twist_th1, twist_th2;
  float tp_start, tp_end, nT, dw0;

  HD_INLINE wpert_sph_t(float th1, float th2, float tp_s, float tp_e, float nT_,
                        float dw0_)
      : twist_th1(th1),
        twist_th2(th2),
        tp_start(tp_s),
        tp_end(tp_e),
        nT(nT_),
        dw0(dw0_) {}

  HD_INLINE Scalar operator()(Scalar t, Scalar r, Scalar th) {
    if (twist_th1 > twist_th2)
      swap_values(twist_th1, twist_th2);

    Scalar mu = (twist_th1 + twist_th2) / 2.0;
    Scalar s = (mu - twist_th1) / 3.0;
    if (t >= tp_start && t <= tp_end && th >= twist_th1 && th <= twist_th2) {
      Scalar omega =
          dw0 * math::exp(-0.5 * square((th - mu) / s)) *
          math::sin((t - tp_start) * 2.0 * M_PI * nT / (tp_end - tp_start));
      return omega;
    } else {
      return 0.0;
    }
  }
};


HOST_DEVICE Scalar
pml_sigma(Scalar x, Scalar xh, Scalar pmlscale, Scalar sig0) {
  if (x > xh)
    return sig0 * square((x - xh) / pmlscale);
  else
    return 0.0;
}

template <typename Conf>
void
inject_particles(particle_data_t& ptc, curand_states_t& rand_states,
                 buffer<float>& surface_ne, buffer<float>& surface_np,
                 int num_per_cell, typename Conf::value_t weight,
                 const grid_t<Conf>& grid, const wpert_sph_t& wpert,
                 int multiplicity) {
  surface_ne.assign_dev(0.0f);
  surface_np.assign_dev(0.0f);

  auto ptc_num = ptc.number();
//they have density such that it is meant to allow for charge starvation in their very specific case
//for us we would probably just have goldreich julien density 

  // First measure surface density
  kernel_launch(
      [ptc_num] __device__(auto ptc, auto surface_ne, auto surface_np) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, ptc_num)) {
          auto c = ptc.cell[n];
          if (c == empty_cell) continue;

          auto idx = typename Conf::idx_t(c, ext);
          auto pos = idx.get_pos();
          if (pos[0] == grid.guard[0]) {
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);

            if (sp == 0)
              atomicAdd(&surface_ne[pos[1]],
                        ptc.weight[n] * math::abs(dev_charges[sp]));
            else if (sp == 1)
              atomicAdd(&surface_np[pos[1]],
                        ptc.weight[n] * math::abs(dev_charges[sp]));
          }
        }
      },
      ptc.get_dev_ptrs(), surface_ne.dev_ptr(), surface_np.dev_ptr());
  CudaSafeCall(cudaDeviceSynchronize());

  if (twist_th1 > twist_th2)
    swap_values(twist_th1, twist_th2);

  // Then inject particles
  kernel_launch(
      [ptc_num, weight, th1, th2] __device__(auto ptc, auto surface_ne, auto surface_np,
                                   auto num_inj, auto states) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        int inj_n0 = grid.guard[0];
        //int id = threadIdx.x + blockIdx.x * blockDim.x;
        cuda_rng_t rng(&states[id]);
        for (auto n1 :
             grid_stride_range(grid.guard[1], grid.dims[1] - grid.guard[1])) {
          size_t offset = ptc_num + n1 * num_inj * 2;
          auto pos = index_t<Conf::dim>(inj_n0, n1);

          //auto cell_x2 = grid.template coord<1>(n1, false);
          //if (cell_x2 < 0.2 || cell_x2 > 4.8) continue;
          auto idx = typename Conf::idx_t(pos, ext);

          //if theta isn't within the range of th1 and th2, skip
          Scalar theta = grid.template coord<1>(n1, false);
          if (theta > twist_th2 + 0.1f || theta < twist_th1 - 0.1f)
            continue;

          //if the surface density is too low, skip. for our case I don't think we need this.
          /*if (std::min(surface_ne[pos[1]], surface_np[pos[1]]) >
              square(1.0f / grid.delta[0]))
            continue;*/
        
          for (int i = 0; i < num_inj; i++) {
            //inject particles randomly & uniformly (wrt phi) in the cell ??
            auto x2 = rng.uniform<float>();
            theta = grid.template coord<1>(n1, x2);
            //inject particles with some initial random momentum ??
            auto p = 0.1 * rng.uniform<float>();
            //for the alfven_wave case this is 0.5f meanwhile for the alfven_charge_starve case this is 1.0f
            ptc.x1[offset + i * 2] = ptc.x1[offset + i * 2 + 1] = 1.0f;
            ptc.x2[offset + i * 2] = ptc.x2[offset + i * 2 + 1] = x2;
            ptc.x3[offset + i * 2] = ptc.x3[offset + i * 2 + 1] = 0.0f;
            //for the alfven_wave case this is p meanwhile for the alfven_charge_starve case this is 0.0f
            ptc.p1[offset + i * 2] = ptc.p1[offset + i * 2 + 1] = 0.0f;
            ptc.p2[offset + i * 2] = ptc.p2[offset + i * 2 + 1] = 0.0f;
            ptc.p3[offset + i * 2] = ptc.p3[offset + i * 2 + 1] = 0.0f;
            //for the alfven_wave case this is math::sqrt(1.0f - p*p) meanwhile for the alfven_charge_starve case this is 1.0f
            ptc.E[offset + i * 2] = ptc.E[offset + i * 2 + 1] = 1.0f;
            ptc.cell[offset + i * 2] = ptc.cell[offset + i * 2 + 1] =
                idx.linear;
            ptc.weight[offset + i * 2] = ptc.weight[offset + i * 2 + 1] =
                weight* math::sin(theta);
            ptc.flag[offset + i * 2] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + i * 2 + 1] =
                set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), surface_ne.dev_ptr(), surface_np.dev_ptr(),
      num_per_cell, rand_states.states());
  CudaSafeCall(cudaDeviceSynchronize());

  ptc.add_num(num_per_cell * 2 * grid.dims[1]);
}


//I THINK THIS FUNCTION IS JUST MAKING SOME BASIC BOUNDARY CONDITION TEMPLATE
template <typename Conf>
boundary_condition<Conf>::boundary_condition(sim_environment& env,
                                             const grid_t<Conf>& grid)
    : system_t(env), m_grid(grid) {
  using multi_array_t = typename Conf::multi_array_t;
  m_env.params().get_value("damping_length", m_damping_length);
  m_env.params().get_value("pmllen", m_pmllen);
  m_env.params().get_value("sigpml", m_sigpml);

  m_prev_E1 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_E2 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_E3 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B1 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B2 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);
  m_prev_B3 = std::make_unique<multi_array_t>(
      extent(m_damping_length, m_grid.dims[1]), MemType::device_only);

  m_prev_E1->assign_dev(0.0f);
  m_prev_E2->assign_dev(0.0f);
  m_prev_E3->assign_dev(0.0f);
  m_prev_B1->assign_dev(0.0f);
  m_prev_B2->assign_dev(0.0f);
  m_prev_B3->assign_dev(0.0f);

  m_prev_E.set_memtype(MemType::host_device);
  m_prev_B.set_memtype(MemType::host_device);
  m_prev_E.resize(3);
  m_prev_B.resize(3);
  m_prev_E[0] = m_prev_E1->dev_ptr();
  m_prev_E[1] = m_prev_E2->dev_ptr();
  m_prev_E[2] = m_prev_E3->dev_ptr();
  m_prev_B[0] = m_prev_B1->dev_ptr();
  m_prev_B[1] = m_prev_B2->dev_ptr();
  m_prev_B[2] = m_prev_B3->dev_ptr();
  m_prev_E.copy_to_device();
  m_prev_B.copy_to_device();
}

//INITIALIZE THE FIELD AND PERTURBATION FROM THE CONFIG FILE    
template <typename Conf>
void
boundary_condition<Conf>::init() {
  m_env.get_data("Edelta", &E);
  m_env.get_data("E0", &E0);
  m_env.get_data("Bdelta", &B);
  m_env.get_data("B0", &B0);
  m_env.get_data("rand_states", &rand_states);
  m_env.get_data("particles", &ptc);

  m_env.params().get_value("th_twist1", m_twist_th1);
  m_env.params().get_value("th_twist2", m_twist_th2);
  m_env.params().get_value("tp_start", m_tp_start);
  m_env.params().get_value("tp_end", m_tp_end);
  m_env.params().get_value("nT", m_nT);
  m_env.params().get_value("dw0", m_dw0);
  m_env.params().get_value("q_e", m_qe);
  m_env.params().get_value("damping_coef", m_damping_coef);
  m_env.params().get_value("muB", m_muB);

  m_surface_ne.set_memtype(MemType::host_device);
  m_surface_ne.resize(m_grid.dims[1]);
  m_surface_np.set_memtype(MemType::host_device);
  m_surface_np.resize(m_grid.dims[1]);

  //added from the alfven_wave case
  auto rho0 = m_env().params().get_as<double>("rho0", 100.0);
  int mult = m_env().params().get_as<int64_t>("multiplicity", 10);
  value_t q_e = m_env().params().get_as<double>("q_e", 1.0);
  m_weight = rho0 / mult / q_e * 10.0f;

  m_surface_n.resize(m_grid.dims[1]);
}

//UPDATE THE BOUNDARY CONDITION
template <typename Conf>
void
boundary_condition<Conf>::update(double dt, uint32_t step) {
  typedef typename Conf::idx_t idx_t;
  typedef typename Conf::value_t value_t;
  //NEED TO MAKE SURE THAT BASICALLY WE ONLY TWIST WITHIN A PARTICULAR THETA RANGE

  value_t time = m_env.get_time();
  wpert_sph_t wpert(m_twist_th1, m_twist_th2, m_tp_start, m_tp_end, m_nT, m_dw0);
  // wpert_cart_t wpert(m_tp_start, m_tp_end, m_nT, m_dw0, m_qe);


  // Apply twist on the stellar surface
   kernel_launch(
       [time] __device__(auto e, auto b, auto e0, auto b0, auto wpert) {
         auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
         auto ext = grid.extent();

         value_t th_m = (m_twist_th1 + m_twist_th2) * 0.5f;

         for (auto n1 : grid_stride_range(0, grid.dims[1])) {
            value_t theta =
              grid_sph_t<Conf>::theta(grid.template coord<1>(n1, false));
            value_t theta_s =
              grid_sph_t<Conf>::theta(grid.template coord<1>(n1, true));


           // For quantities that are not continuous across the surface
           for (int n0 = 0; n0 < grid.guard[0]; n0++) {
             auto idx = idx_t(index_t<2>(n0, n1), ext);
             value_t r =
                grid_sph_t<Conf>::radius(grid.template coord<0>(n0, false));
             value_t omega = wpert(time, r, theta_s);
             e[0][idx] = omega * sin(theta_s) * r * b0[1][idx]*
                            square(math::cos(M_PI * (theta_s - th_m) /
                                             (m_twist_th2- m_twist_th1)));;
             b[1][idx] = 0.0;
             b[2][idx] = 0.0;
           }

           // For quantities that are continuous across the surface
           for (int n0 = 0; n0 < grid.guard[0] + 1; n0++) {
             auto idx = idx_t(index_t<2>(n0, n1), ext);
             value_t r_s =
                grid_sph_t<Conf>::radius(grid.template coord<0>(n0, true));
             value_t omega = wpert(time, r_s, theta);
             b[0][idx] = 0.0;
             e[1][idx] = -omega *sin(theta) * r_s * b0[0][idx]*
                            square(math::cos(M_PI * (theta_s - th_m) /
                                             (m_twist_th2 - m_twist_th1)));;
             e[2][idx] = 0.0;
           }
         }
       },
       E->get_ptrs(), B->get_ptrs(), E0->get_ptrs(), B0->get_ptrs(), wpert);
   CudaSafeCall(cudaDeviceSynchronize());
   CudaCheckError();


  //for them this is at the outer border of the box, for us it would be when our resolution gets low far out
  //this is meant to stop non-physical waves from propagating back (specifically high frequency noise plasma waves)
  // Apply damping boundary condition on the other side
  kernel_launch(
      [] __device__(auto e, auto b, auto prev_e, auto prev_b, auto damping_length,
                    auto damping_coef) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        auto ext_damping = extent(damping_length, grid.dims[1]);
        for (auto n1 : grid_stride_range(0, grid.dims[1])) {
          for (int i = 0; i < damping_length; i++) {
            int n0 = grid.dims[0] - damping_length + i;
            auto idx = idx_t(index_t<2>(n0, n1), ext);
            value_t lambda =
                1.0f - damping_coef * cube((value_t)i / (damping_length - 1));
            e[0][idx] *= lambda;
            e[1][idx] *= lambda;
            e[2][idx] *= lambda;
            b[1][idx] *= lambda;
            b[2][idx] *= lambda;
          }
        }
      },
      E->get_ptrs(), B->get_ptrs(), m_prev_E.dev_ptr(), m_prev_B.dev_ptr(),
      m_damping_length, m_damping_coef);
  CudaSafeCall(cudaDeviceSynchronize());
  CudaCheckError();

  // Store the current values of the field for the damping boundary of next time step
  // kernel_launch([] __device__(auto e, auto prev_e, auto b, auto prev_b,
  //                             auto damping_length) {
  //     auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
  //     auto ext = grid.extent();
  //     auto ext_damping = extent(damping_length, grid.dims[1]);
  //     for (auto n1 : grid_stride_range(0, grid.dims[1])) {
  //       auto n0_start = grid.dims[0] - damping_length;
  //       for (int n0 = n0_start; n0 < grid.dims[0]; n0++) {
  //         auto pos = index(n0, n1);
  //         auto idx = idx_t(pos, ext);
  //         auto idx_damping = idx_t(index(n0 - n0_start, n1), ext_damping);

  //         prev_e[0][idx_damping] = e[0][idx];
  //         prev_e[1][idx_damping] = e[1][idx];
  //         prev_e[2][idx_damping] = e[2][idx];
  //         prev_b[0][idx_damping] = b[0][idx];
  //         prev_b[1][idx_damping] = b[1][idx];
  //         prev_b[2][idx_damping] = b[2][idx];
  //       }
  //     }
  //   }, E->get_ptrs(), m_prev_E.dev_ptr(), B->get_ptrs(), m_prev_B.dev_ptr(), m_damping_length);
  // CudaSafeCall(cudaDeviceSynchronize());
  // CudaCheckError();

  // Inject particles --> is uncommented in the alfven_wave case, but not in the alfven_charge_starve case
  // if (step % 1 == 0 && time > m_tp_start && time < m_tp_end) {
  //   inject_particles<Conf>(*ptc, *rand_states, m_surface_ne, m_surface_np, 1,
  //                          0.2, m_grid, wpert, 1);
  // }

  // Apply damping to particle momenta in the region behind the alfven wave
  //this is taken from the alfven_charge_starve case but we should define our own unique damping radius if we want to use it
  value_t r_damp = time * m_muB - 0.5f;
  value_t ptc_damping_factor = 0.99f;
  auto num = ptc->number();
  kernel_launch(
      [x_damp, num, ptc_damping_factor] __device__(auto ptc) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, num)) {
          auto cell = ptc.cell[n];
          auto idx = Conf::idx(cell, ext);
          auto pos = idx.get_pos();

          auto r = grid.template coord<0>(pos[0], ptc.x1[n]);
          if (r < r_damp) {
            ptc.p1[n] *= ptc_damping_factor;
            ptc.p2[n] *= ptc_damping_factor;
            ptc.p3[n] *= ptc_damping_factor;
            ptc.E[n] = math::sqrt(1.0f + ptc.p1[n] * ptc.p1[n] + ptc.p2[n] * ptc.p2[n]
                                  + ptc.p3[n] * ptc.p3[n]);
          }
        }
      }, ptc->get_dev_ptrs());
}

template class boundary_condition<Config<2>>;

}  // namespace Aperture
