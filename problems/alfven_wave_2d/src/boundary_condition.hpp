/*
 * Copyright (c) 2023 Alex Chen.
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

#pragma once

#include "data/fields.h"
#include "data/rng_states.h"
#include "framework/environment.h"
#include "framework/system.h"
#include "systems/domain_comm.h"
#include "systems/grid_sph.hpp"
#include "utils/nonown_ptr.hpp"
#include "utils/util_functions.h"
#include <memory>

namespace Aperture {

template <typename Conf, template <class> class ExecPolicy>
class boundary_condition : public system_t {
 protected:
  const grid_curv_t<Conf>& m_grid;
  const domain_comm<Conf, ExecPolicy>* m_comm = nullptr;

  double m_w0 = 1.0;
  double m_freq = 5.0;
  double m_Bp = 1.0;
  int m_num_lambda = 4;
  double m_twist_th1 = 0.7;
  double m_twist_th2 = 0.8;
  //double prof_thick = 1.0;

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_curv_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  template <typename Conf> void inject_particles(particle_data_t& ptc, rng_states_t<exec_tags::device>& rng_states,
                 buffer<float>& surface_n, int num_per_cell,
                 typename Conf::value_t weight,
                 const grid_curv_t<Conf>& grid,
                 typename Conf::value_t rpert1,
                 typename Conf::value_t rpert2) {
  surface_n.assign_dev(0.0f);

  auto ptc_num = ptc.number();
  // First measure surface density
  ExecPolicy<Conf>::launch(
      [ptc_num, rpert1, rpert2] LAMBDA(auto ptc, auto surface_n) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        for (auto n : grid_stride_range(0, ptc_num)) {
          auto c = ptc.cell[n];
          if (c == empty_cell) continue;

          auto idx = typename Conf::idx_t(c, ext);
          auto pos = get_pos(idx, ext);
          Scalar th = grid.coord<1>(pos[1], false);
          // if (pos[0] == grid.guard[0] && th > th1 - 0.1 && th < th2 + 0.1) {
          if (pos[0] == grid.guard[0]) {
            auto flag = ptc.flag[n];
            auto sp = get_ptc_type(flag);
            // surface_n[pos[1]] += ptc.weight[n];
            atomicAdd(&surface_n[pos[1]],
                      ptc.weight[n] * math::abs(dev_charges[sp]));
          }
        }
      },
      ptc.get_dev_ptrs(), surface_n.dev_ptr());
  ExecPolicy<Conf>::sync();

  Scalar th1 = math::acos(math::sqrt(1.0f - 1.0f / rpert1));
  Scalar th2 = math::acos(math::sqrt(1.0f - 1.0f / rpert2));
  if (th1 > th2)
    swap_values(th1, th2);

  // Then inject particles
  ExecPolicy<Conf>::launch(
      [ptc_num, weight, th1, th2] LAMBDA(auto ptc, auto surface_n, auto num_inj, auto states) {
        auto& grid = dev_grid<Conf::dim, typename Conf::value_t>();
        auto ext = grid.extent();
        int inj_n0 = grid.guard[0];
        // int id = threadIdx.x + blockIdx.x * blockDim.x;
        rng_t rng(states);
        for (auto n1 :grid_stride_range(grid.guard[1], grid.dims[1] - grid.guard[1])) {
          size_t offset = ptc_num + n1 * num_inj * 2;
          auto pos = index_t<Conf::dim>(inj_n0, n1);
          auto idx = typename Conf::idx_t(pos, ext);
          Scalar theta = grid.template coord<1>(n1, false);
          if (theta > th2 + 0.1f || theta < th1 - 0.1f)
            continue;

          // if (surface_n[pos[1]] >
          //     square(0.5f / grid.delta[1]) * math::sin(theta))
          //   continue;
          for (int i = 0; i < num_inj; i++) {
            auto x2 = rng.uniform<value_t>(state);
            theta = grid.template coord<1>(n1, x2);
            auto p = 0.1 * rng.uniform<value_t>(state);
            ptc.x1[offset + i * 2] = ptc.x1[offset + i * 2 + 1] = 0.5f;
            ptc.x2[offset + i * 2] = ptc.x2[offset + i * 2 + 1] = x2;
            ptc.x3[offset + i * 2] = ptc.x3[offset + i * 2 + 1] = 0.0f;
            ptc.p1[offset + i * 2] = ptc.p1[offset + i * 2 + 1] = p;
            ptc.p2[offset + i * 2] = ptc.p2[offset + i * 2 + 1] = 0.0f;
            ptc.p3[offset + i * 2] = ptc.p3[offset + i * 2 + 1] = 0.0f;
            ptc.E[offset + i * 2] = ptc.E[offset + i * 2 + 1] = math::sqrt(1.0f + p*p);
            ptc.cell[offset + i * 2] = ptc.cell[offset + i * 2 + 1] =
                idx.linear;
            ptc.weight[offset + i * 2] = ptc.weight[offset + i * 2 + 1] =
                weight * math::sin(theta);
            ptc.flag[offset + i * 2] = set_ptc_type_flag(0, PtcType::electron);
            ptc.flag[offset + i * 2 + 1] =
                set_ptc_type_flag(0, PtcType::positron);
          }
        }
      },
      ptc.get_dev_ptrs(), surface_n.dev_ptr(), num_per_cell,rng_states.states().dev_ptr());
  ExecPolicy<Conf>::sync();

  ptc.add_num(num_per_cell * 2 * grid.dims[1]);
}

  void init() override {
    sim_env().get_data("Edelta", E);
    sim_env().get_data("E0", E0);
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("B0", B0);


    sim_env().params().get_value("w0", m_w0);
    sim_env().params().get_value("wave_freq", m_freq);
    sim_env().params().get_value("Bp", m_Bp);
    sim_env().params().get_value("num_lambda", m_num_lambda);
    sim_env().params().get_value("twist_th1", m_twist_th1);
    sim_env().params().get_value("twist_th2", m_twist_th2);
    //sim_env().params().get_value("prof_thick", prof_thick);
  }

  void update(double dt, uint32_t step) override {
    if (m_comm == nullptr || m_comm->domain_info().is_boundary[0]) {
      typedef typename Conf::idx_t idx_t;
      typedef typename Conf::value_t value_t;

      value_t time = sim_env().get_time();
      value_t Bp = m_Bp;
      value_t twist_th1 = m_twist_th1;
      value_t twist_th2 = m_twist_th2;
      value_t omega;
      value_t phase = 2.0 * M_PI * m_freq * time;
      if (phase < 2.0 * M_PI * m_num_lambda)
        omega = m_w0 * sin(phase);// * (1 + tanh((phase / prof_thick) - 3)) / 2;
      else
        omega = 0.0;

      //Logger::print_info("omega: {}, phase: {}, w0: {}",omega,phase,m_w0);

      ExecPolicy<Conf>::launch(
          [omega, Bp, twist_th1, twist_th2,phase] LAMBDA(auto e, auto b, auto e0, auto b0) {
            auto& grid = ExecPolicy<Conf>::grid();
            auto ext = grid.extent();

	    //loop over theta
            ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
              // int n0 = grid.guard[0];

                value_t theta = grid_sph_t<Conf>::theta(grid.coord(1, n1, false));
                value_t th_m = (twist_th1 + twist_th2) * 0.5f;
                value_t sigma = abs(twist_th2 - twist_th1) / 6.0f;
		value_t diff = abs(twist_th2 - twist_th1);

              if (theta >= twist_th1 && theta < twist_th2){
                // For quantities that are not continuous across the surface
		// loop over radius
                for (int n0 = 0; n0 < grid.guard[0]; n0++) {
                  auto idx = idx_t(index_t<2>(n0, n1), ext);

                  b[1][idx] = 0.0; // Fast wave
                  b[2][idx] = 0.0; // Alfven wave
                }

                // For quantities that are continuous across the surface
                for (int n0 = 0; n0 < grid.guard[0] + 1; n0++) {
                  auto idx2 = idx_t(index_t<2>(n0, n1), ext);
		  value_t r = grid_sph_t<Conf>::radius(grid.coord(0,n0,false));
                  b[0][idx2] = 0.0;

		              //dominic's profile
		              e[1][idx2] = -omega*sin(theta)*b0[0][idx2]*pow(cos(M_PI*(theta-th_m)/diff),2.);

                  e[2][idx2] = 0.0; // Fast wave
                }
              }
            });
          },
          E, B, E0, B0);
      ExecPolicy<Conf>::sync();
    }
  }
};

}  // namespace Aperture
