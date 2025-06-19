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
//I NEED TO ADD THE CORRECTED THETA RANGE PARAMETERS TO THE NO CUDA VERSION!
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
  double m_twist_th1 = 0.0; //lower theta bound for twisting
  double m_twist_th2 = M_PI; //upper theta bound for twisting

  nonown_ptr<vector_field<Conf>> E, B, E0, B0;

 public:
  static std::string name() { return "boundary_condition"; }

  boundary_condition(const grid_curv_t<Conf>& grid,
                     const domain_comm<Conf, ExecPolicy>* comm = nullptr)
      : m_grid(grid), m_comm(comm) {}

  void init() override {
    //perturbation of electric field
    sim_env().get_data("Edelta", E);
    sim_env().get_data("E0", E0);
    //perturbation of magnetic field
    sim_env().get_data("Bdelta", B);
    sim_env().get_data("B0", B0);

    sim_env().params().get_value("w0", m_w0);
    sim_env().params().get_value("wave_freq", m_freq);
    sim_env().params().get_value("twist_th1", m_twist_th1);
    sim_env().params().get_value("twist_th2", m_twist_th2);
    //Bp is the initial dipole field
    sim_env().params().get_value("Bp", m_Bp);
    sim_env().params().get_value("num_lambda", m_num_lambda);
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
      //this controls whether or not we are actively twisting the field and driving the wave
      //basically only keep twisting until we complete m_num wavelengths
      if (phase < 2.0 * M_PI * m_num_lambda)
        omega = m_w0 * sin(phase);
      else
        omega = 0.0;
      
      //for now the fast wave gets absorbed at the surface when you get to the bottom so we probably want to run it just with that once to make sure stuff works
      //then will want to change the boundary condition so it actually reflects by making it conducting 
      ExecPolicy<Conf>::launch(
          [omega, Bp, twist_th1, twist_th2] LAMBDA(auto e, auto b, auto e0, auto b0) {
            auto& grid = ExecPolicy<Conf>::grid();
            value_t th_m = (twist_th1 + twist_th2) * 0.5f;
            auto ext = grid.extent();

            ExecPolicy<Conf>::loop(0, grid.dims[1], [&] LAMBDA(auto n1) {
              //I DONT REALLY UNDERSTAND THE DIFFERENCE BETWEEN THE THETA & THETA_S AND THE R & R_S
              value_t theta =
                grid_sph_t<Conf>::theta(grid.template coord<1>(n1, false));
              value_t theta_s =
                grid_sph_t<Conf>::theta(grid.template coord<1>(n1, true));

              //it's just saying if  theta_s is in the range of twist_th1 and twist_th2 go ahead and twist (only twist on the upper hemisphere)
            if (theta_s >= twist_th1 && theta < twist_th2) {
              value_t s = (theta > 0.5f * M_PI ? -1.0f : 1.0f); // enforcing sign of the twist to be hemisphere dependent
              if (theta > 0.5f * M_PI) {
                th_m = M_PI - th_m;
              }
              // For quantities that are not continuous across the surface
              // int n0 = grid.guard[0];
              for (int n0 = 0; n0 < grid.guard[0]; n0++) {
                value_t r =
                    grid_sph_t<Conf>::radius(grid.template coord<0>(n0, false));
              value_t r_s = grid_sph_t<Conf>::radius(grid.coord(0, n0, true));
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                //E_r=0,B_theta=0,B_phi=0
                e[0][idx] = omega * sin(theta_s) * r * b0[1][idx]*
                            square(math::cos(M_PI * (theta_s - th_m) /
                                             (twist_th2 - twist_th1)));
                b[1][idx] = 0.0;
                //we can try imposing B_phi and see what happens or we can impose B_phi and E_theta
                //because its highly magnetized for fast wave we just do E_phi as opposed to E_phi and the other stuff (B_theta and some init U distr.)
                b[2][idx] = 0.0;
              }
              //is the reason we split them up because we have to include the cells directly at the surface if possible?

              // For quantities that are continuous across the surface
              for (int n0 = 0; n0 < grid.guard[0] + 1; n0++) {
                // n0 = grid.guard[0] + 1;
                auto idx = idx_t(index_t<2>(n0, n1), ext);
                //B_r=0,E_theta=0,E_phi=0
                b[0][idx] = 0.0;
                e[1][idx] = -omega *sin(theta_s) * r_s * b0[0][idx]*
                            square(math::cos(M_PI * (theta - th_m) /
                                             (twist_th2 - twist_th1)));
                e[2][idx] = 0.0;
                //you have some e_phi for the fast wave case
                //in our problem we would want to have some e_theta and e_r perturbation
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
