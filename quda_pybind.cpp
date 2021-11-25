#include <stdexcept>
#include <type_traits>

#include <mpi4py/mpi4py.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h> // for std::complex automatic conversion
#include <pybind11/numpy.h> // for numpy array automatic conversion

// Quda headers
#include "quda.h"
#include "mpi_comm_handle.h"
#include "communicator_quda.h"

// Binding headers
#include "cfunc_pybind.hpp"
#include "utility.hpp"
#include "enum_quda_pybind.hpp"
#include "qio_field_pybind.hpp"
#include "communicator_quda_pybind.hpp"
#include "lattice_field_pybind.hpp"

// Declaration
void init_quda_pybind(pybind11::module_ &m);

PYBIND11_MODULE(quda, m) 
{
  // initialize mpi4py's C-API
  if (import_mpi4py() < 0) {
      // mpi4py calls the Python C API
      // we let pybind11 give us the detailed traceback
      throw pybind11::error_already_set();
  }

  m.doc() = R"pbdoc(
        Pybind11-mpi4py example plugin
        ------------------------------
        .. currentmodule:: _pb11mpi
        .. autosummary::
           :toctree: _generate
           example
    )pbdoc";

    // Initialize some common C functions
    init_cfunc_pybind(m);

    // Initialize this first to define types
    init_enum_quda_pybind(m);

    // Initialize quda.h
    init_quda_pybind(m);

    // Initialize qio_field.h 
    init_qio_field_pybind(m);

    // Initialize communicator_quda.h 
    init_communicator_quda_pybind(m);

    // Initialize lattice_field.h
    init_lattice_field_pybind(m);
}

void init_quda_pybind(pybind11::module_ &m) 
{
    // Wrapper around initComms in tests/utils/host_utilities.cpp
    // to initiate communications either with QMP or MPI
    // TODO: Maybe separte into multiple files to expose finer details?
    m.def(
        "init_comms",
        [](const std::vector<int> &comm_dims) {
        int argc = 1;
        char **argv = (char **)malloc(sizeof(char *));
        int n_dims = 4;

        #if defined(QMP_COMMS)
        QMP_thread_level_t tl;
        QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

        // Make sure the QMP logical ordering matches QUDA's
        int map[] = {0, 1, 2, 3}; // [t, z, y, x] row-major with x fastest moving
        QMP_declare_logical_topology_map(comm_dims.data(), n_dims, map, n_dims);
        int rank = QMP_get_node_number();

        #elif defined(MPI_COMMS)
        MPI_Init(&argc, &argv);
        int rank = QMP_get_node_number();
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif

        //QudaCommsMap func = lex_rank_from_coords_x; // TODO: IMPLEMENT THIS!!
        initCommsGridQuda(n_dims, comm_dims.data(), NULL, NULL);

        // Set the random number seed
        srand(17 * rank + 137);

        free(argv);
        initQuda(rank); 
        },
        R"pbdoc(
        Initilialize the topology and library.
        )pbdoc"
    );

    m.def("endQuda", &endQuda,
        R"pbdoc(
        Finalize the Quda library.
        )pbdoc"
    );

    m.def(
        "finalize_comms",
        []() 
        {   
            comm_finalize();
            #if defined(QMP_COMMS)
            QMP_finalize_msg_passing();
            #elif defined(MPI_COMMS)
            MPI_Finalize();
            #endif
        },
        R"pbdoc(
        Finalize all communications.
        )pbdoc"
    );

{ // QudaGaugeParam file:quda.h 
      pybind11::class_<QudaGaugeParam, std::unique_ptr<QudaGaugeParam>> 
          cl(m, "QudaGaugeParam", "Parameters having to do with the gauge field "
          "or the\n interpretation of the gauge field by various Dirac operators");
      cl.def( pybind11::init( [](){ return new QudaGaugeParam(); } ) );
      cl.def_readwrite("struct_size", &QudaGaugeParam::struct_size);
      cl.def_readwrite("location", &QudaGaugeParam::location);
    
      cl.def_property("X", 
        [](pybind11::object& obj) {
          // 4d lattice
          return attr_getter<QudaGaugeParam, int, 4>(obj, &QudaGaugeParam::X);
        },
        [](pybind11::object &obj, const pybind11::array_t<int> &a) {
          return attr_setter<QudaGaugeParam, int, 4>(obj, &QudaGaugeParam::X, a);
        }
      );
      
      cl.def_readwrite("anisotropy", &QudaGaugeParam::anisotropy);
      cl.def_readwrite("tadpole_coeff", &QudaGaugeParam::tadpole_coeff);
      cl.def_readwrite("scale", &QudaGaugeParam::scale);
      cl.def_readwrite("type", &QudaGaugeParam::type);
      cl.def_readwrite("gauge_order", &QudaGaugeParam::gauge_order);
      cl.def_readwrite("t_boundary", &QudaGaugeParam::t_boundary);
      cl.def_readwrite("cpu_prec", &QudaGaugeParam::cpu_prec);
      cl.def_readwrite("cuda_prec", &QudaGaugeParam::cuda_prec);
      cl.def_readwrite("reconstruct", &QudaGaugeParam::reconstruct);
      cl.def_readwrite("cuda_prec_sloppy", &QudaGaugeParam::cuda_prec_sloppy);
      cl.def_readwrite("reconstruct_sloppy", &QudaGaugeParam::reconstruct_sloppy);
      cl.def_readwrite("cuda_prec_refinement_sloppy", &QudaGaugeParam::cuda_prec_refinement_sloppy);
      cl.def_readwrite("reconstruct_refinement_sloppy", &QudaGaugeParam::reconstruct_refinement_sloppy);
      cl.def_readwrite("cuda_prec_precondition", &QudaGaugeParam::cuda_prec_precondition);
      cl.def_readwrite("reconstruct_precondition", &QudaGaugeParam::reconstruct_precondition);
      cl.def_readwrite("cuda_prec_eigensolver", &QudaGaugeParam::cuda_prec_eigensolver);
      cl.def_readwrite("reconstruct_eigensolver", &QudaGaugeParam::reconstruct_eigensolver);
      cl.def_readwrite("gauge_fix", &QudaGaugeParam::gauge_fix);
      cl.def_readwrite("ga_pad", &QudaGaugeParam::ga_pad);
      cl.def_readwrite("site_ga_pad", &QudaGaugeParam::site_ga_pad);
      cl.def_readwrite("staple_pad", &QudaGaugeParam::staple_pad);
      cl.def_readwrite("llfat_ga_pad", &QudaGaugeParam::llfat_ga_pad);
      cl.def_readwrite("mom_ga_pad", &QudaGaugeParam::mom_ga_pad);
      cl.def_readwrite("staggered_phase_type", &QudaGaugeParam::staggered_phase_type);
      cl.def_readwrite("staggered_phase_applied", &QudaGaugeParam::staggered_phase_applied);
      cl.def_readwrite("i_mu", &QudaGaugeParam::i_mu);
      cl.def_readwrite("overlap", &QudaGaugeParam::overlap);
      cl.def_readwrite("overwrite_mom", &QudaGaugeParam::overwrite_mom);
      cl.def_readwrite("use_resident_gauge", &QudaGaugeParam::use_resident_gauge);
      cl.def_readwrite("use_resident_mom", &QudaGaugeParam::use_resident_mom);
      cl.def_readwrite("make_resident_gauge", &QudaGaugeParam::make_resident_gauge);
      cl.def_readwrite("make_resident_mom", &QudaGaugeParam::make_resident_mom);
      cl.def_readwrite("return_result_gauge", &QudaGaugeParam::return_result_gauge);
      cl.def_readwrite("return_result_mom", &QudaGaugeParam::return_result_mom);
      cl.def_readwrite("gauge_offset", &QudaGaugeParam::gauge_offset);
      cl.def_readwrite("mom_offset", &QudaGaugeParam::mom_offset);
      cl.def_readwrite("site_size", &QudaGaugeParam::site_size);
}

{ // QudaInvertParam file:quda.h

      pybind11::class_<QudaInvertParam, std::unique_ptr<QudaInvertParam>> 
          cl(m, "QudaInvertParam", "Parameters relating to the solver and the "
                "choice of Dirac operator.");
      cl.def( pybind11::init( [](){ return new QudaInvertParam(); } ) );
      cl.def_readwrite("struct_size", &QudaInvertParam::struct_size);
      cl.def_readwrite("input_location", &QudaInvertParam::input_location);
      cl.def_readwrite("output_location", &QudaInvertParam::output_location);
      cl.def_readwrite("dslash_type", &QudaInvertParam::dslash_type);
      cl.def_readwrite("inv_type", &QudaInvertParam::inv_type);
      cl.def_readwrite("mass", &QudaInvertParam::mass);
      cl.def_readwrite("kappa", &QudaInvertParam::kappa);
      cl.def_readwrite("m5", &QudaInvertParam::m5);
      cl.def_readwrite("Ls", &QudaInvertParam::Ls);
      
      // Think about what to do later. C complex type seems to be 
      // hard to deal with.
      /*
      cl.def_property("b_5", 
        [](pybind11::object& obj) {
          //QudaInvertParam& o = obj.cast<QudaInvertParam&>(); 
          //typedef decltype(o.b_5[0]) double_complex;
          //std::complex<double> cppb5[QUDA_MAX_DWF_LS]; 
          //for (int i=0; i<QUDA_MAX_DWF_LS; i++) o.b_5[i] = cppb5[i];
          //std::complex<double> cppb5;
          //double_complex cb5 = o.b_5[0];
          //return pybind11::array_t<std::complex<double>>{QUDA_MAX_DWF_LS, cppb5, obj};
          throw pybind11::attribute_error("b_5 attribute not implemented in Python");
        },
        [](pybind11::object &obj, const pybind11::array_t<double_complex> &a) {
          throw pybind11::attribute_error("b_5 attribute not implemented in Python");
          //return attr_setter<QudaInvertParam, double_complex>(obj, 
          //                                  &QudaInvertParam::b_5, QUDA_MAX_DWF_LS, a);
        }
      );
      cl.def_property("c_5", 
        [](pybind11::object& obj) {
          //QudaInvertParam& o = obj.cast<QudaInvertParam&>(); 
          //typedef decltype(o.b_5[0]) double_complex;
          //std::complex<double> cppb5[QUDA_MAX_DWF_LS]; 
          //for (int i=0; i<QUDA_MAX_DWF_LS; i++) o.b_5[i] = cppb5[i];
          //std::complex<double> cppb5;
          //double_complex cb5 = o.b_5[0];
          //return pybind11::array_t<std::complex<double>>{QUDA_MAX_DWF_LS, cppb5, obj};
          throw pybind11::attribute_error("c_5 attribute not implemented in Python");
        },
        [](pybind11::object &obj, const pybind11::array_t<double_complex> &a) {
          throw pybind11::attribute_error("c_5 attribute not implemented in Python");
          //return attr_setter<QudaInvertParam, double_complex>(obj, 
          //                                  &QudaInvertParam::b_5, QUDA_MAX_DWF_LS, a);
        }
      );
      */
      
      //cl.def_readwrite("c_5", &QudaInvertParam::c_5); // there are problems binding complex_double
      cl.def_readwrite("eofa_shift", &QudaInvertParam::eofa_shift);
      cl.def_readwrite("eofa_pm", &QudaInvertParam::eofa_pm);
      cl.def_readwrite("mq1", &QudaInvertParam::mq1);
      cl.def_readwrite("mq2", &QudaInvertParam::mq2);
      cl.def_readwrite("mq3", &QudaInvertParam::mq3);
      cl.def_readwrite("mu", &QudaInvertParam::mu);
      cl.def_readwrite("epsilon", &QudaInvertParam::epsilon);
      cl.def_readwrite("twist_flavor", &QudaInvertParam::twist_flavor);
      cl.def_readwrite("laplace3D", &QudaInvertParam::laplace3D);
      cl.def_readwrite("tol", &QudaInvertParam::tol);
      cl.def_readwrite("tol_restart", &QudaInvertParam::tol_restart);
      cl.def_readwrite("tol_hq", &QudaInvertParam::tol_hq);
      cl.def_readwrite("compute_true_res", &QudaInvertParam::compute_true_res);
      cl.def_readwrite("true_res", &QudaInvertParam::true_res);
      cl.def_readwrite("true_res_hq", &QudaInvertParam::true_res_hq);
      cl.def_readwrite("maxiter", &QudaInvertParam::maxiter);
      cl.def_readwrite("reliable_delta", &QudaInvertParam::reliable_delta);
      cl.def_readwrite("reliable_delta_refinement", &QudaInvertParam::reliable_delta_refinement);
      cl.def_readwrite("use_alternative_reliable", &QudaInvertParam::use_alternative_reliable);
      cl.def_readwrite("use_sloppy_partial_accumulator", &QudaInvertParam::use_sloppy_partial_accumulator);
      cl.def_readwrite("solution_accumulator_pipeline", &QudaInvertParam::solution_accumulator_pipeline);
      cl.def_readwrite("max_res_increase", &QudaInvertParam::max_res_increase);
      cl.def_readwrite("max_res_increase_total", &QudaInvertParam::max_res_increase_total);
      cl.def_readwrite("max_hq_res_increase", &QudaInvertParam::max_hq_res_increase);
      cl.def_readwrite("max_hq_res_restart_total", &QudaInvertParam::max_hq_res_restart_total);
      cl.def_readwrite("heavy_quark_check", &QudaInvertParam::heavy_quark_check);
      cl.def_readwrite("pipeline", &QudaInvertParam::pipeline);
      cl.def_readwrite("num_offset", &QudaInvertParam::num_offset);
      cl.def_readwrite("num_src", &QudaInvertParam::num_src);
      cl.def_readwrite("num_src_per_sub_partition", &QudaInvertParam::num_src_per_sub_partition);

      cl.def_property("split_grid", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, int, QUDA_MAX_DIM>(obj, 
                        &QudaInvertParam::split_grid);
        },
        [](pybind11::object &obj, const pybind11::array_t<int> &a) {
          return attr_setter<QudaInvertParam, int, QUDA_MAX_DIM>(obj, 
                        &QudaInvertParam::split_grid, a);
        }
      );

      cl.def_readwrite("overlap", &QudaInvertParam::overlap);

      cl.def_property("offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::offset, a);
        }
      );

      cl.def_property("tol_offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::tol_offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::tol_offset, a);
        }
      );

      cl.def_property("tol_hq_offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::tol_hq_offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::tol_hq_offset, a);
        }
      );

      cl.def_property("true_res_offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::true_res_offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::true_res_offset, a);
        }
      );

      cl.def_property("iter_res_offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::iter_res_offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::iter_res_offset, a);
        }
      );

      cl.def_property("true_res_hq_offset", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::true_res_hq_offset);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::true_res_hq_offset, a);
        }
      );

      cl.def_property("residue", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::residue);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, QUDA_MAX_MULTI_SHIFT>(obj, 
                        &QudaInvertParam::residue, a);
        }
      );

      cl.def_readwrite("compute_action", &QudaInvertParam::compute_action);

      cl.def_property("action", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, 2>(obj, &QudaInvertParam::action);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, 2>(obj, &QudaInvertParam::action, a);
        }
      );

      cl.def_readwrite("solution_type", &QudaInvertParam::solution_type);
      cl.def_readwrite("solve_type", &QudaInvertParam::solve_type);
      cl.def_readwrite("matpc_type", &QudaInvertParam::matpc_type);
      cl.def_readwrite("dagger", &QudaInvertParam::dagger);
      cl.def_readwrite("mass_normalization", &QudaInvertParam::mass_normalization);
      cl.def_readwrite("solver_normalization", &QudaInvertParam::solver_normalization);
      cl.def_readwrite("preserve_source", &QudaInvertParam::preserve_source);
      cl.def_readwrite("cpu_prec", &QudaInvertParam::cpu_prec);
      cl.def_readwrite("cuda_prec", &QudaInvertParam::cuda_prec);
      cl.def_readwrite("cuda_prec_sloppy", &QudaInvertParam::cuda_prec_sloppy);
      cl.def_readwrite("cuda_prec_refinement_sloppy", &QudaInvertParam::cuda_prec_refinement_sloppy);
      cl.def_readwrite("cuda_prec_precondition", &QudaInvertParam::cuda_prec_precondition);
      cl.def_readwrite("cuda_prec_eigensolver", &QudaInvertParam::cuda_prec_eigensolver);
      cl.def_readwrite("dirac_order", &QudaInvertParam::dirac_order);
      cl.def_readwrite("gamma_basis", &QudaInvertParam::gamma_basis);
      cl.def_readwrite("clover_location", &QudaInvertParam::clover_location);
      cl.def_readwrite("clover_cpu_prec", &QudaInvertParam::clover_cpu_prec);
      cl.def_readwrite("clover_cuda_prec", &QudaInvertParam::clover_cuda_prec);
      cl.def_readwrite("clover_cuda_prec_sloppy", &QudaInvertParam::clover_cuda_prec_sloppy);
      cl.def_readwrite("clover_cuda_prec_refinement_sloppy", &QudaInvertParam::clover_cuda_prec_refinement_sloppy);
      cl.def_readwrite("clover_cuda_prec_precondition", &QudaInvertParam::clover_cuda_prec_precondition);
      cl.def_readwrite("clover_cuda_prec_eigensolver", &QudaInvertParam::clover_cuda_prec_eigensolver);
      cl.def_readwrite("clover_order", &QudaInvertParam::clover_order);
      cl.def_readwrite("use_init_guess", &QudaInvertParam::use_init_guess);
      cl.def_readwrite("clover_csw", &QudaInvertParam::clover_csw);
      cl.def_readwrite("clover_coeff", &QudaInvertParam::clover_coeff);
      cl.def_readwrite("clover_rho", &QudaInvertParam::clover_rho);
      cl.def_readwrite("compute_clover_trlog", &QudaInvertParam::compute_clover_trlog);

      cl.def_property("trlogA", 
        [](pybind11::object& obj) {
          return attr_getter<QudaInvertParam, double, 2>(obj, &QudaInvertParam::trlogA);
        },
        [](pybind11::object &obj, const pybind11::array_t<double> &a) {
          return attr_setter<QudaInvertParam, double, 2>(obj, &QudaInvertParam::trlogA, a);
        }
      );

      cl.def_readwrite("compute_clover", &QudaInvertParam::compute_clover);
      cl.def_readwrite("compute_clover_inverse", &QudaInvertParam::compute_clover_inverse);
      cl.def_readwrite("return_clover", &QudaInvertParam::return_clover);
      cl.def_readwrite("return_clover_inverse", &QudaInvertParam::return_clover_inverse);
      cl.def_readwrite("verbosity", &QudaInvertParam::verbosity);
      cl.def_readwrite("sp_pad", &QudaInvertParam::sp_pad);
      cl.def_readwrite("cl_pad", &QudaInvertParam::cl_pad);
      cl.def_readwrite("iter", &QudaInvertParam::iter);
      cl.def_readwrite("gflops", &QudaInvertParam::gflops);
      cl.def_readwrite("secs", &QudaInvertParam::secs);
      cl.def_readwrite("tune", &QudaInvertParam::tune);
      cl.def_readwrite("Nsteps", &QudaInvertParam::Nsteps);
      cl.def_readwrite("gcrNkrylov", &QudaInvertParam::gcrNkrylov);
      cl.def_readwrite("inv_type_precondition", &QudaInvertParam::inv_type_precondition);
      cl.def_readwrite("deflate", &QudaInvertParam::deflate);
      cl.def_readwrite("dslash_type_precondition", &QudaInvertParam::dslash_type_precondition);
      cl.def_readwrite("verbosity_precondition", &QudaInvertParam::verbosity_precondition);
      cl.def_readwrite("tol_precondition", &QudaInvertParam::tol_precondition);
      cl.def_readwrite("maxiter_precondition", &QudaInvertParam::maxiter_precondition);
      cl.def_readwrite("omega", &QudaInvertParam::omega);
      cl.def_readwrite("ca_basis", &QudaInvertParam::ca_basis);
      cl.def_readwrite("ca_lambda_min", &QudaInvertParam::ca_lambda_min);
      cl.def_readwrite("ca_lambda_max", &QudaInvertParam::ca_lambda_max);
      cl.def_readwrite("precondition_cycle", &QudaInvertParam::precondition_cycle);
      cl.def_readwrite("schwarz_type", &QudaInvertParam::schwarz_type);
      cl.def_readwrite("residual_type", &QudaInvertParam::residual_type);
      cl.def_readwrite("cuda_prec_ritz", &QudaInvertParam::cuda_prec_ritz);
      cl.def_readwrite("n_ev", &QudaInvertParam::n_ev);
      cl.def_readwrite("max_search_dim", &QudaInvertParam::max_search_dim);
      cl.def_readwrite("rhs_idx", &QudaInvertParam::rhs_idx);
      cl.def_readwrite("deflation_grid", &QudaInvertParam::deflation_grid);
      cl.def_readwrite("eigenval_tol", &QudaInvertParam::eigenval_tol);
      cl.def_readwrite("eigcg_max_restarts", &QudaInvertParam::eigcg_max_restarts);
      cl.def_readwrite("max_restart_num", &QudaInvertParam::max_restart_num);
      cl.def_readwrite("inc_tol", &QudaInvertParam::inc_tol);
      cl.def_readwrite("make_resident_solution", &QudaInvertParam::make_resident_solution);
      cl.def_readwrite("use_resident_solution", &QudaInvertParam::use_resident_solution);
      cl.def_readwrite("chrono_make_resident", &QudaInvertParam::chrono_make_resident);
      cl.def_readwrite("chrono_replace_last", &QudaInvertParam::chrono_replace_last);
      cl.def_readwrite("chrono_use_resident", &QudaInvertParam::chrono_use_resident);
      cl.def_readwrite("chrono_max_dim", &QudaInvertParam::chrono_max_dim);
      cl.def_readwrite("chrono_index", &QudaInvertParam::chrono_index);
      cl.def_readwrite("chrono_precision", &QudaInvertParam::chrono_precision);
      cl.def_readwrite("extlib_type", &QudaInvertParam::extlib_type);
      cl.def_readwrite("native_blas_lapack", &QudaInvertParam::native_blas_lapack);
  }

    // newQudaGaugeParam() file:quda.h
    m.def("newQudaGaugeParam", (QudaGaugeParam(*)()) &newQudaGaugeParam, 
          "A new QudaGaugeParam should always be initialized immediately\n "
          "after it's defined (and prior to explicitly setting its members)\n "
          "using this function.  Typical usage is as follows:\n\n   "
          "QudaGaugeParam gauge_param = newQudaGaugeParam();\n\n"
          "C++: newQudaGaugeParam() --> struct QudaGaugeParam_s");

    // newQudaInvertParam() file:quda.h 
    m.def("newQudaInvertParam", (QudaInvertParam (*)()) &newQudaInvertParam, 
          "A new QudaInvertParam should always be initialized immediately\n "
          "after it's defined (and prior to explicitly setting its members)\n "
          "using this function.  Typical usage is as follows:\n\n   "
          "QudaInvertParam invert_param = newQudaInvertParam();\n\n"
          "C++: newQudaInvertParam() --> struct QudaInvertParam_s");

    m.def("loadGaugeQuda", 
        [] (pybind11::array &gauge, QudaGaugeParam* param)
        {
            pybind11::buffer_info buf = gauge.request();
            void *tmp[4]; // because QIO does not like *gauge directly for some reasons
            auto local_volume = param->X[0] * param->X[1] * param->X[2] * param->X[3];
            int gauge_site_size = 18; // (real + imag) * 3 * 3 for QDP ordering

            init_gauge_pointer_array(tmp, buf.ptr, param->cpu_prec, 
                                     local_volume, gauge_site_size);
            loadGaugeQuda(tmp, param);
        }
    );

    m.def("freeGaugeQuda", &freeGaugeQuda);

    m.def("plaqQuda",
        [] (pybind11::array_t<double> plaq)
        {
            pybind11::buffer_info buf = plaq.request();

            // Safety checks
            if (buf.ndim != 1 || buf.shape[0] != 3)
                throw std::runtime_error("plaq must be a 1D array");
            if (buf.shape[0] != 3)
                throw std::runtime_error("plaq must be a 1D array of size 3");

            double *tmp = (double*) buf.ptr;
            plaqQuda(tmp);
        }
    );
}; // end init_quda_pybind