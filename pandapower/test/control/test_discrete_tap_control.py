# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandapower as pp
from pandapower.control import DiscreteTapControl
import pytest
import pandapower.networks as nw
import logging as log

logger = log.getLogger(__name__)


def test_discrete_tap_control_lv():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'lv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  -1

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -2
    # reduce voltage from 1.03 pu to 0.949 pu
    net.ext_grid.vm_pu = 0.949
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1


def test_discrete_tap_control_hv():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'hv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 2
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 0.949
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -1


def test_discrete_tap_control_lv_from_tap_step_percent():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'lv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl.from_tap_step_percent(net, tid=0, side='lv', vm_set_pu=0.98)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  -1

    # check if it changes the lower and upper limits
    net.controller.object.at[0].vm_set_pu = 1
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 1.00725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.99275) < 1e-6
    net.controller.object.at[0].vm_set_pu = 0.98
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 0.98725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.97275) < 1e-6

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -2
    # reduce voltage from 1.03 pu to 0.969 pu
    net.ext_grid.vm_pu = 0.969
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1


def test_discrete_tap_control_hv_from_tap_step_percent():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'hv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl.from_tap_step_percent(net, tid=0, side='lv', vm_set_pu=0.98)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  1

    # check if it changes the lower and upper limits
    net.controller.object.at[0].vm_set_pu = 1
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 1.00725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.99275) < 1e-6
    net.controller.object.at[0].vm_set_pu = 0.98
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 0.98725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.97275) < 1e-6

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 2
    # reduce voltage from 1.03 pu to 0.969 pu
    net.ext_grid.vm_pu = 0.969
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -1


def test_discrete_tap_control_vectorized_lv():
    # --- load system and run power flow
    net = pp.create_empty_network()
    pp.create_buses(net, 6, 110)
    pp.create_buses(net, 5, 20)
    pp.create_ext_grid(net, 0)
    pp.create_lines(net, np.zeros(5), np.arange(1, 6), 10, "243-AL1/39-ST1A 110.0")
    for hv, lv in zip(np.arange(1, 6), np.arange(6,11)):
        pp.create_transformer(net, hv, lv, "63 MVA 110/20 kV")
        pp.create_load(net, lv, 25*(lv-8), 25*(lv-8) * 0.4)
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- run loadflow
    pp.runpp(net)

    net_ref = net.deepcopy()

    # create with individual controllers for comparison
    for tid in net.trafo.index.values:
        # pp.control.ContinuousTapControl(net_ref, tid=tid, side='lv', vm_set_pu=1.02)
        DiscreteTapControl(net_ref, tid=tid, side='lv', vm_lower_pu=1.01, vm_upper_pu=1.03)

    # run control reference
    pp.runpp(net_ref, run_control=True)

    # now create the vectorized version
    DiscreteTapControl(net, tid=net.trafo.index.values, side='lv', vm_lower_pu=1.01, vm_upper_pu=1.03)
    pp.runpp(net, run_control=True)

    assert np.all(net_ref.trafo.tap_pos == net.trafo.tap_pos)


def test_discrete_tap_control_vectorized_hv():
    # --- load system and run power flow
    net = pp.create_empty_network()
    pp.create_buses(net, 6, 110)
    pp.create_buses(net, 5, 20)
    pp.create_ext_grid(net, 0)
    pp.create_lines(net, np.zeros(5), np.arange(1, 6), 10, "243-AL1/39-ST1A 110.0")
    for hv, lv in zip(np.arange(1, 6), np.arange(6,11)):
        pp.create_transformer(net, hv, lv, "63 MVA 110/20 kV")
        pp.create_load(net, lv, 25*(lv-8), 25*(lv-8) * 0.4)
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- run loadflow
    pp.runpp(net)

    net_ref = net.deepcopy()

    # create with individual controllers for comparison
    for tid in net.trafo.index.values:
        # pp.control.ContinuousTapControl(net_ref, tid=tid, side='hv', vm_set_pu=1.02)
        DiscreteTapControl(net_ref, tid=tid, side='hv', vm_lower_pu=1.01, vm_upper_pu=1.03)

    # run control reference
    pp.runpp(net_ref, run_control=True)

    # now create the vectorized version
    DiscreteTapControl(net, tid=net.trafo.index.values, side='hv', vm_lower_pu=1.01, vm_upper_pu=1.03)
    pp.runpp(net, run_control=True)

    assert np.all(net_ref.trafo.tap_pos == net.trafo.tap_pos)



if __name__ == '__main__':
    pytest.main(['-xs', __file__])
