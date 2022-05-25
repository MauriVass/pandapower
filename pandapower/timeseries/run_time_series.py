# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import tempfile
from collections.abc import Iterable
import tqdm

import pandapower as pp
from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.control.run_control import ControllerNotConverged, prepare_run_ctrl, \
    run_control, NetCalculationNotConverged
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter

import numpy as np
from pandapower.toolbox import write_to_net

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)


def init_default_outputwriter(net, time_steps, **kwargs):
    """
    Initializes the output writer. If output_writer is None, default output_writer is created

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list) - time steps to be calculated

    """
    output_writer = kwargs.get("output_writer", None)
    if output_writer is not None:
        # write the output_writer to net
        logger.warning("deprecated: output_writer should not be given to run_timeseries(). "
                       "This overwrites the stored one in net.output_writer.")
        net.output_writer.iat[0, 0] = output_writer
    if "output_writer" not in net or net.output_writer.iat[0, 0] is None:
        # create a default output writer for this net
        ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir())
        logger.info("No output writer specified. Using default:")
        logger.info(ow)


def init_output_writer(net, time_steps):
    # init output writer before time series calculation
    output_writer = net.output_writer.iat[0, 0]
    output_writer.time_steps = time_steps
    output_writer.init_all(net)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    the code is mentioned in : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def controller_not_converged(time_step, ts_variables):
    logger.error('ControllerNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ControllerNotConverged


def pf_not_converged(time_step, ts_variables):
    logger.error('CalculationNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ts_variables['errors'][0]


def control_time_step(controller_order, time_step):
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.time_step(net, time_step)

def finalize_step(controller_order, time_step):
    for levelorder in controller_order:
        for ctrl, net in levelorder:
            ctrl.finalize_step(net, time_step)


def output_writer_routine(net, time_step, pf_converged, ctrl_converged, recycle_options):
    output_writer = net["output_writer"].iat[0, 0]
    # update time step for output writer
    output_writer.time_step = time_step
    # save
    output_writer.save_results(net, time_step, pf_converged=pf_converged, ctrl_converged=ctrl_converged,
                               recycle_options=recycle_options)


def _call_output_writer(net, time_step, pf_converged, ctrl_converged, ts_variables):
    output_writer_routine(net, time_step, pf_converged, ctrl_converged, ts_variables['recycle_options'])

def run_time_step(net, time_step, ts_variables, run_control_fct=run_control, output_writer_fct=_call_output_writer,
                  **kwargs):
    """
    Time Series step function
    Is called to run the PANDAPOWER AC power flows with the timeseries module

    INPUT:
        **net** - The pandapower format network

        **time_step** (int) - time_step to be calculated

        **ts_variables** (dict) - contains settings for controller and time series simulation. See init_time_series()
    """
    ctrl_converged = True
    pf_converged = True

    #Calculate the PF for the current time step
    # run time step function for each controller
    control_time_step(ts_variables['controller_order'], time_step)
    try:
        # calls controller init, control steps and run function (runpp usually is called in here)
        #run_control.py
        run_control_fct(net, ctrl_variables=ts_variables, **kwargs)
    except ControllerNotConverged:
        ctrl_converged = False
        # If controller did not converge do some stuff
        controller_not_converged(time_step, ts_variables)
    except ts_variables['errors']:
        # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
        pf_converged = False
        pf_not_converged(time_step, ts_variables)

    output_writer_fct(net, time_step, pf_converged, ctrl_converged, ts_variables)

    finalize_step(ts_variables['controller_order'], time_step)


#Mauri
def get_state(ts_variables, time_step):
    loadp_controller = ts_variables['controller_order'][0][0][0] #First index is the level, the second the tuple (controller,net), the third the controller
    loadq_controller = ts_variables['controller_order'][0][1][0]
    genp_controller = ts_variables['controller_order'][0][2][0]

    state_load_p = loadp_controller.data_source.get_time_step_value(time_step=time_step,
                                                       profile_name=loadp_controller.profile_name,
                                                       scale_factor=loadp_controller.scale_factor)
    state_load_q = loadq_controller.data_source.get_time_step_value(time_step=time_step,
                                                       profile_name=loadq_controller.profile_name,
                                                       scale_factor=loadq_controller.scale_factor)
    state_gen_p = genp_controller.data_source.get_time_step_value(time_step=time_step,
                                                       profile_name=genp_controller.profile_name,
                                                       scale_factor=genp_controller.scale_factor)
    state = np.concatenate([state_load_p,state_load_q,state_gen_p])
    return state

def volatge_violation_function(vm_pus):
	violations = []
	a = 0.7
	m = 1
	tollerance = 0.05
	for vm in vm_pus:
		diff = 1-vm
		q = a*(tollerance**2) - m * tollerance
		if(np.abs(diff)<tollerance):
			v = a*(np.abs(diff)**2)
		elif(diff>0):
			v = m * diff + q
		else:
			v = -m * diff + q
		violations.append(v)
	return violations

def run_time_step_rl(net, time_step, rl, ts_variables, run_control_fct=run_control, output_writer_fct=_call_output_writer,
                  **kwargs):
    """
    Time Series step function
    Is called to run the PANDAPOWER AC power flows with the timeseries module

    INPUT:
        **net** - The pandapower format network

        **time_step** (int) - time_step to be calculated

        **ts_variables** (dict) - contains settings for controller and time series simulation. See init_time_series()
    """
    ctrl_converged = True
    pf_converged = True

    #Mauri
    rlagent = rl['agent']
    save = rl['save']
    train = rl['train']
    # if('vm_pu_labels' in rl.keys()):
    vm_pu_labels = rl['vm_pu_labels']

    #RL agent action must be here to affectively apply changes
    genp_controller = ts_variables['controller_order'][0][2][0]
    max_timestep = genp_controller.data_source.get_time_steps_len()
    if(time_step+1<max_timestep):
        ### --- Get state --- ###
        state = get_state(ts_variables,time_step)
        ### --- Choose action --- ###
        action = rlagent.choose_action(state, explore=train)
        action = action.numpy()
        action_len = int(len(action)/2)
        action_p = action[:action_len]
        action_q = action[action_len:]
        # print(action.shape)
        # print(action_p.shape)
        # print(action_q.shape)

        #Apply the chosen action -> curtail the generators at time step t+1
        next_gen_p = genp_controller.data_source.get_time_step_value(time_step=time_step+1,
                                                       profile_name=genp_controller.profile_name,
                                                       scale_factor=1)
        updated_values = next_gen_p * (1-action_p)
        genp_controller.data_source.set_time_step_value(time_step=time_step+1, profile_name=genp_controller.profile_name, values=updated_values)

        genq_controller = ts_variables['controller_order'][0][3][0]
        updated_values =  action_q
        genq_controller.data_source.set_time_step_value(time_step=time_step+1, profile_name=genq_controller.profile_name, values=updated_values)

        ### --- Run PF --- ###
        #Calculate the PF for the next time step t+1
        # run time step function for each controller
        control_time_step(ts_variables['controller_order'], time_step+1)
        try:
            # calls controller init, control steps and run function (runpp usually is called in here)
            # print('A Net values (before pf)', np.sum(net.sgen.p_mw), np.sum(net.sgen.p_mw)/np.sum(next_gen_p))
            # next_gen_p = genp_controller.data_source.get_time_step_value(time_step=time_step,
            #                                                profile_name=genp_controller.profile_name,
            #                                                scale_factor=1)
            # print('A Really changed (?)', np.sum(next_gen_p))
            run_control_fct(net, ctrl_variables=ts_variables, **kwargs)
            # print('A Net values (after pf)', np.sum(net.sgen.p_mw), np.sum(net.sgen.p_mw)/np.sum(next_gen_p))
        except ControllerNotConverged:
            ctrl_converged = False
            # If controller did not converge do some stuff
            # controller_not_converged(time_step, ts_variables)
        except ts_variables['errors']:
            # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
            pf_converged = False
            # pf_not_converged(time_step, ts_variables)

        ### --- Calculate reward --- ###
        vm_pus = net.res_bus.vm_pu.drop(58) #Remove external grid
        # max_vm_pu = np.max(vm_pus)
        # print(f'Max: {max_vm_pu}, max: {np.min(vm_pus)}, mean: {np.mean(vm_pus)}, sum: {np.sum(vm_pus)}')
        # min_vm_pu = np.min(vm_pus)

        ###First try (bad results). RF #1
        # tollerance = 0.1
        # max_vm_pu_value = 1 + tollerance / 2
        # min_vm_pu_value = 1 - tollerance / 2
        # reward = - alpha * curtailment_percent #- beta*(np.abs(1.05-max_vm_pu)+0.5*np.abs(0.95-min_vm_pu))
        # reward = reward * beta*np.abs(max_vm_pu_value-max_vm_pu) if max_vm_pu_value-max_vm_pu<0 else reward
        # reward = reward * beta*np.abs(min_vm_pu_value-min_vm_pu) if min_vm_pu_value-min_vm_pu>0 else reward
        # if(time_step%2000==0 and time_step>0):
        #     print(f'###DEBUG###\nTime step: {time_step}, \nSum curtailment[%]: {(curtailment_percent):.3f}, \nOvervoltage penalty: {(np.abs(1.05-max_vm_pu)):.4f}({max_vm_pu:.4f}), \nUndervoltage penalty: {(np.abs(0.95-min_vm_pu)):.4f}({min_vm_pu:.4f}), \nTotal reward: {(reward):.3f}')

        ###Second try. RF #2 (it works but too much energy curt when not needed)
        # curtailment_percent = np.sum(action)
        # alpha = 5
        # beta = 1000
        # gamma = 100
        # reward_curt = - alpha * curtailment_percent
        # voltage_violation = np.sum(volatge_violation_function(vm_pus))
        # reward_volt_viol = - beta * voltage_violation
        #
        # is_critial_situation = np.max(vm_pus) > 1.05
        # reward_result = gamma * (vm_pu_labels[time_step] - 2 * is_critial_situation)
        # reward = reward_curt + reward_volt_viol + reward_result
        #Cases:
        # 0 - 0 - no critial situation before and after the agent's action (OK, nothing happens)
        # 0 - 1 - no critial situation before but introduced after the agent's action (WORST POSSIBLE CASE, -2*gamma punishment)
        # 1 - 0 - critial situation before and solved after the agent's action (BEST POSSIBLE CASE, +gamma reward)
        # 1 - 1 - critial situation before and after the agent's action (NOT REALLY OK, -gamma punishment)

        ###Third try. RF #3
        curtailment_percent = np.sum(action_p)
        reacrive_power_changes = np.sum(np.abs(action_q))
        alpha_p = 200      #20 #good resutls (nice plot, lot of unwanted p, lot of uv)
        alpha_q = 100      #2
        beta = 800         #100
        gamma = 100        #200
        reward_p = - alpha_p * curtailment_percent
        reward_q = - alpha_q * reacrive_power_changes

        voltage_violation = np.sum(volatge_violation_function(vm_pus))
        reward_volt_viol = - beta * voltage_violation

        is_critial_situation = np.max(vm_pus) > 1.05 or np.min(vm_pus) < 0.95
        if(vm_pu_labels is not None):
            reward_result = gamma * (vm_pu_labels[time_step] - 4 * is_critial_situation + 1)
            #Cases:
            # 0 - 0 - no critial situation before and after the agent's action (OK, +1gamma reward)
            # 0 - 1 - no critial situation before but introduced after the agent's action (WORST POSSIBLE CASE, -3*gamma punishment)
            # 1 - 0 - critial situation before and solved after the agent's action (BEST POSSIBLE CASE, +2gamma reward)
            # 1 - 1 - critial situation before and after the agent's action (NOT REALLY OK, -2gamma punishment)
        else:
            reward_result = 0
        reward = reward_p + reward_q + reward_volt_viol + reward_result

        # if(time_step%2000==0 and time_step>0):
        #     print(f'###DEBUG###\nTime step: {time_step}, \nSum curtailment[%]: {(curtailment_percent):.3f}, \nVoltage penalty: {volatge_violation:.4f}, \nTotal reward: {(reward):.3f}')

        # rlagent.history.append(reward)
        rlagent.history.append([reward_p, reward_q, reward_volt_viol, reward_result,reward])
        #Keep track of the curtailment as % and the actual value as MW -> [%, kW]
        rlagent.curtailment.append([curtailment_percent, np.sum(next_gen_p * action_p), np.sum(next_gen_p), np.sum(action_q)])


        ### --- Next state --- ###
        next_state = None
        next_state = get_state(ts_variables,time_step+1)

        ### --- Check system condition --- ###
        done = True if (not ctrl_converged or not pf_converged) else False

        ### --- Store experience --- ###
        if(next_state is not None and train):
            rlagent.save_experience(state, action, reward, next_state, done)

        if(train):
            rlagent.learn()
            rlagent.increment_step_counter()
    else:
        control_time_step(ts_variables['controller_order'], time_step)
        try:
            # calls controller init, control steps and run function (runpp usually is called in here)
            run_control_fct(net, ctrl_variables=ts_variables, **kwargs)
        except ControllerNotConverged:
            ctrl_converged = False
            # If controller did not converge do some stuff
            # controller_not_converged(time_step, ts_variables)
        except ts_variables['errors']:
            # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
            pf_converged = False
            # pf_not_converged(time_step, ts_variables)
        done = True if (not ctrl_converged or not pf_converged) else False


    if(save):
        output_writer_fct(net, time_step, pf_converged, ctrl_converged, ts_variables)
    finalize_step(ts_variables['controller_order'], time_step)
    return done


def _check_controller_recyclability(net):
    # if a parameter is set to True here, it will be recalculated during the time series simulation
    recycle = dict(trafo=False, gen=False, bus_pq=False)
    if "controller" not in net:
        # everything can be recycled since no controller is in net. But the time series simulation makes no sense
        # then anyway...
        return recycle

    for idx in net.controller.index:
        # todo: write to controller data frame recycle column instead of using self.recycle of controller instance
        ctrl_recycle = net.controller.at[idx, "recycle"]
        if not isinstance(ctrl_recycle, dict):
            # if one controller has a wrong recycle configuration it is deactived
            recycle = False
            break
        # else check which recycle parameter are set to True
        for rp in ["trafo", "bus_pq", "gen"]:
            recycle[rp] = recycle[rp] or ctrl_recycle[rp]

    return recycle


def _check_output_writer_recyclability(net, recycle):
    if "output_writer" not in net:
        raise ValueError("OutputWriter not defined")
    ow = net.output_writer.at[0, "object"]
    # results which are read with a faster batch function after the time series simulation
    recycle["batch_read"] = list()
    recycle["only_v_results"] = False
    new_log_variables = list()

    for output in ow.log_variables:
        table, variable = output[0], output[1]
        if table not in ["res_bus", "res_line", "res_trafo", "res_trafo3w"] or recycle["trafo"] or len(output) > 2:
            # no fast read of outputs possible if other elements are required as these or tap changer is active
            recycle["only_v_results"] = False
            recycle["batch_read"] = False
            return recycle
        else:
            # fast read is possible
            if variable in ["vm_pu", "va_degree"]:
                new_log_variables.append(('ppc_bus', 'vm'))
                new_log_variables.append(('ppc_bus', 'va'))

            recycle["only_v_results"] = True
            recycle["batch_read"].append((table, variable))

    ow.log_variables = new_log_variables
    ow.log_variable('ppc_bus', 'vm')
    ow.log_variable('ppc_bus', 'va')
    return recycle


def get_recycle_settings(net, **kwargs):
    """
    checks if "run" is specified in kwargs and calls this function in time series loop.
    if "recycle" is in kwargs we use the TimeSeriesRunpp class (not implemented yet)

    INPUT:
        **net** - The pandapower format network

    RETURN:
        **recycle** - a dict with recycle options to be used by runpp
    """

    recycle = kwargs.get("recycle", None)
    if recycle is not False:
        # check if every controller can be recycled and what can be recycled
        recycle = _check_controller_recyclability(net)
        # if still recycle is not None, also check for fast output_writer features
        if recycle is not False:
            recycle = _check_output_writer_recyclability(net, recycle)

    return recycle


def init_time_steps(net, time_steps, **kwargs):
    # initializes time steps if as a range
    if not isinstance(time_steps, Iterable):
        if isinstance(time_steps, tuple):
            time_steps = range(time_steps[0], time_steps[1])
        elif time_steps is None and ("start_step" in kwargs and "stop_step" in kwargs):
            logger.warning("start_step and stop_step are depricated. "
                           "Please use a tuple like time_steps = (start_step, stop_step) instead or a list")
            time_steps = range(kwargs["start_step"], kwargs["stop_step"] + 1)
        else:
            logger.warning("No time steps to calculate are specified. "
                           "I'll check the datasource of the first controller for avaiable time steps")
            ds = net.controller.object.at[0].data_source
            if ds is None:
                raise UserWarning("No time steps are specified and the first controller doesn't have a data source"
                                  "the time steps could be retrieved from")
            else:
                max_timestep = ds.get_time_steps_len()
            time_steps = range(max_timestep)
    return time_steps


def init_time_series(net, time_steps, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    inits the time series calculation
    creates the dict ts_variables, which includes necessary variables for the time series / control function

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

    OPTIONAL:

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or logger debug messages
    """

    time_steps = init_time_steps(net, time_steps, **kwargs)

    init_default_outputwriter(net, time_steps, **kwargs)
    # get run function
    run = kwargs.pop("run", pp.runpp)
    recycle_options = None
    if hasattr(run, "__name__") and run.__name__ == "runpp":
        # use faster runpp options if possible
        recycle_options = get_recycle_settings(net, **kwargs)

    init_output_writer(net, time_steps)
    # as base take everything considered when preparing run_control
    ts_variables = prepare_run_ctrl(net, None, run=run, **kwargs)
    # recycle options, which define what can be recycled
    ts_variables["recycle_options"] = recycle_options
    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged run is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence
    # print settings
    ts_variables["verbose"] = verbose

    if logger.level != 10 and verbose:
        # simple progress bar
        ts_variables['progress_bar'] = tqdm.tqdm(total=len(time_steps))

    return ts_variables


def cleanup(net, ts_variables):
    if isinstance(ts_variables["recycle_options"], dict):
        # Todo: delete internal variables and dumped results which are not needed
        net._ppc = None  # remove _ppc because if recycle == True and a new timeseries calculation is started with a different setup (in_service of lines or trafos, open switches etc.) it can lead to a disaster


def print_progress(i, time_step, time_steps, verbose, **kwargs):
    # simple status print in each time step.
    if logger.level != 10 and verbose:
        kwargs['ts_variables']["progress_bar"].update(1)

    # print debug info
    if logger.level == pplog.DEBUG and verbose:
        logger.debug("run time step %i" % time_step)

    # call a custom progress function
    if "progress_function" in kwargs:
        func = kwargs["progress_function"]
        func(i, time_step, time_steps, **kwargs)


def run_loop(net, ts_variables, rl=None, run_control_fct=run_control, output_writer_fct=_call_output_writer, **kwargs):
    """
    runs the time series loop which calls pp.runpp (or another run function) in each iteration

    Parameters
    ----------
    net - pandapower net
    ts_variables - settings for time series

    """
    for i, time_step in enumerate(ts_variables["time_steps"]):
        print_progress(i, time_step, ts_variables["time_steps"], ts_variables["verbose"], ts_variables=ts_variables,
                       **kwargs)
        if(rl is None):
            run_time_step(net, time_step, ts_variables, run_control_fct, output_writer_fct, **kwargs)
        else:
            done = run_time_step_rl(net, time_step, rl, ts_variables, run_control_fct, output_writer_fct, **kwargs)
            if(done):
                print(f'--- Diverged at time step: {time_step}. Starting a new episode  ---')
                break
        # if(i>4):
        #     break
        # print()
    # print(i, time_step, ts_variables["time_steps"])


def run_timeseries(net, time_steps=None, rl=None, continue_on_divergence=False, verbose=True, **kwargs):
    """
    Time Series main function

    Runs multiple PANDAPOWER AC power flows based on time series which are stored in a **DataSource** inside
    **Controllers**. Optionally other functions than the pp power flow can be called by setting the run function in kwargs

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or if logger.level == Debug it prints debug messages

        **kwargs** - Keyword arguments for run_control and runpp. If "run" is in kwargs the default call to runpp()
        is replaced by the function kwargs["run"]
    """

    ts_variables = init_time_series(net, time_steps, continue_on_divergence, verbose, **kwargs)

    # cleanup ppc before first time step
    cleanup(net, ts_variables)

    control_diagnostic(net)
    run_loop(net, ts_variables, rl, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(net, ts_variables)
    # both cleanups, at the start AND at the end, are important!
