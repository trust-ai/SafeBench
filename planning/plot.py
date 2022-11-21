import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

FIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "figure")

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data_in,
              xaxis='Epoch',
              value="EpRet",
              condition="Condition1",
              smooth=1,
              **kwargs):
    data = []
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data_in:
            if value not in datum:
                value = value.split("Test")[-1]
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
            data.append(datum[[xaxis, value, condition, "cost_limit"]])

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, join="outer")
    sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd')

    if data["cost_limit"][0] is not None and (value == "EpCost"
                                              or value == "TestEpCost"):
        plt.axhline(y=data["cost_limit"][0],
                    linestyle="--",
                    label="cost limit",
                    linewidth=2.5)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    #plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})
    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def plot_cost(data_in, smooth=5, thres=16 * 1e5, **kwargs):
    xaxis = 'TotalEnvInteracts'
    value = "EpCostSmoothed"
    condition = "Condition1"
    data = []
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for i, datum in enumerate(data_in):
            if "TestEpCost" in datum:
                x = np.asarray(datum["TestEpCost"])
            else:
                x = np.asarray(datum["EpCost"])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x
            datum = datum[datum["TotalEnvInteracts"] > thres]
            data.append(datum[[xaxis, value, condition, "cost_limit"]])

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True, join="outer")

    data = data.replace(
        {"Condition1": {
            "ppo_lagrangian": "ppo_lag",
            "trpo_lagrangian": "trpo_lag"
        }})

    sns.set(style="darkgrid", font_scale=3.5)
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    # Draw a nested boxplot to show bills by day and time
    # g = sns.violinplot(x="Condition1", y="EpCostSmoothed", width=0.7, scale="width",
    #             data=data_cat,)
    g = sns.boxplot(
        x="Condition1",
        y="EpCostSmoothed",
        width=0.7,
        data=data,
    )
    # g.set(ylim=(0, 80))
    g.set_yticklabels(g.get_yticks(), size=15)
    x_ticks = g.get_xticklabels()

    print(x_ticks)
    g.set_xticklabels(x_ticks, size=15)

    plt.axhline(y=data["cost_limit"][0],
                linestyle="--",
                label="cost limit",
                color='m',
                linewidth=2.5)
    # sns.despine(offset=10, trim=True)
    plt.legend(loc='best', title=None, frameon=True, fontsize='large')
    plt.xlabel("Methods", labelpad=5, fontsize='large')
    plt.ylabel("Episode Costs", labelpad=5, fontsize='large')
    # xscale = np.max(np.asarray(data[xaxis])) > 5e3
    # if xscale:
    #     # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            cost_limit = None
            env = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'env' in config:
                    env = config['env']
                if 'exp_name' in config:
                    exp_name = config['exp_name']

                if "cost_lim" in config:
                    # Safety gym starter agent
                    cost_limit = config["cost_lim"]
                elif "policy" in config:
                    # our agent
                    policy_name = config["policy"]
                    if "cost_limit" in config[policy_name]:
                        cost_limit = config[policy_name]["cost_limit"]
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
                exp_data = exp_data.rename(columns=lambda x: x.split("/")[-1])
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet' if 'AverageEpRet' in exp_data else 'EpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'cost_limit', cost_limit)
            exp_data.insert(len(exp_data.columns), 'env', env)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 

        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x: osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir = os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs,
               legend=None,
               xaxis=None,
               values=None,
               count=False,
               thres=16 * 1e5,
               font_scale=1.5,
               smooth=1,
               select=None,
               exclude=None,
               estimator='mean'):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np,
                        estimator)  # choose what to show on main curve: mean? max? min?
    env = ""                    
    for d in data:
        if d["env"][0] is not None:
            env = d["env"][0]
            break
    cost_lim = data[0]["cost_limit"][0]
    for value in values:
        plt.figure(figsize=(8, 6))
        plot_data(data,
                  xaxis=xaxis,
                  value=value,
                  condition=condition,
                  smooth=smooth,
                  estimator=estimator)

        name = env + '_' + str(cost_lim) + '_' + value + '.png'
        plt.savefig(osp.join(FIG_DIR, name))
    # plt.show()

    # plot the converged cost box plot
    plt.figure(figsize=(8, 6))
    plot_cost(data, smooth=smooth, thres=thres)
    name = env + '_' + str(cost_lim) + '_' + "converge" + '.png'
    plt.savefig(osp.join(FIG_DIR, name))
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--thres', '-t', default=1600000)
    parser.add_argument('--value', '-y', default=['Performance', 'EpCost'], nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=2)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir,
               args.legend,
               args.xaxis,
               args.value,
               args.count,
               args.thres,
               smooth=args.smooth,
               select=args.select,
               exclude=args.exclude,
               estimator=args.est)


if __name__ == "__main__":
    main()
