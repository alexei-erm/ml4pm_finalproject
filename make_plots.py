import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer


def make_df(
    event_mux: EventMultiplexer,
    experiment_names: dict[str, str],
    tag: str,
    truncate: int | None = None,
) -> pd.DataFrame:
    dfs = []
    for run, tags in event_mux.Runs().items():
        experiment_name = run.split("/")[0]
        if experiment_name in experiment_names.keys() and tag in tags["scalars"]:
            scalars = event_mux.Scalars(run, tag)
            df = pd.DataFrame(scalars, columns=["step", "value"])

            if truncate is not None and len(df) > truncate:
                df.drop(df.tail(len(df) - truncate).index, inplace=True)

            df["experiment"] = experiment_names[experiment_name]
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def make_training_plot(
    event_mux: EventMultiplexer,
    experiment_names: dict[str, str],
    tag: str,
    title: str,
    ylabel: str,
    file_name: str,
    yrange: tuple[float, float] | None = None,
    legend_loc: str = "upper left",
    trunc_step: int | None = None,
) -> None:

    df = make_df(event_mux=event_mux, experiment_names=experiment_names, tag=tag, truncate=trunc_step)

    def errorbar(x: pd.Series):
        mean = x.mean()
        std = x.std()
        return (max(mean - std, 0), mean + std)

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df,
        x="step",
        y="value",
        hue="experiment",
        hue_order=list(experiment_names.values()),
        errorbar=errorbar,
        ax=ax,
    )
    if len(experiment_names) > 1:
        ax.legend(loc=legend_loc, shadow=True, framealpha=1.0)
    else:
        ax.get_legend().remove()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(
        ymin=yrange[0] - 0.05 * (yrange[1] - yrange[0]) if yrange is not None else None,
        ymax=yrange[1] + 0.05 * (yrange[1] - yrange[0]) if yrange is not None else None,
    )
    fig.tight_layout(pad=0.5)

    file_path = os.path.abspath(file_name)
    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    fig.savefig(file_path)


def main() -> None:
    event_mux = EventMultiplexer().AddRunsFromDirectory("logs")
    event_mux.Reload()

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)

    make_training_plot(
        event_mux,
        experiment_names={
            "LSTMAE_VG5_turbine_equilibrium_stat_coil_ph01_01_tmp": "LSTMAE_VG5_turbine_equilibrium_stat_coil_ph01_01_tmp",
            "LSTMAE_VG5_turbine_equilibrium": "LSTMAE_VG5_turbine_equilibrium",
        },
        tag="Loss/Training",
        title="Training loss",
        ylabel="Loss",
        legend_loc="upper right",
        file_name="plots/loss.png",
    )


if __name__ == "__main__":
    main()
