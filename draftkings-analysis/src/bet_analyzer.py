"""
This module is going to prompt the user to load an export from the
Draftkings Chrome Extension, and then infer some results, do some slight
aggregation, and generate (hopefully) helpful graphs to help the user tighten
up their sports betting and find more edge.
"""


import csv
import logging
from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

from atp_scraper import AtpScraper, PlayerDetails


def init_logger() -> logging.Logger:
    """Initializes a custom logger to use for this module"""
    logging.basicConfig()
    custom_logger = logging.getLogger("bet_analyzer")
    custom_logger.setLevel(logging.DEBUG)
    return custom_logger


logger = init_logger()

DATETIME_FMT = "%b %d %Y %I:%M:%S %p"


class Sports(Enum):
    """Sports that we can detect"""

    TENNIS = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return str(self.name).lower()


@dataclass
class DataRow:
    """A class to represent a single row of our exported csv"""

    title: str
    odds: float
    market_type: str
    status: str
    stake: str
    returns: str
    datetime: datetime


class DraftkingsReader:
    """This class is going to read, infer, aggregate, and allow access
    to core DataFrames to use for visualizations"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, filename: str) -> None:
        self._early_return = False
        self._filename = filename
        self._aggregated_data: Optional[list[DataRow]] = None
        self._core_dataframe: Optional[pd.DataFrame] = None
        self._agg_dataframe_1d: Optional[pd.DataFrame] = None
        self._agg_dataframe_7d: Optional[pd.DataFrame] = None
        self._agg_status_dataframe_7d: Optional[pd.DataFrame] = None
        self._atp_player_details: list[PlayerDetails] = AtpScraper().get_top_players()

        self._steps: list[Callable[[], None]] = [
            self._read,
            self._get_dataframe,
            self._clean_currency,
            self._detect_and_add_sport,
            self._aggregate_data,
        ]

    @property
    def filename(self):
        """The filename that was loaded"""
        return self._filename

    @property
    def data(self) -> pd.DataFrame:
        """The core unfiltered and unaggregated data"""
        if self._core_dataframe is None:
            self._get_dataframe()
        return self._core_dataframe

    @property
    def agg_1d_data(self) -> pd.DataFrame:
        """Aggregated 1 day data"""
        if self._agg_dataframe_1d is None:
            self.load_and_analyze()
        return self._agg_dataframe_1d

    @property
    def agg_7d_data(self) -> pd.DataFrame:
        """Aggregated 7 day data"""
        if self._agg_dataframe_7d is None:
            self.load_and_analyze()
        return self._agg_dataframe_7d

    @property
    def agg_status_7d_data(self) -> pd.DataFrame:
        """Aggregated 7 day data by status"""
        if self._agg_status_dataframe_7d is None:
            self.load_and_analyze()
        return self._agg_status_dataframe_7d

    def _parse_row(self, row: list[str]) -> DataRow:
        assert len(row) >= len(
            fields(DataRow)
        ), "our CSV row should be able to populate our internal DataRow"
        return DataRow(
            row[0],
            float(row[1]),
            row[2],
            row[3],
            row[4],
            row[5],
            datetime.strptime(row[6] + row[7] + row[8], DATETIME_FMT),
        )

    def _read(self) -> None:
        if self._aggregated_data:
            return
        aggregated_data: list[DataRow] = []
        with open(self._filename, newline="", encoding="utf-8") as csvfile:
            data = csv.reader(csvfile)
            next(data, None)  # skip header
            for row in data:
                aggregated_data.append(self._parse_row(row))

        self._aggregated_data = aggregated_data

    def _get_dataframe(self) -> None:
        if self._aggregated_data is None:
            self._read()
        assert self._aggregated_data
        initial_dataframe = pd.DataFrame.from_records(
            [vars(row) for row in self._aggregated_data]
        )
        self._core_dataframe = initial_dataframe

    def _clean_currency(self) -> None:
        assert self._core_dataframe is not None
        transformed_df = self._core_dataframe

        def clean_currency(item: str) -> str:
            if isinstance(item, str):
                return item.replace("$", "").replace(",", "")
            return item

        transformed_df["stake"] = (
            transformed_df["stake"].apply(clean_currency).astype("float")
        )
        transformed_df["returns"] = (
            transformed_df["returns"].apply(clean_currency).astype("float")
        )
        self._core_dataframe = transformed_df

    def _detect_and_add_sport(self) -> None:
        """This method will attempt to detect the sport that the bet was made in.
        As of right now, it's going to focus on tennis, given that's the majority of the bets I make.
        """
        assert self._core_dataframe is not None
        transformed_df = self._core_dataframe
        last_names = set(player.last_name for player in self._atp_player_details)

        def any_tennis_indicators(row: dict) -> str:
            bet_title = row["title"]
            bet_title_tokens = bet_title.split(" ")
            if any(
                bet_title_token in last_names for bet_title_token in bet_title_tokens
            ):
                return str(Sports.TENNIS)
            return str(Sports.UNKNOWN)

        transformed_df["sport"] = transformed_df.apply(any_tennis_indicators, axis=1)
        return transformed_df

    def _aggregate_data(self) -> None:
        assert self._core_dataframe is not None
        transformed_df = self._core_dataframe

        def combine_col(tup: Tuple[str, str]) -> str:
            if "" in tup:
                return "".join(tup)
            return "_".join(tup)

        raw_time_grouping_dict = {
            "odds": ["mean", "median"],
            "stake": ["mean", "median", "max", "min"],
            "returns": ["mean", "median", "max", "min"],
            "status": ["count"],
        }

        overall_group = (
            transformed_df.agg(raw_time_grouping_dict).reset_index().fillna(0)
        )
        logger.info("overall aggregate breakdown: \n%s", overall_group.head())

        grouped_daily = transformed_df.groupby([pd.Grouper(key="datetime", freq="D")])
        grouped_daily = (
            grouped_daily.agg(raw_time_grouping_dict).reset_index().fillna(0)
        )
        grouped_daily.columns = grouped_daily.columns.map(combine_col)
        self._agg_dataframe_1d = grouped_daily

        grouped_weekly = transformed_df.groupby(
            [pd.Grouper(key="datetime", freq="W-MON")]
        )
        grouped_weekly = (
            grouped_weekly.agg(raw_time_grouping_dict).reset_index().fillna(0)
        )
        grouped_weekly.columns = grouped_weekly.columns.map(combine_col)
        self._agg_dataframe_7d = grouped_weekly

        grouped_weekly_by_status = transformed_df.groupby(
            ["status", pd.Grouper(key="datetime", freq="W-MON")]
        )
        grouped_weekly_by_status = (
            grouped_weekly_by_status.size().reset_index(name="counts").fillna(0)
        )
        grouped_weekly_by_status.columns = grouped_weekly_by_status.columns.map(
            combine_col
        )
        self._agg_status_dataframe_7d = grouped_weekly_by_status

    def load_and_analyze(self) -> None:
        """The main entrypoint for this class. Loads the file and performs all necessary steps"""
        for step in self._steps:
            if self._early_return:
                return
            step()
        logger.info("done loading and analyzing")


class DraftkingsGrapher:
    """This class is going to use the data from Draftkings to generate helpful charts"""

    def __init__(
        self,
        draftkings_data: pd.DataFrame,
        agg_1d_data: pd.DataFrame,
        agg_7d_data: pd.DataFrame,
    ) -> None:
        self._early_return = False
        self._curr_figure = 1
        self._data = draftkings_data
        self._agg_1d_data = agg_1d_data
        self._agg_7d_data = agg_7d_data
        self._generate_steps: list[Callable[[], None]] = [
            self._setup_defaults,
            self._odds_by_datetime,
            self._cleansed_odds_by_datetime,
            self._stake_by_datetime,
            self._returns_by_datetime,
            self._returns_by_odds,
            self._status_count_bar,
            self._status_breakdown_bar,
            self._profit_by_week_bar,
            self._avg_wager_over_week,
            self._odds_range_breakdown_bar,
            self._ideal_kelly_criterion_scatter,
            self._show_all,
        ]

    @property
    def data(self) -> pd.DataFrame:
        """Gets the core data"""
        return self._data

    @property
    def agg_1d_data(self) -> pd.DataFrame:
        """Gets the 1d aggregated data"""
        return self._agg_1d_data

    @property
    def agg_7d_data(self) -> pd.DataFrame:
        """Gets the 7d aggregated data"""
        return self._agg_7d_data

    def _get_new_figure(self, title: str) -> Tuple[Any, Any]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_title(label=title, fontdict={"fontsize": 14})
        self._curr_figure += 1
        return fig, ax

    def _setup_defaults(self) -> None:
        plt.close("all")
        sns.set_style("darkgrid")

    def _odds_by_datetime(self) -> None:
        _, ax = self._get_new_figure("Odds Bet over Time")
        sns.scatterplot(ax=ax, data=self.data, x="datetime", y="odds")

    def _cleansed_odds_by_datetime(self) -> None:
        _, ax = self._get_new_figure("Filtered Odds Bet over Time")
        avg_odds = self.data["odds"].mean()
        med_odds = self.data["odds"].median()
        std_odds = self.data["odds"].std()
        logger.info("avg_odds: %s", avg_odds)
        logger.info("med_odds: %s", med_odds)
        logger.info("std_odds: %s", std_odds)
        # Going to use the median here, so that we can filter for outliers
        num_std_dev_cutoff = 1
        high_thresh = med_odds + std_odds * num_std_dev_cutoff
        low_thresh = med_odds - std_odds * num_std_dev_cutoff

        # Filter
        cleansed_odds = self.data[self.data["odds"].between(low_thresh, high_thresh)]
        sns.scatterplot(ax=ax, data=cleansed_odds, x="datetime", y="odds")

    def _stake_by_datetime(self) -> None:
        _, ax = self._get_new_figure("Stake Bet over Time")
        sns.scatterplot(ax=ax, data=self.data, x="datetime", y="stake")

    def _returns_by_datetime(self) -> None:
        _, ax = self._get_new_figure("Returns over Time")
        sns.scatterplot(ax=ax, data=self.data, x="datetime", y="returns")

    def _returns_by_odds(self) -> None:
        _, ax = self._get_new_figure("Returns by Odds (filtered)")
        cleansed_odds = self.data[self.data["odds"].between(-500, 500)]
        sns.regplot(ax=ax, data=cleansed_odds, x="odds", y="returns")

    def _status_count_bar(self) -> None:
        _, ax = self._get_new_figure("Bet Status Breakdown by Type")
        status_count = (
            self.data.groupby("status")["status"].count().reset_index(name="cnt")
        )
        status_count["status"] = status_count["status"].str.lower()
        barplot = sns.barplot(ax=ax, data=status_count, x="status", y="cnt")
        for _, row in status_count.iterrows():
            barplot.text(
                row.name, row.cnt, round(row.cnt, 2), color="black", ha="center"
            )

    def _status_breakdown_bar(self) -> None:
        _, ax = self._get_new_figure("Bet Status Returns by Type")
        status_breakdown = self.data.groupby("status").agg(np.sum).reset_index()
        status_breakdown["status"] = status_breakdown["status"].str.lower()
        barplot = sns.barplot(ax=ax, data=status_breakdown, x="status", y="returns")
        for _, row in status_breakdown.iterrows():
            barplot.text(
                row.name, row.returns, round(row.returns, 2), color="black", ha="center"
            )

    def _profit_by_week_bar(self) -> None:
        """
        graph type: vertical bar chart
        x-axis: week breakdown
        y-axis: profit per that week
        """
        fig, ax = self._get_new_figure("Returns by Week")
        sns.barplot(ax=ax, data=self.agg_7d_data, x="datetime", y="returns_mean")
        ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])
        fig.autofmt_xdate()

    def _avg_wager_over_week(self) -> None:
        """
        graph type: vertical bar chart
        x-axis: week breakdown
        y-axis: wager size per that week
        """
        fig, ax = self._get_new_figure("Stake by Week")
        sns.barplot(ax=ax, data=self.agg_7d_data, x="datetime", y="stake_median")
        ax.set_xticklabels([t.get_text().split("T")[0] for t in ax.get_xticklabels()])
        fig.autofmt_xdate()

    def _odds_range_breakdown_bar(self) -> None:
        """
        graph type: stacked horizontal bar chart
        x-axis: raw count of won and loss
        y-axis: bucketed odds range i.e. list(range(-500, 500, 100))
        """
        _, ax = self._get_new_figure("Odds Range Breakdown")
        data_to_group = self.data
        grouped_by_odds = (
            data_to_group.groupby(
                ["status", pd.cut(data_to_group["odds"], np.arange(-500, 500, 100))]
            )
            .size()
            .reset_index(name="cnt")
        )
        w_grouped_by_odds = grouped_by_odds[grouped_by_odds["status"] == "WON"]
        l_grouped_by_odds = grouped_by_odds[grouped_by_odds["status"] == "LOST"]
        w_or_l_merged_df = pd.merge(
            w_grouped_by_odds, l_grouped_by_odds, on="odds", suffixes=("_won", "_lost")
        )
        w_or_l_merged_df["total_count"] = (
            w_or_l_merged_df["cnt_won"] + w_or_l_merged_df["cnt_lost"]
        )

        # # Plot the total (call it won bc lost will overlap)
        sns.set_color_codes("pastel")
        sns.barplot(
            x="total_count", y="odds", data=w_or_l_merged_df, label="won", color="b"
        )

        # # Plot the total lost
        sns.set_color_codes("muted")
        sns.barplot(
            x="cnt_lost", y="odds", data=w_or_l_merged_df, label="lost", color="b"
        )

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="lower right", frameon=True)
        ax.set(ylabel="Odds Range", xlabel="Breakdown")
        sns.despine(left=True, bottom=True)

    def _ideal_kelly_criterion_scatter(self) -> None:
        """
        Kelly Criterion is:
        
        f* = p - (1 - p) / b

        Where
         f* - the fraction of the bankroll to wager 
         b - the proportion of the bet gained
         p - the probability of the bet winning

        We really want to look at p vs b. I.e. breakdown the 
        probability of the bet winning versus the proportion of the bet gained 
        and then fit a best fit line to that.
        """

        bucketing_ranges = [10, 50, 100]
        
        def plot_impl(bucketing_range: int) -> None:
            _, ax = \
                self._get_new_figure(f"Proportion of Bet Gained (Bucket: {bucketing_range}) vs Prob of Bet winning")

            # Computing prob of bet winning means computing for a given 
            # american odds, how often did that bet hit or miss
            # we should bucket into sizes of american odds of ten
            data_to_group = self.data
            # Bucket
            grouped_by_odds = (
                data_to_group.groupby(
                    ["status", pd.cut(data_to_group["odds"], np.arange(-500, 500, bucketing_range))]
                )
                .size()
                .reset_index(name="cnt")
            )
            
            w_grouped_by_odds = grouped_by_odds[grouped_by_odds["status"] == "WON"]
            l_grouped_by_odds = grouped_by_odds[grouped_by_odds["status"] == "LOST"]
            w_or_l_merged_df = pd.merge(
                w_grouped_by_odds, l_grouped_by_odds, on="odds", suffixes=("_won", "_lost")
            )
            w_or_l_merged_df["total_count"] = (
                w_or_l_merged_df["cnt_won"] + w_or_l_merged_df["cnt_lost"]
            )
            w_or_l_merged_df["prob_winning"] = w_or_l_merged_df["cnt_won"] / w_or_l_merged_df["total_count"]
            w_or_l_merged_df.dropna(inplace=True)
            w_or_l_merged_df["odd_center"] = w_or_l_merged_df["odds"].apply(lambda x: x.mid)
            w_or_l_merged_df.odds = w_or_l_merged_df.odds.astype(str)

            bestfitplot = sns.regplot(ax=ax, x="odd_center", y="prob_winning", data=w_or_l_merged_df)

            # Calculate best fit slope and intercept
            slope, intercept, _, _, _ = linregress(
                x=bestfitplot.get_lines()[0].get_xdata(),
                y=bestfitplot.get_lines()[0].get_ydata()
            )

            _, _, actual_r, _, _ = linregress(
                x=w_or_l_merged_df["odd_center"],
                y=w_or_l_merged_df["prob_winning"],
            )

            # Add regression equation
            bestfitplot.text(-400, 0.2, 'y = ' + str(round(intercept,3)) + ' + ' + str(round(slope,3)) + 'x')
            bestfitplot.text(-400, 0.1, 'R^2 = ' + str(round(actual_r**2, 3)))

        for bucketing_range in bucketing_ranges:
            plot_impl(bucketing_range)

    def _show_all(self) -> None:
        plt.show(block=True)

    def generate_graphs(self) -> None:
        """The main entrypoint for this class. Generates all appropriate graphs"""
        for generate_step in self._generate_steps:
            if self._early_return:
                return
            generate_step()


if __name__ == "__main__":
    # Example: /Users/johnlarkin/Downloads/DraftKings_03_31_23.csv
    filepath = input("Specify fullpath to Dkng Chrome Extension: ")
    reader = DraftkingsReader(filepath)
    reader.load_and_analyze()

    grapher = DraftkingsGrapher(reader.data, reader.agg_1d_data, reader.agg_7d_data)
    grapher.generate_graphs()
