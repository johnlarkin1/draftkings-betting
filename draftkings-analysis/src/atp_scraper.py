"""
This module targets the AtpTour Singles Ranking page to automatically build
out a List[PlayerDetails] from the top 200 players. 

This will be used for detecting if a bet is related to tennis (based on the assumption 
that most bets are inclusive of a Top 200 player).
"""


import logging
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag

REQUEST_TIMEOUT = 5  # seconds


@dataclass
class PlayerDetails:
    """A dataclass to represent player information"""

    ranking: str
    full_name: str
    first_name: str
    last_name: str
    country: str


class AtpScraper:
    """A web scraper to inspect the Top 200 ATP players"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        self.base_url = "https://www.atptour.com/en/rankings/singles"

    def get_top_players(
        self, start_range: int = 1, end_range: int = 200
    ) -> list[PlayerDetails]:
        """Returns a list of the players from start_range to end_range"""
        # pylint: disable=too-many-locals
        target_url = (
            f"{self.base_url}?countryCode=all&rankRange={start_range}-{end_range}"
        )

        players: list[PlayerDetails] = []
        logging.info("request being sent to target url %r", target_url)
        response: Optional[requests.models.Response] = None
        try:
            response = requests.get(target_url, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.Timeout as err:
            logging.error(
                "timeout received from target url %r err: %r", target_url, err
            )
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="mega-table")
        assert isinstance(table, Tag)
        table_body = table.tbody
        assert table_body, "couldn't find table body from ATP"
        for row in table_body.find_all("tr"):
            player_rank = row.find("td", class_="rank-cell").text.strip()
            player_name = row.find("td", class_="player-cell").text.strip()
            player_country_div = row.find("div", class_="country-item")
            player_country_alt: str = player_country_div.find("img", alt=True)["alt"]
            name_split = player_name.split(" ")
            first_name, last_name = (
                name_split[0],
                " ".join(name_split[1:]) if len(name_split) > 1 else "",
            )
            players.append(
                PlayerDetails(
                    player_rank,
                    player_name,
                    first_name,
                    last_name,
                    player_country_alt.lower(),
                )
            )
        return players


if __name__ == "__main__":
    scraper = AtpScraper()
    print(scraper.get_top_players())
