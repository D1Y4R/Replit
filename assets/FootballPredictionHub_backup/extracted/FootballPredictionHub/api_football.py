import requests
import logging

class FootballDataAPI:
    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key=None):
        self.api_key = api_key or "668dd03e0aea41b58fce760cdf4eddc8"
        self.headers = {
            "X-Auth-Token": self.api_key
        }

    def get_fixtures(self, date=None, league=None, team=None, season=None, unfold_goals=False):
        """Get matches/fixtures based on various parameters"""
        endpoint = f"{self.BASE_URL}/matches"
        params = {}

        if date:
            params["date"] = date
        if league:
            params["competitions"] = league
        if team:
            # If team is provided, switch to team matches endpoint
            endpoint = f"{self.BASE_URL}/teams/{team}/matches"
        if season:
            params["season"] = season

        return self._make_request(endpoint, params, unfold_goals=unfold_goals)

    def get_competitions(self):
        """Get all available competitions"""
        endpoint = f"{self.BASE_URL}/competitions"
        return self._make_request(endpoint)

    def get_competition_standings(self, competition_id, season=None):
        """Get standings for a particular competition"""
        endpoint = f"{self.BASE_URL}/competitions/{competition_id}/standings"
        params = {}
        if season:
            params["season"] = season
        return self._make_request(endpoint, params)

    def get_team_info(self, team_id):
        """Get information about a specific team"""
        endpoint = f"{self.BASE_URL}/teams/{team_id}"
        return self._make_request(endpoint)

    def get_match_info(self, match_id, unfold_goals=False):
        """Get detailed information about a match"""
        endpoint = f"{self.BASE_URL}/matches/{match_id}"
        return self._make_request(endpoint, unfold_goals=unfold_goals)

    def get_scorers(self, competition_id, limit=10):
        """Get top scorers for a competition"""
        endpoint = f"{self.BASE_URL}/competitions/{competition_id}/scorers"
        params = {"limit": limit}
        return self._make_request(endpoint, params)

    def _make_request(self, endpoint, params=None, unfold_goals=False):
        """Make API request with proper error handling"""
        try:
            headers = self.headers.copy()
            if unfold_goals:
                headers["X-Unfold-Goals"] = "true"

            logging.debug(f"API request to: {endpoint} with params: {params}")
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            result = response.json()
            logging.debug(f"API response status: {response.status_code}")
            return result
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {str(e)}")
            return {"error": str(e), "message": "API request failed"}