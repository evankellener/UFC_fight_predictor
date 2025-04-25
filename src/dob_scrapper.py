import requests
from bs4 import BeautifulSoup
import re

class DOBScraper:
    def __init__(self):
        self.base_url = "https://www.sherdog.com"
        self.search_url = f"{self.base_url}/stats/fightfinder?SearchTxt="
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def _format_fighter_name(self, name):
        """Format fighter name for the search URL."""
        return name.replace(" ", "+").upper()

    def _get_fighter_profile_url(self, fighter_name):
        """Search for the fighter and retrieve their profile URL."""
        formatted_name = self._format_fighter_name(fighter_name)
        search_url = f"{self.search_url}{formatted_name}"
        print(f"Fetching search URL: {search_url}")  # Debugging

        try:
            response = requests.get(search_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch search results for {fighter_name}: HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.content, "html.parser")
            fighter_link = soup.select_one("tr[onclick^=\"document.location=\"]")
            if fighter_link:
                onclick_attr = fighter_link["onclick"]
                match = re.search(r"'(.*?)'", onclick_attr)
                if match:
                    profile_url = f"{self.base_url}{match.group(1)}"
                    print(f"Found profile URL for {fighter_name}: {profile_url}")  # Debugging
                    return profile_url

            print(f"No profile found for {fighter_name}")
            return None
        except Exception as e:
            print(f"Error fetching fighter profile URL for {fighter_name}: {e}")
            return None

    def _get_fighter_dob(self, profile_url):
        """Extract the date of birth from the fighter's profile page."""
        print(f"Fetching profile page: {profile_url}")  # Debugging
        try:
            response = requests.get(profile_url, headers=self.headers)
            if response.status_code != 200:
                print(f"Failed to fetch profile page: {profile_url}: HTTP {response.status_code}")
                return "Date of birth not found"

            soup = BeautifulSoup(response.content, "html.parser")
            dob_element = soup.select_one("span[itemprop='birthDate']")
            if dob_element:
                dob = dob_element.text.strip()
                print(f"Extracted DOB: {dob}")  # Debugging
                return dob

            print("Date of birth element not found on profile page.")
            return "Date of birth not found"
        except Exception as e:
            print(f"Error fetching DOB from profile URL: {e}")
            return "Date of birth not found"

    def scrape_dob(self, fighter_name):
        """Public method to scrape DOB for a given fighter."""
        profile_url = self._get_fighter_profile_url(fighter_name)
        if not profile_url:
            print(f"Invalid or missing profile URL for fighter: {fighter_name}")
            return "Date of birth not found"
        return self._get_fighter_dob(profile_url)