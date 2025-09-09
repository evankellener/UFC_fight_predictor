import requests
import pandas as pd

def get_event_odds(event_pk):
    url = "https://api.fightodds.io/gql"

    headers = {
        "Content-Type": "application/json",
        "Origin": "https://fightodds.io",
        "Referer": "https://fightodds.io/",
        "User-Agent": "Mozilla/5.0"
    }

    query = """
    query EventOddsQuery($eventPk: Int!) {
      sportsbooks: allSportsbooks(hasOdds: true) {
        edges {
          node {
            id
            shortName
            slug
          }
        }
      }
      eventOfferTable(pk: $eventPk) {
        name
        pk
        fightOffers {
          edges {
            node {
              id
              fighter1 {
                firstName
                lastName
              }
              fighter2 {
                firstName
                lastName
              }
              bestOdds1
              bestOdds2
              straightOffers {
                edges {
                  node {
                    sportsbook {
                      shortName
                    }
                    outcome1 {
                      odds
                    }
                    outcome2 {
                      odds
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "eventPk": event_pk
    }

    payload = {
        "query": query,
        "variables": variables
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"❌ Status: {response.status_code}\n{response.text}")
        return pd.DataFrame()

    try:
        json_data = response.json()
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return pd.DataFrame()

    fights = json_data["data"]["eventOfferTable"]["fightOffers"]["edges"]

    results = []
    for fight in fights:
        node = fight["node"]
        f1 = f"{node['fighter1']['firstName']} {node['fighter1']['lastName']}"
        f2 = f"{node['fighter2']['firstName']} {node['fighter2']['lastName']}"
        best1 = node.get("bestOdds1")
        best2 = node.get("bestOdds2")

        offer_edges = node.get("straightOffers", {}).get("edges", [])
        offers = []
        for offer in offer_edges:
            sportsbook = offer["node"]["sportsbook"]["shortName"]
            odds1 = offer["node"]["outcome1"]["odds"]
            odds2 = offer["node"]["outcome2"]["odds"]
            offers.append((sportsbook, odds1, odds2))

        results.append({
            "Fighter 1": f1,
            "Fighter 2": f2,
            "Best Odds 1": best1,
            "Best Odds 2": best2,
            "All Sportsbooks": offers
        })

    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    event_pk = 6068  # 🔁 Replace with other eventPk IDs to scrape other cards
    df = get_event_odds(event_pk)
    pd.set_option("display.max_colwidth", None)
    print(df)
